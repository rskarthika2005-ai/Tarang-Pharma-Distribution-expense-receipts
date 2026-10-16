import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from modules.analysis import load_claims, parse_amount, parse_date
from modules.validation import count_read_problems, validate_receipt
from source import config, llm_client

RESULT_COLUMNS = [
    "claim_id", "split", "image_file", "ocr_status",
    "vendor", "receipt_date", "receipt_number", "gstin", "total_amount",
    "date_as_printed", "validation_flags", "read_pass", "model", "final_model",
    "conf_vendor", "conf_date", "conf_number", "conf_gstin", "conf_amount",
    "photo_rotated", "handwritten_notes", "from_cache",
    "prompt_tokens", "completion_tokens",
    "esc_prompt_tokens", "esc_completion_tokens", "error_message",
    "image_hash", "business_type", "vehicle_no", "litres", "guest_name", "gst_tax_invoice",
    "rate_per_night_before_gst", "nights", "driver_receipt", "looks_altered", "altered_reason",
    "ev_prompt_tokens", "ev_completion_tokens",
]

LABEL_PATTERN = re.compile(
    r"^(?:(?:bill|receipt|invoice|inv|ticket|awb|docket|token)\s*)?(?:no\.?|number|#)(?=[\s:.\-]+\S)[\s:.\-]*",
    re.IGNORECASE,
)


def load_cache():
    if config.OCR_CACHE.exists():
        return json.loads(config.OCR_CACHE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.OCR_CACHE.write_text(json.dumps(cache), encoding="utf-8")


CACHE_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()
_unsaved_writes = 0


def remember(cache, key, value):
    """Store one GPT answer. Photos are read in parallel, so writes are locked; the file is saved every 10 answers."""
    global _unsaved_writes
    with CACHE_LOCK:
        cache[key] = value
        _unsaved_writes += 1
        if _unsaved_writes >= 10:
            save_cache(cache)
            _unsaved_writes = 0


def file_hash(image_path):
    return hashlib.sha256(Path(image_path).read_bytes()).hexdigest()


def make_cache_key(image_path, model, prompt_version=None):
    """Same photo + same model + same prompt version -> same key."""
    return f"{file_hash(image_path)}|{model}|{prompt_version or config.PROMPT_VERSION}"



def clean_text(value):
    return "" if value is None else str(value).strip()


def clean_receipt_number(value):
    """'Bill No: 6158' -> '6158'. Leading zeros are kept."""
    return LABEL_PATTERN.sub("", clean_text(value)).strip()


def clean_gstin(value):
    """Upper-case and remove spaces. Validity is checked in validation.py."""
    return re.sub(r"\s+", "", clean_text(value)).upper()


def clean_amount(value):
    amount = parse_amount(value)
    return "" if amount is None else amount


def clean_gpt_fields(raw):
    confidence = raw.get("confidence") or {}
    return {
        "vendor": clean_text(raw.get("vendor")),
        "date_as_printed": clean_text(raw.get("date_as_printed")),
        "receipt_date": parse_date(raw.get("receipt_date")) or "",
        "receipt_number": clean_receipt_number(raw.get("receipt_number")),
        "gstin": clean_gstin(raw.get("gstin")),
        "total_amount": clean_amount(raw.get("total_amount")),
        "conf_vendor": confidence.get("vendor"),
        "conf_date": confidence.get("receipt_date"),
        "conf_number": confidence.get("receipt_number"),
        "conf_gstin": confidence.get("gstin"),
        "conf_amount": confidence.get("total_amount"),
        "photo_rotated": raw.get("photo_rotated"),
        "handwritten_notes": clean_text(raw.get("handwritten_notes")),
        "readable": raw.get("readable", True),
    }


def extract_receipt_fields(image_path, cache, model=None):
    model = model or llm_client.get_model_name()
    key = make_cache_key(image_path, model)
    if key in cache:
        saved = cache[key]
        return clean_gpt_fields(saved["fields"]), saved["usage"], True

    raw_fields, usage = llm_client.read_receipt_with_gpt(image_path, model)
    remember(cache, key, {"fields": raw_fields, "usage": usage})
    return clean_gpt_fields(raw_fields), usage, False


def read_with_second_chance(image_path, claim, cache):
    first_model = llm_client.get_model_name()
    fields, usage, from_cache = extract_receipt_fields(image_path, cache, first_model)
    flags = validate_receipt(fields, claim)
    info = {"read_pass": 1, "model": first_model, "final_model": first_model, "from_cache": from_cache,
            "prompt_tokens": usage.get("prompt_tokens", ""), "completion_tokens": usage.get("completion_tokens", ""),
            "esc_prompt_tokens": "", "esc_completion_tokens": ""}

    second_model = llm_client.get_escalation_model_name()
    if not (config.ESCALATE_FAILED_READS and count_read_problems(flags) and second_model != first_model):
        return fields, flags, info

    try:
        fields2, usage2, _ = extract_receipt_fields(image_path, cache, second_model)
    except (llm_client.NoApiKeyError, llm_client.OutOfCreditsError):
        raise  
    except Exception:
        return fields, flags, info  
    flags2 = validate_receipt(fields2, claim)
    info["esc_prompt_tokens"] = usage2.get("prompt_tokens", "")
    info["esc_completion_tokens"] = usage2.get("completion_tokens", "")
    if count_read_problems(flags2) <= count_read_problems(flags): 
        info.update({"read_pass": 2, "final_model": second_model})
        return fields2, flags2, info
    return fields, flags, info


def clean_evidence(raw):
    return {
        "business_type": clean_text(raw.get("business_type")).lower(),
        "vehicle_no": re.sub(r"[^A-Z0-9]", "", clean_text(raw.get("vehicle_no")).upper()),
        "litres": clean_amount(raw.get("litres")),
        "guest_name": clean_text(raw.get("guest_name")),
        "gst_tax_invoice": raw.get("is_gst_tax_invoice"),
        "rate_per_night_before_gst": clean_amount(raw.get("room_rate_per_night_before_gst")),
        "nights": clean_amount(raw.get("nights")),
        "driver_receipt": raw.get("is_driver_receipt"),
        "looks_altered": raw.get("looks_altered"),
        "altered_reason": clean_text(raw.get("altered_reason")),
    }


def extract_evidence(image_path, cache):
    model = llm_client.get_escalation_model_name()
    key = make_cache_key(image_path, model, config.EVIDENCE_PROMPT_VERSION)
    if key not in cache:
        raw, usage = llm_client.read_evidence_with_gpt(image_path, model)
        remember(cache, key, {"fields": raw, "usage": usage})
    return clean_evidence(cache[key]["fields"]), cache[key]["usage"]


def build_work_list(split):
    mapping = pd.read_csv(config.MAPPING_REPORT, dtype=str, keep_default_na=False)
    quality = pd.read_csv(config.QUALITY_REPORT).rename(columns={"image_status": "quality_status"})
    quality = quality[["claim_id", "quality_status", "blur_score", "brightness_score", "contrast_score"]]
    claims = pd.concat([load_claims(config.TRAIN_CSV, "train"), load_claims(config.TEST_CSV, "test")])
    claims = claims[["claim_id", "category", "expense_date", "amount_claimed"]]
    work = mapping.merge(quality, on="claim_id", how="left").merge(claims, on="claim_id", how="left")
    if split in ("train", "test"):
        work = work[work["split"] == split]
    return work


def read_one_claim(row, claim, cache):
    try:
        fields, flags, info = read_with_second_chance(claim["resolved_image_path"], claim, cache)
        row.update(fields)
        row.update(info)
        row["validation_flags"] = ";".join(flags)
        row["ocr_status"] = "ok" if fields["readable"] else "gpt_says_unreadable"
        row["image_hash"] = file_hash(claim["resolved_image_path"])[:16]
        if config.READ_EVIDENCE and (config.EVIDENCE_CATEGORIES is None or claim["category"] in config.EVIDENCE_CATEGORIES):
            try:
                evidence, ev_usage = extract_evidence(claim["resolved_image_path"], cache)
                row.update(evidence)
                row["ev_prompt_tokens"] = ev_usage.get("prompt_tokens", "")
                row["ev_completion_tokens"] = ev_usage.get("completion_tokens", "")
            except (llm_client.NoApiKeyError, llm_client.OutOfCreditsError):
                raise
            except Exception as error:
                row["error_message"] = ("evidence read failed: " + str(error))[:200]
    except (llm_client.NoApiKeyError, llm_client.OutOfCreditsError):
        raise 
    except Exception as error:
        row["ocr_status"] = "error"
        row["error_message"] = str(error)[:200]


def run_ocr(split="all", limit=None):
    if not llm_client.is_configured():
        raise llm_client.NoApiKeyError(llm_client.NO_KEY_MESSAGE)

    cache = load_cache()
    rows, jobs = [], []

    for _, claim in build_work_list(split).iterrows():
        row = {column: "" for column in RESULT_COLUMNS}
        row.update({"claim_id": claim["claim_id"], "split": claim["split"],
                    "image_file": claim["file_name_cleaned"]})
        if claim["quality_status"] == "missing":
            row["ocr_status"] = "missing_image"
        elif claim["quality_status"] == "unreadable":
            row["ocr_status"] = "unreadable_file"
        elif limit is not None and len(jobs) >= limit:
            continue
        else:
            jobs.append((row, claim))
        rows.append(row)

    done, started = [0], time.time()

    def read_and_report(job):
        read_one_claim(job[0], job[1], cache)
        with PRINT_LOCK:
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(jobs):
                print(f"  {done[0]}/{len(jobs)} photos done ({time.time() - started:.0f}s)", flush=True)

    print(f"Reading {len(jobs)} photos with {config.OCR_WORKERS} at a time (Ctrl+C stops safely)...", flush=True)
    pool = ThreadPoolExecutor(max_workers=config.OCR_WORKERS)
    try:
        list(pool.map(read_and_report, jobs))
        pool.shutdown()
    except KeyboardInterrupt:  # you pressed Ctrl+C: stop quickly and keep what was read
        pool.shutdown(wait=False, cancel_futures=True)
        print("\nStopped. Photos read so far are saved; run the same command again to continue.")
    except (llm_client.NoApiKeyError, llm_client.OutOfCreditsError):
        pool.shutdown(wait=False, cancel_futures=True)
        print("\nStopped: OpenAI credits or API key problem. Photos read so far are saved in the cache; "
              "fix it and run the same command again.")
        raise
    finally:
        save_cache(cache)

    results = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    if config.OCR_RESULTS.exists():  # keep rows from earlier runs (e.g. test now, train later)
        earlier = pd.read_csv(config.OCR_RESULTS, dtype=str, keep_default_na=False)
        earlier = earlier[~earlier["claim_id"].isin(results["claim_id"])].reindex(columns=RESULT_COLUMNS)
        results = pd.concat([earlier, results], ignore_index=True).sort_values("claim_id", ignore_index=True)
    results.to_csv(config.OCR_RESULTS, index=False)
    return results


def estimate_cost_usd(results):
    total = 0.0
    for model_column, in_column, out_column in (("model", "prompt_tokens", "completion_tokens"),
                                                 (None, "esc_prompt_tokens", "esc_completion_tokens")):
        for _, row in results.iterrows():
            tokens_in = pd.to_numeric(row[in_column], errors="coerce")
            if pd.isna(tokens_in):
                continue
            model = row["model"] if model_column else llm_client.get_escalation_model_name()
            price_in, price_out = config.PRICES_PER_M_TOKENS.get(model, (0, 0))
            tokens_out = pd.to_numeric(row[out_column], errors="coerce")
            total += tokens_in / 1e6 * price_in + (0 if pd.isna(tokens_out) else tokens_out) / 1e6 * price_out
    price_in, price_out = config.PRICES_PER_M_TOKENS.get(llm_client.get_escalation_model_name(), (0, 0))
    tokens_in = pd.to_numeric(results["ev_prompt_tokens"], errors="coerce").sum()
    tokens_out = pd.to_numeric(results["ev_completion_tokens"], errors="coerce").sum()
    return total + tokens_in / 1e6 * price_in + tokens_out / 1e6 * price_out


def summarize_ocr(results):
    lines = ["OCR SUMMARY", f"Rows: {len(results)}",
             "Status: " + str(results["ocr_status"].value_counts().to_dict())]
    read = results[results["ocr_status"] == "ok"]
    if len(read):
        lines.append("Photos read a second time with the stronger model and kept: "
                     f"{(read['read_pass'].astype(str) == '2').sum()} of {len(read)}")
        for field in ("vendor", "receipt_date", "receipt_number", "gstin", "total_amount"):
            blank = (read[field].astype(str).str.strip() == "").sum()
            lines.append(f"  blank {field}: {blank} of {len(read)}")
        flags = read["validation_flags"].str.split(";").explode()
        flags = flags[flags != ""].value_counts().to_dict()
        lines.append(f"Checks still failing after reads: {flags if flags else 'none'}")
        lines.append(f"Evidence read for {(read['business_type'] != '').sum()} photos | "
                     f"looks_altered=True: {(read['looks_altered'].astype(str) == 'True').sum()} | "
                     f"vehicle numbers read: {(read['vehicle_no'].astype(str) != '').sum()}")
        lines.append(f"Estimated cost of all reads: ~USD {estimate_cost_usd(read):.2f} "
                     "(assumed prices; counts cached reads too)")
    errors = results[results["ocr_status"] == "error"]["error_message"].value_counts().head(3)
    if len(errors):
        lines.append("Top errors: " + str(errors.to_dict()))
    return "\n".join(lines)