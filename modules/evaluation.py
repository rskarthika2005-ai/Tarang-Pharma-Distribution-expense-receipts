import re
from difflib import SequenceMatcher

import pandas as pd

from modules.analysis import parse_amount, parse_date
from modules.decision import format_amount, load_decision_table
from modules.validation import GSTIN_PATTERN, check_gstin, same_vendor
from source import config

FIELDS = ["vendor", "receipt_date", "receipt_number", "gstin", "total_amount"]



def normalize_vendor(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower()) if pd.notna(text) else ""


def vendor_ratio(ocr_vendor, clerk_vendor):
    a, b = normalize_vendor(ocr_vendor), normalize_vendor(clerk_vendor)
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def vendors_match(ocr_vendor, clerk_vendor):
    return same_vendor(ocr_vendor, clerk_vendor)


def normalize_receipt_number(text, strip_zeros=False):
    value = re.sub(r"\s+", "", str(text)).upper() if pd.notna(text) else ""
    if strip_zeros and value.isdigit():
        value = value.lstrip("0")
    return value


def normalize_gstin(text):
    return re.sub(r"\s+", "", str(text)).upper() if pd.notna(text) else ""


def amounts_match(ocr_amount, clerk_amount):
    a, b = parse_amount(ocr_amount), parse_amount(clerk_amount)
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) < 0.01



def join_ocr_with_clerk():
    raw = pd.read_csv(config.OCR_RESULTS, dtype=str, keep_default_na=False)
    raw = raw[(raw["split"] == "train") & (raw["ocr_status"] == "ok")]
    raw = raw.drop_duplicates(subset="claim_id", keep="last")      # FIX 1: one OCR row per claim
    raw = raw.rename(columns={"gstin": "raw_gstin", "total_amount": "raw_total_amount"})
    raw_columns = ["claim_id", "image_file", "raw_gstin", "raw_total_amount"]
    if "read_pass" in raw.columns:
        raw_columns.append("read_pass")

    table = load_decision_table()
    repaired = table[["claim_id", "bill_vendor", "bill_date", "bill_number", "bill_gstin", "bill_amount",
                      "gstin_repaired", "amount_repaired"]]
    repaired = repaired.rename(columns={"bill_vendor": "vendor", "bill_date": "receipt_date",
                                        "bill_number": "receipt_number", "bill_gstin": "gstin",
                                        "bill_amount": "total_amount"})
    repaired = repaired.drop_duplicates(subset="claim_id", keep="last")
    repaired["total_amount"] = repaired["total_amount"].map(format_amount)

    train = pd.read_csv(config.TRAIN_CSV, dtype=str)
    has_clerk_entry = train["clerk_bill_no"].fillna("").str.strip() != ""
    train = train[has_clerk_entry]

    joined = raw[raw_columns].merge(repaired, on="claim_id")
    return joined.merge(train[["claim_id", "category", "clerk_vendor", "clerk_bill_date", "clerk_bill_no",
                               "clerk_bill_amount", "clerk_gstin"]], on="claim_id")


def compare_fields(joined):
    result = joined.copy()
    result["ok_vendor"] = [vendors_match(o, c) for o, c in zip(result["vendor"], result["clerk_vendor"])]
    result["ok_receipt_date"] = [(o or None) == parse_date(c)
                                 for o, c in zip(result["receipt_date"], result["clerk_bill_date"])]
    result["ok_number_strict"] = [normalize_receipt_number(o) == normalize_receipt_number(c)
                                  for o, c in zip(result["receipt_number"], result["clerk_bill_no"])]
    result["ok_receipt_number"] = [normalize_receipt_number(o, True) == normalize_receipt_number(c, True)
                                   for o, c in zip(result["receipt_number"], result["clerk_bill_no"])]
    result["ok_gstin"] = [normalize_gstin(o) == normalize_gstin(c)
                          for o, c in zip(result["gstin"], result["clerk_gstin"])]
    result["ok_total_amount"] = [amounts_match(o, c)
                                 for o, c in zip(result["total_amount"], result["clerk_bill_amount"])]
    result["ok_gstin_raw"] = [normalize_gstin(o) == normalize_gstin(c)
                              for o, c in zip(result["raw_gstin"], result["clerk_gstin"])]
    result["ok_total_amount_raw"] = [amounts_match(o, c)
                                     for o, c in zip(result["raw_total_amount"], result["clerk_bill_amount"])]
    result["ok_all_five"] = result[[f"ok_{f}" for f in FIELDS]].all(axis=1)
    return result




def build_accuracy_lines(compared):
    total = len(compared)
    lines = ["OCR ACCURACY vs CLERK-TYPED VALUES (train claims)", "=" * 60,
             f"Compared: {total} claims (OCR status ok and a clerk entry exists)",
             "GSTIN and amount are the repaired values the decisions use (raw values are shown further down).",
             "Clerk values contain typos and dropped leading zeros, so true accuracy can differ.",
             "", "Field accuracy:"]
    for field in FIELDS:
        lines.append(f"  {field:15s} {compared[f'ok_{field}'].mean() * 100:5.1f}%  "
                     f"({compared[f'ok_{field}'].sum()}/{total})")
    lines.append(f"  receipt_number strict (leading zeros must match): "
                 f"{compared['ok_number_strict'].mean() * 100:.1f}%")
    lines.append(f"  ALL FIVE fields right: {compared['ok_all_five'].mean() * 100:.1f}% "
                 f"({compared['ok_all_five'].sum()}/{total})")

    lines += ["", "Accuracy by category (all five fields):"]
    by_category = compared.groupby("category")["ok_all_five"].agg(["mean", "count"])
    for category, row in by_category.iterrows():
        lines.append(f"  {category:13s} {row['mean'] * 100:5.1f}%  (n={int(row['count'])})")

    same_gstin = compared[compared["ok_gstin"] & (compared["gstin"] != "")]
    fails_format = (~same_gstin["gstin"].map(lambda g: bool(GSTIN_PATTERN.match(g)))).sum()
    printed = compared[compared["gstin"] != ""]
    passing = (printed["gstin"].map(check_gstin) == "").sum()
    lines += ["", f"GSTINs (after repair) that pass format + check-digit test: {passing} of {len(printed)}"]
    if "read_pass" in compared.columns:
        second = compared[compared["read_pass"] == "2"]
        lines.append(f"Photos kept from the second (stronger) read: {len(second)}; "
                     f"all-five accuracy there: {second['ok_all_five'].mean() * 100 if len(second) else 0:.1f}%")
    lines += ["", f"GSTINs where OCR and clerk agree: {len(same_gstin)}; "
                  f"of those, failing the standard format check: {fails_format}",
              "  (if this is high, the format check is too strict for this data, or the bills carry bad GSTINs)"]
    return lines


def build_repair_lines(compared):
    lines = ["", "EFFECT OF THE REPAIRS (raw GPT read vs the value the decisions use)", "=" * 60]
    for name, raw_column, fixed_column in (("gstin", "ok_gstin_raw", "ok_gstin"),
                                           ("total_amount", "ok_total_amount_raw", "ok_total_amount")):
        helped = int((~compared[raw_column] & compared[fixed_column]).sum())
        hurt = int((compared[raw_column] & ~compared[fixed_column]).sum())
        lines.append(f"  {name:13s} raw {compared[raw_column].mean() * 100:5.1f}%  ->  "
                     f"repaired {compared[fixed_column].mean() * 100:5.1f}%   helped {helped}, hurt {hurt}")
    lines.append("  helped / hurt = now agrees / no longer agrees with the clerk. The clerk can be wrong, "
                 "so open the photo of every 'hurt' claim.")

    gstin_changed = compared[compared["gstin_repaired"] == "True"]
    amount_changed = compared[compared["amount_repaired"] == "True"]
    lines.append(f"\nClaims where a repair changed a value: GSTIN {len(gstin_changed)}, amount {len(amount_changed)}")
    for _, row in gstin_changed.iterrows():
        lines.append(f"  {row['claim_id']} gstin RAW={row['raw_gstin']!r} REPAIRED={row['gstin']!r} "
                     f"CLERK={row['clerk_gstin']!r}")
    for _, row in amount_changed.iterrows():
        lines.append(f"  {row['claim_id']} amount RAW={row['raw_total_amount']!r} REPAIRED={row['total_amount']!r} "
                     f"CLERK={row['clerk_bill_amount']!r}")
    return lines


def build_mismatch_lines(compared, per_field=12):
    lines = ["", "DISAGREEMENTS (look at the photo to see who is right)", "=" * 60]
    pairs = {"vendor": "clerk_vendor", "receipt_date": "clerk_bill_date", "receipt_number": "clerk_bill_no",
             "gstin": "clerk_gstin", "total_amount": "clerk_bill_amount"}
    for field, clerk_column in pairs.items():
        wrong = compared[~compared[f"ok_{field}"]]
        shown = wrong if field == "vendor" else wrong.head(per_field)
        label = "showing all" if field == "vendor" else f"showing up to {per_field}"
        lines.append(f"\n{field}: {len(wrong)} disagreements ({label})")
        for _, row in shown.iterrows():
            extra = f"  ratio={vendor_ratio(row['vendor'], row['clerk_vendor']):.2f}" if field == "vendor" else ""
            lines.append(f"  {row['claim_id']} [{row['category']}] OCR={row[field]!r}  CLERK={row[clerk_column]!r}"
                         f"{extra}  file={row['image_file']}")
    return lines


def run_ocr_evaluation():
    compared = compare_fields(join_ocr_with_clerk())
    text = "\n".join(build_accuracy_lines(compared) + build_repair_lines(compared) + build_mismatch_lines(compared))
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.EVALUATION_REPORT.write_text(text, encoding="utf-8")
    return text


def validate_predictions_file():
    problems = []
    sample = pd.read_csv(config.SAMPLE_SUBMISSION, dtype=str)
    test = pd.read_csv(config.TEST_CSV, dtype=str)
    predictions = pd.read_csv(config.PREDICTIONS_CSV, dtype=str, keep_default_na=False)

    if list(predictions.columns) != list(sample.columns):
        problems.append(f"columns differ: {list(predictions.columns)} vs {list(sample.columns)}")
    if predictions["claim_id"].duplicated().any():
        problems.append("duplicate claim_id rows")
    if set(predictions["claim_id"]) != set(test["claim_id"]):
        problems.append("claim_ids do not match claims_test.csv")
    if not predictions["decision"].isin(["approve", "review"]).all():
        problems.append("decision must be approve or review")
    dates = predictions["receipt_date"][predictions["receipt_date"] != ""]
    if not dates.str.match(r"^\d{4}-\d{2}-\d{2}$").all():
        problems.append("receipt_date not in YYYY-MM-DD")
    amounts = predictions["total_amount"][predictions["total_amount"] != ""]
    if pd.to_numeric(amounts, errors="coerce").isna().any():
        problems.append("total_amount has non-numeric values")
    if (predictions["reason"] == "").any():
        problems.append("some rows have no reason")
    return problems