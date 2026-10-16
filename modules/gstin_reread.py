"""Targeted re-read of GSTINs that still fail the check digit. Spends credits only on those photos.

Run from the project folder (virtual environment active):
    python -m modules.gstin_reread --dry-run            free: lists the photos it would read, costs nothing
    python -m modules.gstin_reread --limit 3 --retry    reads 3 photos first, so you can see the cost and the hit rate
    python -m modules.gstin_reread --retry              reads every remaining target (test claims by default)
    python -m modules.gstin_reread --retry --crop-top 0.5   same, but half the attempts see only the top half of the photo
    python -m modules.gstin_reread --split all          test and train claims

--retry: also read again the photos that were recorded as 'failed' or 'error' on an earlier run
(use it after changing the prompt). Photos already 'fixed' are never read again.

Rules that keep this safe:
  * Only GSTINs that fail the check digit are read again, and only those the free vendor-based repair cannot fix.
  * The model is NOT shown the old (wrong) reading, so it cannot just copy it back. The old reading is used only to
    check the answer.
  * A new reading is accepted ONLY if it passes the official check digit AND is at most 4 characters away from
    the old reading. Anything else is recorded as 'failed' and the old value is left alone.
  * Results go to output/gstin_reread.csv. output/ocr_results.csv and the photos are never changed.
    (The photo is resized in memory for the API call, exactly as the normal read does; nothing is saved.)
  * Finished claims are skipped on re-run, and the file is saved after every photo, so Ctrl+C is safe.
"""
import argparse

import pandas as pd

from modules.ocr import build_work_list
from modules.validation import build_gstin_book, check_gstin, edit_distance, repair_gstin
from source import config, llm_client

OUT_FILE = config.OUTPUT_DIR / "gstin_reread.csv"
COLUMNS = ["claim_id", "split", "old_gstin", "new_gstin", "status", "attempts", "model",
           "prompt_tokens", "completion_tokens", "est_cost_usd"]
MAX_EDITS = 4   

SYSTEM = ("You read ONE field, the GSTIN, from a bill photo and answer ONLY with JSON. "
          "Never invent characters. If no GSTIN is printed, return an empty string.")


def make_prompt(old):
    """The old reading is deliberately NOT put in the prompt: shown a wrong value, the model tends to copy it back.
    (`old` stays as an argument because reread_one() passes it; it is used only to check the answer.)"""
    return ("Find the supplier's GSTIN on this bill (usually near the top, after the word GSTIN or GST No).\n"
            "The photo may show only the upper part of the bill, or the bill may be at an angle.\n"
            "Read it fresh from the photo, one character at a time. Count the characters in the printed GSTIN "
            "before answering; do not skip or repeat any.\n"
            "A GSTIN has exactly 15 characters: positions 1-2 digits (state code, usually 27), positions 3-7 FIVE "
            "LETTERS, 8-11 four digits, 12 a letter, 13 a letter or a non-zero digit, 14 the letter Z, 15 a letter "
            "or digit.\n"
            "Bill printers often print the letter O almost like the digit 0. Positions 3-7 are always letters, so a "
            "zero-like character there is the letter O. The same care applies to I and 1, S and 5, B and 8, G and 6, "
            "Z and 2, U and V.\n"
            'Return JSON: {"characters": ["2","7", ...each character in order...], "gstin": "the 15 characters joined"}.\n'
            'If no GSTIN is printed return {"characters": [], "gstin": ""}.')


def find_targets(split):
    ocr = pd.read_csv(config.OCR_RESULTS, dtype=str, keep_default_na=False)
    book = build_gstin_book(ocr["vendor"], ocr["gstin"])      
    rows = ocr[(ocr["ocr_status"] == "ok") & (ocr["gstin"] != "")]
    if split in ("train", "test"):
        rows = rows[rows["split"] == split]
    targets = []
    for row in rows.itertuples():
        if check_gstin(row.gstin) == "":
            continue                                           
        if repair_gstin(row.vendor, row.gstin, book)[1]:
            continue                                           
        targets.append((row.claim_id, row.split, row.gstin))
    return targets


def load_done():
    if OUT_FILE.exists():
        return pd.read_csv(OUT_FILE, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=COLUMNS)


def candidates(data):
    letters = data.get("characters") or []
    joined = llm_client._gstin_text("".join(str(c) for c in letters)) if len(letters) == 15 else ""
    return [c for c in (joined, llm_client._gstin_text(data.get("gstin"))) if c]


def _top_data_url(path, fraction):
    import base64
    import io
    from PIL import Image, ImageOps
    longest = int(getattr(config, "OCR_MAX_SIDE", 2000))
    with Image.open(path) as photo:
        photo = ImageOps.exif_transpose(photo).convert("RGB")
        width, height = photo.size
        photo = photo.crop((0, 0, width, max(1, int(height * fraction))))
        photo.thumbnail((longest, longest))
        buffer = io.BytesIO()
        photo.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def reread_one(path, old, model, attempts, crop_top=0.0):
    full_url = llm_client._image_data_url(path)
    top_url = _top_data_url(path, crop_top) if crop_top > 0 else None
    prompt = make_prompt(old)
    tokens_in = tokens_out = 0
    for attempt in range(attempts):
        image_url = top_url if (top_url and attempt % 2 == 0) else full_url
        data, usage = llm_client._ask(image_url, SYSTEM, prompt, model, temperature=0.0 if attempt == 0 else 0.5)
        tokens_in += usage["prompt_tokens"]
        tokens_out += usage["completion_tokens"]
        for candidate in candidates(data):
            if llm_client._gstin_valid(candidate) and edit_distance(candidate, old) <= MAX_EDITS:
                return candidate, attempt + 1, tokens_in, tokens_out
    return "", attempts, tokens_in, tokens_out


def main():
    parser = argparse.ArgumentParser(description="Re-read GSTINs that still fail the check digit.")
    parser.add_argument("--split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--limit", type=int, default=None, help="read at most this many photos")
    parser.add_argument("--attempts", type=int, default=3, help="API reads per photo (default 3)")
    parser.add_argument("--model", default=None, help="default: the stronger model from your settings")
    parser.add_argument("--crop-top", type=float, default=0.0,
                        help="e.g. 0.5: also send only the top half of each photo (in memory, nothing saved); "
                             "default 0 = off")
    parser.add_argument("--retry", action="store_true",
                        help="read again the photos that failed or had an error before (never the fixed ones)")
    parser.add_argument("--dry-run", action="store_true", help="list the photos only; no API calls, no cost")
    args = parser.parse_args()

    targets = find_targets(args.split)
    done = load_done()
    skip = set(done["claim_id"]) if not args.retry else set(done.loc[done["status"] == "fixed", "claim_id"])
    todo = [t for t in targets if t[0] not in skip]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"GSTINs failing the check digit that the free repair cannot fix: {len(targets)} "
          f"({len(targets) - len([t for t in targets if t[0] not in skip])} already done). To read now: {len(todo)}.")
    if args.dry_run:
        for claim_id, split, old in todo:
            print(f"  {claim_id} ({split})  {old}")
        print("Dry run: nothing was read and nothing was spent.")
        return
    if not todo:
        return
    if not llm_client.is_configured():
        print(llm_client.NO_KEY_MESSAGE)
        return

    model = args.model or llm_client.get_escalation_model_name()
    work = build_work_list("all")
    paths = dict(zip(work["claim_id"], work["resolved_image_path"]))
    price_in, price_out = getattr(config, "PRICES_PER_M_TOKENS", {}).get(model, (0, 0))
    rows = done[~done["claim_id"].isin([t[0] for t in todo])].to_dict("records")
    spent_in = spent_out = fixed = 0
    try:
        for number, (claim_id, split, old) in enumerate(todo, 1):
            try:
                new, used, tin, tout = reread_one(paths[claim_id], old, model, args.attempts, args.crop_top)
                status = "fixed" if new else "failed"
            except (llm_client.NoApiKeyError, llm_client.OutOfCreditsError) as error:
                print(f"\nStopped: {error}")
                break
            except Exception as error:
                new, used, tin, tout, status = "", 0, 0, 0, "error"
                print(f"  {claim_id}: error {str(error)[:100]}")
            spent_in, spent_out, fixed = spent_in + tin, spent_out + tout, fixed + (status == "fixed")
            cost = tin / 1e6 * price_in + tout / 1e6 * price_out
            rows.append({"claim_id": claim_id, "split": split, "old_gstin": old, "new_gstin": new, "status": status,
                         "attempts": used, "model": model, "prompt_tokens": tin, "completion_tokens": tout,
                         "est_cost_usd": f"{cost:.4f}"})
            pd.DataFrame(rows, columns=COLUMNS).to_csv(OUT_FILE, index=False)   # saved after every photo
            print(f"  {number}/{len(todo)} {claim_id}: {status}" + (f"  {old} -> {new}" if new else ""))
    except KeyboardInterrupt:
        print("\nStopped by you. Finished photos are saved; run the same command again to continue.")
    total = spent_in / 1e6 * price_in + spent_out / 1e6 * price_out
    print(f"Fixed {fixed}. Tokens used: {spent_in:,} in, {spent_out:,} out. "
          + (f"Estimated cost about USD {total:.2f} (from your PRICES_PER_M_TOKENS)." if price_in else
             f"No price listed for {model} in config.PRICES_PER_M_TOKENS, so cost is unknown; check your OpenAI usage page."))
    print("Next: python -c \"from modules.decision import run_decisions; print(run_decisions())\"")


if __name__ == "__main__":
    main()