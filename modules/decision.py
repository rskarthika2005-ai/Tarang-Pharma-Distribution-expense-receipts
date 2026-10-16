
import re
from collections import Counter

import pandas as pd

from modules.analysis import load_claims, parse_amount
from modules.validation import (build_gstin_book, build_mismatch_table, check_amount, check_date, check_gstin,
                                check_vehicle, check_vendor, is_placeholder_gstin, photo_too_poor, repair_gstin,
                                repair_trailing_seven, text_similarity, vendor_key)
from source import config


EXPECTED_BUSINESS = {"Fuel": {"fuel"}, "Food": {"food", "restaurant", "hotel"}, "Lodging": {"hotel"},
                     "Local travel": {"cab"}, "Courier": {"courier"}, "Parking": {"parking"}}
PHOTO_PROBLEMS = {"missing_image": "bill photo missing (upload failed) - ask rep to resend",
                  "unreadable_file": "bill photo file cannot be opened",
                  "gpt_says_unreadable": "bill is not readable in the photo",
                  "error": "bill could not be read automatically", "": "bill not read yet"}


CLERK_CHECK_AMOUNT = 2000


GSTIN_NOTE = "GSTIN needs manual check (fails check digit)"
GSTIN_REPAIRED_NOTE = "GSTIN corrected from this vendor's other bills"
AMOUNT_REPAIRED_NOTE = "amount read with an extra 7 from '/-', corrected to match the claim"


INSTRUCTION_PATTERNS = re.compile(
    r"automated processing|automated system|approve (it|this)|skip (any|the) (duplicate|check)"
    r"|in your output|ignore (all |any |the )?(previous|above|prior)|pre-verified", re.IGNORECASE)


def is_true(value):
    return str(value).strip().lower() == "true"


def has_embedded_instructions(text):
    return bool(INSTRUCTION_PATTERNS.search(str(text)))


def to_number(value):
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def number_key(text):
    value = re.sub(r"\s+", "", str(text)).upper()
    return value.lstrip("0") if value.isdigit() else value


def format_amount(value):
    number = to_number(value)
    return "" if number is None else f"{number:.2f}".rstrip("0").rstrip(".")


def load_decision_table():
    claims = pd.concat([load_claims(config.TRAIN_CSV, "train"), load_claims(config.TEST_CSV, "test")],
                       ignore_index=True)
    employees = pd.read_csv(config.EMPLOYEES_CSV, dtype=str)
    quality = pd.read_csv(config.QUALITY_REPORT)[["claim_id", "blur_score", "brightness_score", "contrast_score"]]
    ocr = pd.read_csv(config.OCR_RESULTS, dtype=str, keep_default_na=False).drop(columns=["split", "image_file"])
    ocr = ocr.rename(columns={"vendor": "bill_vendor", "receipt_date": "bill_date", "receipt_number": "bill_number",
                              "gstin": "bill_gstin", "total_amount": "bill_amount", "vehicle_no": "bill_vehicle_no",
                              "guest_name": "bill_guest_name"})
    clerk_pairs = {}
    if "clerk_gstin" in claims.columns:
        typed = claims[claims["split"] == "train"][["claim_id", "clerk_gstin"]].dropna()
        typed = typed.merge(ocr[["claim_id", "bill_vendor"]], on="claim_id")
        clerk_pairs = {c: (v, str(g).strip().upper())
                       for c, v, g in zip(typed["claim_id"], typed["bill_vendor"], typed["clerk_gstin"])}
    book = build_gstin_book(list(ocr["bill_vendor"]) + [v for v, _ in clerk_pairs.values()],
                            list(ocr["bill_gstin"]) + [g for _, g in clerk_pairs.values()])

    def book_for(claim_id):
        own = clerk_pairs.get(claim_id)
        if own is None:
            return book
        key, gstin = vendor_key(own[0]), own[1]
        if key not in book or not book[key].get(gstin) or check_gstin(gstin):
            return book
        trimmed = dict(book)
        trimmed[key] = book[key] - Counter({gstin: 1})
        return trimmed

    fixed = [repair_gstin(v, g, book_for(c)) for c, v, g in zip(ocr["claim_id"], ocr["bill_vendor"], ocr["bill_gstin"])]
    ocr["bill_gstin"] = [f[0] for f in fixed]
    ocr["gstin_repaired"] = [str(f[1]) for f in fixed]

    table = (claims.merge(employees, on="emp_id", how="left")
                   .merge(quality, on="claim_id", how="left")
                   .merge(ocr, on="claim_id", how="left"))
    ocr_columns = [c for c in ocr.columns if c != "claim_id"]
    table[ocr_columns] = table[ocr_columns].fillna("")
    amounts = [repair_trailing_seven(a, c, cat) if status == "ok" else (a, False)
               for a, c, cat, status in zip(table["bill_amount"], table["amount_claimed"],
                                            table["category"], table["ocr_status"])]
    table["bill_amount"] = [f[0] for f in amounts]
    table["amount_repaired"] = [str(f[1]) for f in amounts]
    unread = (table["split"] == "train") & (table["ocr_status"] == "") & table["clerk_bill_no"].notna()
    table["dup_vendor"] = table["bill_vendor"].where(~unread, table["clerk_vendor"].fillna(""))
    table["dup_number"] = table["bill_number"].where(~unread, table["clerk_bill_no"].fillna(""))
    table["dup_ready"] = (table["ocr_status"] == "ok") | unread
    table["number_key"] = table["dup_number"].map(number_key)
    table["bill_number_key"] = table["bill_number"].map(number_key)
    return table

def find_duplicate_claims(table):
    """Returns {claim_id: reason} for bills that look claimed twice (same bill number and vendor, or same photo)."""
    found, seen = {}, []
    for _, row in table[table["dup_ready"]].sort_values("claim_id").iterrows():
        match = None
        for earlier in seen:
            same_bill = (row["number_key"] != "" and row["number_key"] == earlier["number_key"]
                         and text_similarity(row["dup_vendor"], earlier["dup_vendor"]) >= 0.7)
            same_photo = row["image_hash"] != "" and row["image_hash"] == earlier["image_hash"]
            if same_bill or same_photo:
                match = earlier
                break
        if match is not None:
            other_person = match["emp_id"] != row["emp_id"]
            found[row["claim_id"]] = (f"same bill as {match['claim_id']}"
                                      + (f" claimed by {match['emp_id']} - claimed once, by the payer" if other_person
                                         else " - claimed twice by the same person"))
            if other_person and match["split"] == "test":
                found.setdefault(match["claim_id"], f"same bill also claimed in {row['claim_id']} by {row['emp_id']}")
        seen.append(row)
    return found


def review_reasons(row, duplicates):
    """Returns (reasons, notes). reasons = list of (code, text) that force review; notes = passed checks."""
    status = row["ocr_status"]
    if status != "ok":
        return [("no_photo", PHOTO_PROBLEMS.get(status, "bill could not be read"))], []

    reasons, notes = [], []
    category = row["category"]
    bill_amount = to_number(row["bill_amount"])

    if row["claim_id"] in duplicates:
        reasons.append(("duplicate", duplicates[row["claim_id"]]))
    if is_true(row["looks_altered"]):
        reasons.append(("altered", "bill looks altered: " + (row["altered_reason"] or "figures tampered with")))
    if has_embedded_instructions(row.get("handwritten_notes", "")) or has_embedded_instructions(row.get("altered_reason", "")):
        reasons.append(("instructions", "bill carries text addressed to an automated reviewer; a person must check it"))
    if photo_too_poor(row["blur_score"], row["brightness_score"], row["contrast_score"]):
        reasons.append(("poor_photo", "photo too blurry, dark or flat to trust the read"))

   
    amount_flag = check_amount(row["bill_amount"], row["amount_claimed"])
    if amount_flag == "amount_missing":
        reasons.append(("amount", "bill amount not readable"))
    elif amount_flag:
        reasons.append(("amount", f"claimed Rs {format_amount(parse_amount(row['amount_claimed']))} "
                                  f"but bill shows Rs {format_amount(bill_amount)}"))
    else:
        notes.append(f"Rs {format_amount(bill_amount)} matches claim")

    date_flag = check_date(row["bill_date"], row["expense_date"])
    if date_flag == "date_missing":
        reasons.append(("date", "bill date not readable"))
    elif date_flag:
        reasons.append(("date", f"bill date {row['bill_date']} differs from expense date"))

    if row["bill_number_key"] == "":
        reasons.append(("number", "bill number not readable, so the duplicate check is not possible"))
    elif category == "Courier" and not re.match(r"^[A-Z]{2}\d{8}$", row["bill_number_key"]):
        reasons.append(("number", "courier number looks misread"))
    else:
        notes.append("bill number not seen on any other claim" if row["claim_id"] not in duplicates else "")

    for field, column in (("amount", "conf_amount"), ("date", "conf_date"), ("bill number", "conf_number")):
        confidence = to_number(row[column])
        if confidence is not None and confidence < config.MIN_CONFIDENCE:
            reasons.append(("low_confidence", f"low read confidence on {field} ({confidence:.2f})"))

    
    if check_vendor(row["bill_vendor"], row["vendor"], category):
        reasons.append(("vendor", f"bill vendor '{row['bill_vendor']}' differs from claimed '{row['vendor']}'"))
    business = row["business_type"]
    if business and business != "other" and business not in EXPECTED_BUSINESS.get(category, {business}):
        reasons.append(("category", f"bill looks like a {business} bill but was claimed as {category}"))

    gstin = str(row["bill_gstin"]).strip().upper()
    if is_placeholder_gstin(gstin):
        reasons.append(("gstin", "GSTIN looks like a placeholder, so the read cannot be trusted"))
    elif gstin and category != "Lodging" and check_gstin(gstin):
        reasons.append(("gstin", "GSTIN fails the format or check-digit test"))


    if category == "Fuel":
        vehicle_flag = check_vehicle(row["bill_vehicle_no"], row["vehicle_no"])
        if vehicle_flag == "vehicle_missing":
            reasons.append(("vehicle", "fuel bill shows no vehicle number (policy requires it)"))
        elif vehicle_flag:
            reasons.append(("vehicle", f"vehicle {row['bill_vehicle_no']} on bill is not the claimant's"))
        else:
            notes.append(f"vehicle {row['bill_vehicle_no']} is the claimant's")
        is_car_user = "area sales manager" in row["designation"].lower()
        litres = to_number(row["litres"])
        if not is_car_user:
            if litres is not None and litres > config.TWO_WHEELER_MAX_LITRES:
                reasons.append(("fuel_volume", f"{litres:g} L is more than a two-wheeler can hold"))
            elif litres is None and bill_amount is not None and bill_amount > config.TWO_WHEELER_MAX_FUEL_RS:
                reasons.append(("fuel_volume", f"Rs {format_amount(bill_amount)} fuel is too high for a two-wheeler"))

    elif category == "Parking":
        if row["bill_vehicle_no"] and check_vehicle(row["bill_vehicle_no"], row["vehicle_no"]):
            reasons.append(("vehicle", f"parking ticket is for vehicle {row['bill_vehicle_no']}, not the claimant's"))
        elif row["bill_vehicle_no"]:
            notes.append("ticket vehicle is the claimant's")

    elif category == "Lodging":
        if not row["bill_gstin"]:
            reasons.append(("gst_invoice", "hotel bill has no GSTIN (GST tax invoice required)"))
        elif check_gstin(row["bill_gstin"]):
            reasons.append(("gst_invoice", "hotel GSTIN fails the check-digit test"))
        elif is_true(row.get("gstin_repaired", "")):
            # a hotel GST invoice is a policy requirement, so an auto-corrected GSTIN is never approved unseen
            reasons.append(("gst_invoice", "hotel GSTIN was auto-corrected from other bills; clerk to confirm it"))
        if not is_true(row["gst_tax_invoice"]):
            reasons.append(("gst_invoice", "hotel bill is not a GST tax invoice"))
        guest = row["bill_guest_name"]
        if not guest:
            reasons.append(("guest_name", "no guest name on hotel bill (must be in employee's name)"))
        elif max(text_similarity(guest, token) for token in str(row["name"]).split()) < 0.8 \
                and text_similarity(guest, row["name"]) < 0.6:
            reasons.append(("guest_name", f"hotel bill is in the name '{guest}', not the employee's"))
        rate, limit = to_number(row["rate_per_night_before_gst"]), config.LODGING_LIMIT_PER_NIGHT.get(row["grade"])
        if rate is None:
            reasons.append(("room_rate", "room rate per night before GST not readable"))
        elif limit and rate > limit:
            reasons.append(("room_rate", f"Rs {format_amount(rate)}/night before GST is over the {row['grade']} limit Rs {limit}"))

    elif category == "Local travel":
        if not is_true(row["driver_receipt"]):
            reasons.append(("driver_receipt", "no driver's receipt on the cab bill"))

    elif category == "Food":
        if bill_amount is not None and bill_amount > config.FOOD_LIMIT_PER_BILL:
            reasons.append(("food_limit", f"Rs {format_amount(bill_amount)} is over the Rs {config.FOOD_LIMIT_PER_BILL} per-bill limit"))


    if bill_amount is not None and bill_amount >= CLERK_CHECK_AMOUNT:
        reasons.append(("big_bill", f"Rs {format_amount(bill_amount)} is Rs {CLERK_CHECK_AMOUNT:,} or more, "
                                    "a clerk checks it"))

    return reasons, [note for note in notes if note]


def short_reason(reasons):
    texts = [text for _, text in reasons]
    return "; ".join(texts[:3]) + (f" (+{len(texts) - 3} more)" if len(texts) > 3 else "")


def approve_reason(row, notes):
    return "Bill read clearly; " + "; ".join(notes[:3]) + "; passes policy checks"


def add_gstin_note(row, reason):
    gstin = str(row["bill_gstin"]).strip()
    if row["ocr_status"] != "ok" or not gstin or "check-digit" in reason or GSTIN_NOTE in reason:
        return reason
    return reason + "; " + GSTIN_NOTE if check_gstin(gstin) else reason


def add_repair_notes(row, reason):
    if row["ocr_status"] != "ok":
        return reason
    if is_true(row.get("gstin_repaired", "")) and GSTIN_REPAIRED_NOTE not in reason:
        reason += "; " + GSTIN_REPAIRED_NOTE
    if is_true(row.get("amount_repaired", "")) and AMOUNT_REPAIRED_NOTE not in reason:
        reason += "; " + AMOUNT_REPAIRED_NOTE
    return reason



def decide_all(table):
    duplicates = find_duplicate_claims(table)
    rows = []
    for _, row in table.iterrows():
        reasons, notes = review_reasons(row, duplicates)
        reason = add_gstin_note(row, short_reason(reasons) if reasons else approve_reason(row, notes))
        rows.append({
            "claim_id": row["claim_id"], "split": row["split"], "emp_id": row["emp_id"], "category": row["category"],
            "amount_claimed": row["amount_claimed"], "bill_amount": row["bill_amount"],
            "decision": "review" if reasons else "approve",
            "reason": add_repair_notes(row, reason),
            "flags": ";".join(code for code, _ in reasons),
            "ocr_status": row["ocr_status"], "clerk_status": row.get("status", ""), "clerk_note": row.get("clerk_note", ""),
        })
    return pd.DataFrame(rows)


def build_predictions(table, decisions):
    merged = table.merge(decisions[["claim_id", "decision", "reason"]], on="claim_id")
    test = merged[merged["split"] == "test"]
    out = pd.DataFrame({
        "claim_id": test["claim_id"], "vendor": test["bill_vendor"], "receipt_date": test["bill_date"],
        "receipt_number": test["bill_number"], "gstin": test["bill_gstin"],
        "total_amount": test["bill_amount"].map(format_amount),
        "decision": test["decision"], "reason": test["reason"],
    })
    return out


def summarize_decisions(decisions):
    lines = ["DECISION SUMMARY"]
    for split in ("test", "train"):
        part = decisions[decisions["split"] == split]
        if split == "train":
            part = part[part["ocr_status"] != ""]  # only train claims whose photo has been read
        lines.append(f"{split}: {len(part)} claims -> {part['decision'].value_counts().to_dict()}")
    counted = decisions[~((decisions["split"] == "train") & (decisions["ocr_status"] == ""))]  # skip unread train claims
    flags = counted[counted["decision"] == "review"]["flags"].str.split(";").explode().value_counts()
    lines.append("Review reasons (a claim can have several): " + str(flags.to_dict()))
    train = decisions[(decisions["split"] == "train") & (decisions["ocr_status"] != "")]
    stopped = train[train["clerk_status"].isin(["queried", "rejected"])]
    lines.append(f"Train claims the clerks queried or rejected: {len(stopped)}; "
                 f"our decision was review for {(stopped['decision'] == 'review').sum()} of them")
    approved = train[train["clerk_status"] == "approved"]
    lines.append(f"Train claims the clerks approved: {len(approved)}; we would send "
                 f"{(approved['decision'] == 'review').sum()} of them to review")
    return "\n".join(lines)


def run_decisions():
    table = load_decision_table()
    decisions = decide_all(table)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    decisions.to_csv(config.CLAIM_DECISIONS, index=False)
    build_mismatch_table(table).to_csv(config.MISMATCH_REPORT, index=False)
    build_predictions(table, decisions).to_csv(config.PREDICTIONS_CSV, index=False)
    return summarize_decisions(decisions)
