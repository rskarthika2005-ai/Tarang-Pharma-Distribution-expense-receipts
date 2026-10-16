import re
from collections import Counter
from difflib import SequenceMatcher

import pandas as pd

from modules.analysis import parse_amount, parse_date
from source import config

GSTIN_PATTERN = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
COURIER_NUMBER_PATTERN = re.compile(r"^[A-Z]{2}\d{8}$")  


GSTIN_MIN_SIGHTINGS = 3
GSTIN_MAX_EDITS = 2

GSTIN_USE_LOOKALIKE = True          
LOOKALIKE_PAIRS = {frozenset(pair) for pair in ("O0", "I1", "Z2", "S5", "B8", "G6", "UV")}
LOOKALIKE_COST = 0.4
GSTIN_MAX_COST = 2.0                
GSTIN_MAX_PLAIN_EDITS = 3            


PLACEHOLDER_GSTINS = {"27ABCDE1234F1ZH"}


FUEL_BRANDS = ("hpcl", "bpcl", "iocl", "indianoil", "bharatpetroleum", "hindustanpetroleum",
               "nayara", "shell", "reliance")

READ_PROBLEM_FLAGS = {
    "gstin_bad_format", "gstin_bad_checksum", "courier_number_format", "number_missing",
    "date_missing", "date_differs_from_claim", "amount_missing", "amount_differs_from_claim",
}


def gstin_checksum_ok(gstin):
    total = 0
    for position, char in enumerate(gstin[:14]):
        value = GSTIN_CHARS.index(char) * (1 if position % 2 == 0 else 2)
        total += value // 36 + value % 36
    return GSTIN_CHARS[(36 - total % 36) % 36] == gstin[14]


def check_gstin(gstin):
    if not gstin:
        return ""  
    if not GSTIN_PATTERN.match(gstin):
        return "gstin_bad_format"
    return "" if gstin_checksum_ok(gstin) else "gstin_bad_checksum"


def is_placeholder_gstin(gstin):
    gstin = str(gstin).strip().upper()
    return bool(gstin) and (gstin in PLACEHOLDER_GSTINS or gstin[2:7] == "ABCDE")


def check_receipt_number(number, category):
    if not number:
        return "number_missing"
    if category == "Courier" and not COURIER_NUMBER_PATTERN.match(number.upper()):
        return "courier_number_format"
    return ""


def check_date(receipt_date, expense_date):
    if not receipt_date:
        return "date_missing"
    claimed = parse_date(expense_date)
    if claimed is None:
        return ""
    gap = abs((pd.Timestamp(receipt_date) - pd.Timestamp(claimed)).days)
    return "date_differs_from_claim" if gap > config.DATE_WINDOW_DAYS else ""


def amounts_agree(bill_amount, claimed_amount):
    step = config.AMOUNT_ROUNDING_STEP
    return abs(bill_amount - claimed_amount) < 0.5 or abs(round(bill_amount / step) * step - claimed_amount) < 0.5


def check_amount(total_amount, amount_claimed):
    bill = parse_amount(total_amount) if total_amount != "" else None
    if bill is None:
        return "amount_missing"
    claimed = parse_amount(amount_claimed)
    if claimed is None:
        return ""
    return "" if amounts_agree(bill, claimed) else "amount_differs_from_claim"


def repair_trailing_seven(total, claimed, category):
    if category != "Local travel" or total == "":
        return total, False
    bill, claim = parse_amount(total), parse_amount(claimed)
    if bill is None or claim is None or bill != int(bill) or amounts_agree(bill, claim):
        return total, False
    text = str(int(bill))
    if len(text) > 1 and text.endswith("7") and amounts_agree(float(text[:-1]), claim):
        return float(text[:-1]), True
    return total, False


def repair_amount_fields(fields, claim):
    new_total, changed = repair_trailing_seven(fields["total_amount"], claim["amount_claimed"], claim["category"])
    if not changed:
        return fields, False
    return {**fields, "total_amount": new_total}, True


def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def lookalike_distance(first, second):
    previous = [float(j) for j in range(len(second) + 1)]
    for i, a in enumerate(first, 1):
        current = [float(i)]
        for j, b in enumerate(second, 1):
            swap = 0.0 if a == b else (LOOKALIKE_COST if frozenset((a, b)) in LOOKALIKE_PAIRS else 1.0)
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + swap))
        previous = current
    return previous[-1]


def vendor_key(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def build_gstin_book(vendors, gstins):
    book = {}
    for vendor, gstin in zip(vendors, gstins):
        gstin = str(gstin).strip().upper()
        if vendor_key(vendor) and gstin and check_gstin(gstin) == "":
            book.setdefault(vendor_key(vendor), Counter())[gstin] += 1
    return book


def repair_gstin(vendor, gstin, book):
    gstin = str(gstin).strip().upper()
    if not gstin or check_gstin(gstin) == "":
        return gstin, False
    seen = book.get(vendor_key(vendor), {})
    if GSTIN_USE_LOOKALIKE:
        close = [g for g, n in seen.items() if n >= GSTIN_MIN_SIGHTINGS
                 and round(lookalike_distance(g, gstin), 2) <= GSTIN_MAX_COST
                 and edit_distance(g, gstin) <= GSTIN_MAX_PLAIN_EDITS]
    else:
        close = [g for g, n in seen.items() if n >= GSTIN_MIN_SIGHTINGS
                 and edit_distance(g, gstin) <= GSTIN_MAX_EDITS]
    return (close[0], True) if len(close) == 1 else (gstin, False)


def validate_receipt(fields, claim):
    flags = [
        check_receipt_number(fields["receipt_number"], claim["category"]),
        check_gstin(fields["gstin"]),
        check_date(fields["receipt_date"], claim["expense_date"]),
        check_amount(fields["total_amount"], claim["amount_claimed"]),
    ]
    return [flag for flag in flags if flag]


def count_read_problems(flags):
    return sum(flag in READ_PROBLEM_FLAGS for flag in flags)


def text_similarity(first, second):
    def clean(text):
        return re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split()

    a, b = clean(first), clean(second)
    if not a or not b:
        return 0.0
    joined_a, joined_b = "".join(sorted(a)), "".join(sorted(b))
    shorter, longer = sorted((joined_a, joined_b), key=len)
    if len(shorter) >= 6 and shorter in longer:
        return 0.9  # 'rajdhani dhaba' inside 'hotel rajdhani dhaba'
    return max(SequenceMatcher(None, joined_a, joined_b).ratio(),
               SequenceMatcher(None, "".join(a), "".join(b)).ratio())


def same_vendor(left, right):
    def clean(value):
        return "" if pd.isna(value) else re.sub(r"[^a-z0-9]", "", str(value).lower())

    a, b = clean(left), clean(right)
    if not a or not b:
        return a == b
    if a in b or b in a:               
        return True
    if SequenceMatcher(None, a, b).ratio() >= 0.8:
        return True
    words = lambda v: "".join(sorted(re.findall(r"[a-z0-9]+", str(v).lower())))
    return SequenceMatcher(None, words(left), words(right)).ratio() >= 0.8


def photo_too_poor(blur, brightness, contrast):
    values = pd.to_numeric(pd.Series([blur, brightness, contrast]), errors="coerce")
    if values.isna().any():
        return False
    return bool(values[0] < config.OCR_SKIP_BLUR or values[1] < config.OCR_SKIP_BRIGHTNESS
                or values[2] < config.OCR_SKIP_CONTRAST)


def check_vendor(bill_vendor, claimed_vendor, category=""):
    if not bill_vendor or not claimed_vendor:
        return ""
    if category == "Fuel" and any(brand in vendor_key(bill_vendor) for brand in FUEL_BRANDS):
        return ""
    return "vendor_mismatch" if text_similarity(bill_vendor, claimed_vendor) < config.VENDOR_MISMATCH_BELOW else ""


def vehicle_key(text):
    return re.sub(r"[^A-Z0-9]", "", str(text).upper()) if pd.notna(text) else ""

ALL_PLATES = set(pd.read_csv(config.EMPLOYEES_CSV, dtype=str)["vehicle_no"].map(vehicle_key))


def one_char_off(first, second):
    if abs(len(first) - len(second)) > 1:
        return False
    if len(first) == len(second):
        return sum(a != b for a, b in zip(first, second)) == 1
    longer, shorter = (first, second) if len(first) > len(second) else (second, first)
    for i in range(len(longer)):
        if longer[:i] + longer[i + 1:] == shorter:
            return True
    return False


def check_vehicle(bill_vehicle, employee_vehicle):
    bill = vehicle_key(bill_vehicle)
    own = vehicle_key(employee_vehicle)
    if not bill:
        return "vehicle_missing"
    if bill == own:
        return ""
    if bill in ALL_PLATES:
        return "vehicle_mismatch"
    if one_char_off(bill, own):
        return ""
    return "vehicle_mismatch"



def build_mismatch_table(table):
    rows = []

    def add(claim, field, submitted, extracted, reason):
        rows.append({"claim_id": claim["claim_id"], "field": field, "submitted_value": submitted,
                     "extracted_value": extracted, "reason": reason})

    for _, claim in table[table["ocr_status"] == "ok"].iterrows():
        if check_vendor(claim["bill_vendor"], claim["vendor"], claim["category"]):
            add(claim, "vendor", claim["vendor"], claim["bill_vendor"], "names differ")
        if check_date(claim["bill_date"], claim["expense_date"]) == "date_differs_from_claim":
            add(claim, "date", parse_date(claim["expense_date"]), claim["bill_date"], "bill date differs from expense date")
        if check_amount(claim["bill_amount"], claim["amount_claimed"]) == "amount_differs_from_claim":
            add(claim, "amount", parse_amount(claim["amount_claimed"]), claim["bill_amount"],
                "claimed amount differs from bill (beyond Rs 10 rounding)")
        if claim["category"] in ("Fuel", "Parking") and claim["bill_vehicle_no"]:
            if check_vehicle(claim["bill_vehicle_no"], claim["vehicle_no"]) == "vehicle_mismatch":
                add(claim, "vehicle_no", claim["vehicle_no"], claim["bill_vehicle_no"],
                    "vehicle on bill is not the claimant's vehicle")
    return pd.DataFrame(rows, columns=["claim_id", "field", "submitted_value", "extracted_value", "reason"])