
import re
from pathlib import Path

import pandas as pd
from dateutil import parser as date_parser

from source import config

def load_claims(csv_path, split_name):
    claims = pd.read_csv(csv_path, dtype=str)
    claims["split"] = split_name
    return claims


def get_file_name_only(image_file):
    """'/storage/emulated/0/DCIM/Camera/IMG_1.jpg' -> 'IMG_1.jpg' (handles / and \\ paths)."""
    if pd.isna(image_file) or not str(image_file).strip():
        return ""
    return re.split(r"[\\/]", str(image_file).strip())[-1]


def parse_amount(text):
    if pd.isna(text):
        return None
    match = re.search(r"\d[\d,]*(?:\.\d+)?", str(text))
    return float(match.group().replace(",", "")) if match else None


def parse_date(text):
    if pd.isna(text) or not str(text).strip():
        return None
    text = str(text).strip()
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}$", text):  
            return date_parser.parse(text, yearfirst=True, dayfirst=False).strftime("%Y-%m-%d")
        return date_parser.parse(text, dayfirst=True).strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return None




def find_image_path(image_file, images_dir=config.IMAGES_DIR):
    name = get_file_name_only(image_file)
    if not name:
        return None, "not_found"

    files_on_disk = {f.name.lower(): f for f in images_dir.glob("*") if f.is_file()}

    exact = files_on_disk.get(name.lower())
    if exact:
        how = "exact" if name == str(image_file).strip() and exact.name == name else "fixed_name"
        return exact, how

    stem = Path(name).stem.lower()
    for extension in config.IMAGE_EXTENSIONS:
        candidate = files_on_disk.get(stem + extension)
        if candidate:
            return candidate, "fixed_name"
    return None, "not_found"


def build_receipt_mapping(claims):
    rows = []
    for _, claim in claims.iterrows():
        path, how = find_image_path(claim["image_file"])
        rows.append({
            "claim_id": claim["claim_id"],
            "split": claim["split"],
            "image_file_recorded": claim["image_file"],
            "file_name_cleaned": get_file_name_only(claim["image_file"]),
            "resolved_image_path": str(path) if path else "",
            "match_type": how,
            "image_status": "found" if path else "missing",
        })
    return pd.DataFrame(rows)



def find_possible_duplicate_claims(claims):
    work = claims.copy()
    work["vendor_key"] = work["vendor"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    work["amount_num"] = work["amount_claimed"].map(parse_amount)
    work["date_iso"] = work["expense_date"].map(parse_date)

    same_person_day = work[work.duplicated(["emp_id", "vendor_key", "date_iso"], keep=False)]
    same_vendor_amount = work[
        (work["category"] == "Lodging")
        & work.duplicated(["vendor_key", "amount_num"], keep=False)
    ]
    hints = pd.concat([same_person_day, same_vendor_amount]).drop_duplicates("claim_id")
    return hints.sort_values(["vendor_key", "date_iso"])[
        ["claim_id", "split", "emp_id", "vendor", "expense_date", "amount_claimed", "category"]
    ]

def describe_claims(claims, title):
    lines = [f"--- {title} ---", f"Rows: {len(claims)}",
             f"Duplicate claim_id: {claims['claim_id'].duplicated().sum()}"]

    lines.append("Category counts: " + str(claims["category"].value_counts().to_dict()))

    amounts = claims["amount_claimed"].map(parse_amount)
    lines.append(f"Amounts that cannot be parsed: {amounts.isna().sum()}")
    lines.append(f"Amount stats (Rs): min {amounts.min():.0f}, median {amounts.median():.0f}, "
                 f"max {amounts.max():.0f}")
    lines.append(f"Claims under Rs 2,000: {(amounts < 2000).sum()} "
                 f"({(amounts < 2000).mean() * 100:.0f}%)")

    bad_dates = claims["expense_date"].map(parse_date).isna().sum()
    lines.append(f"Expense dates that cannot be parsed: {bad_dates}")

    names = claims["image_file"].fillna("")
    lines.append(f"image_file with full phone path: {names.str.contains(r'[\\\\/]').sum()}")
    lines.append(f"image_file ending .jpeg: {names.str.lower().str.endswith('.jpeg').sum()}")
    lines.append(f"image_file in WhatsApp style (IMG-...-WA...): {names.str.contains('-WA').sum()}")
    missing = claims.isna().sum()
    lines.append("Missing values: " + str(missing[missing > 0].to_dict()))
    return lines


def write_dataset_report(train, test, employees, mapping, duplicate_hints):
    lines = ["DATASET ANALYSIS REPORT", "=" * 60]
    lines += describe_claims(train, "claims_train.csv")
    lines.append("Clerk decisions: " + str(train["status"].value_counts().to_dict()))
    lines.append("Clerk notes (top): " + str(train["clerk_note"].value_counts().head(8).to_dict()))
    lines += [""] + describe_claims(test, "claims_test.csv")

    lines += ["", "--- employees.csv ---", f"Employees: {len(employees)}",
              "Grades: " + str(employees["grade"].value_counts().to_dict())]
    unknown = (set(train["emp_id"]) | set(test["emp_id"])) - set(employees["emp_id"])
    lines.append(f"emp_id in claims but not in employees.csv: {sorted(unknown) or 'none'}")

    lines += ["", "--- Receipt mapping ---"]
    for split in ("train", "test"):
        part = mapping[mapping["split"] == split]
        lines.append(f"{split}: {len(part)} claims, {(part['image_status'] == 'found').sum()} photos found, "
                     f"{(part['image_status'] == 'missing').sum()} missing")
    lines.append(f"Unique image names used: {mapping['file_name_cleaned'].nunique()} of {len(mapping)} claims")
    lines.append("NOTE: 'missing' = not in data/images/. It can mean the folder is incomplete "
                 "OR the upload failed in the app.")

    lines += ["", f"--- Possible duplicate claims (hints only): {len(duplicate_hints)} ---",
              duplicate_hints.to_string(index=False) if len(duplicate_hints) else "none"]

    text = "\n".join(lines)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.DATASET_REPORT.write_text(text, encoding="utf-8")
    return text


def run_dataset_analysis():
    train = load_claims(config.TRAIN_CSV, "train")
    test = load_claims(config.TEST_CSV, "test")
    employees = pd.read_csv(config.EMPLOYEES_CSV)

    all_claims = pd.concat([train, test], ignore_index=True)
    mapping = build_receipt_mapping(all_claims)
    mapping.to_csv(config.MAPPING_REPORT, index=False)

    duplicate_hints = find_possible_duplicate_claims(all_claims)
    return write_dataset_report(train, test, employees, mapping, duplicate_hints)
