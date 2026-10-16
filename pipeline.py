import functools
import os
import re
import traceback
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from modules.analysis import load_claims, run_dataset_analysis
from modules.decision import load_decision_table, run_decisions
from modules.preprocess import run_image_quality_analysis
from modules.validation import same_vendor
from source import config

CONF_COLUMNS = ["conf_vendor", "conf_date", "conf_number", "conf_amount"]
DATASET_REPORT = config.OUTPUT_DIR / "dataset_analysis_report.txt"
EVALUATION_REPORT = config.OUTPUT_DIR / "evaluation_report.txt"

FILTER_DEFAULTS = {"f_claim": "", "f_split": "All", "f_cat": [], "f_img": [], "f_blur": [], "f_dec": [], "f_conf": 0.0}
FILTER_LABELS = {"f_claim": "Claim ID", "f_split": "Claims set", "f_cat": "Category", "f_img": "Image status",
                 "f_blur": "Blur status", "f_dec": "Decision", "f_conf": "Min OCR confidence"}

CLERK_MONTHLY_COST = 38000
CLERK_HOURS_PER_MONTH = 160
MINUTES_PER_CLAIM = 3.5


def safe(section):
    @functools.wraps(section)
    def wrapper(*args, **kwargs):
        try:
            return section(*args, **kwargs)
        except Exception as error:
            st.error(f"This section could not be shown ({type(error).__name__}: {str(error)[:200]})")
            with st.expander("Technical details (send this to the developer)"):
                st.code(traceback.format_exc())
    return wrapper


GSTIN_PATTERN = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gstin_checksum_ok(value):
    text = str(value).strip().upper()
    if len(text) != 15 or any(c not in _GSTIN_CHARS for c in text):
        return False
    total = 0
    for i, char in enumerate(text[:14]):
        product = _GSTIN_CHARS.index(char) * (1 if i % 2 == 0 else 2)
        total += product // 36 + product % 36
    return _GSTIN_CHARS[(36 - total % 36) % 36] == text[14]
AMOUNT_TOLERANCE = 10   
CLERK_COMPARE_REPORT = config.OUTPUT_DIR / "ocr_vs_clerk_differences.csv"



def clean_amount(value):
    found = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))   
    return float(found.group()) if found else None


def clean_date(value):
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    iso = re.match(r"^\d{4}-\d{1,2}-\d{1,2}", text)
    parsed = pd.to_datetime(text, dayfirst=not iso, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def _blank(value):
    return value is None or (not isinstance(value, str) and pd.isna(value))


def clean_number(value):
    if _blank(value):
        return ""
    text = re.sub(r"\s", "", str(value)).upper()
    return text.lstrip("0") if text.isdigit() else text  


def clean_text(value):
    if _blank(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def one_char_dropped(longer, shorter):
    return len(longer) == len(shorter) + 1 and any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))


def clean_gstin(value):
    text = str(value).strip().upper()
    return "" if text == "NAN" else text


def _as_text(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)   
    return str(value)



def find_mismatches(ocr, claims):
    table = ocr.merge(claims, on="claim_id", how="left", suffixes=("", "_claim"))
    found = []

    def add(row, field, source, read, other, note=""):
        found.append({"claim_id": row["claim_id"], "field": field, "compared_with": source,
                      "ocr_value": read, "other_value": other, "note": note})

    for _, row in table.iterrows():
        read_vendor, read_date = row["vendor"], clean_date(row["receipt_date"])
        read_number, read_amount = clean_number(row["receipt_number"]), clean_amount(row["total_amount"])
        read_gstin = clean_gstin(row["gstin"])
        if row.get("split") == "train" and pd.notna(row.get("clerk_vendor")):
            if not same_vendor(read_vendor, row["clerk_vendor"]):
                add(row, "vendor", "clerk", read_vendor, row["clerk_vendor"])
            if read_date != clean_date(row["clerk_bill_date"]):
                add(row, "receipt_date", "clerk", read_date, clean_date(row["clerk_bill_date"]))
            clerk_number = clean_number(row["clerk_bill_no"])
            if read_number != clerk_number:
                note = ("clerk dropped a character (tool value is longer)" if one_char_dropped(read_number, clerk_number)
                        else "tool dropped a character" if one_char_dropped(clerk_number, read_number) else "")
                add(row, "receipt_number", "clerk", row["receipt_number"], row["clerk_bill_no"], note)
            clerk_amount = clean_amount(row["clerk_bill_amount"])
            if read_amount is None or clerk_amount is None or abs(read_amount - clerk_amount) > 0.5:
                add(row, "total_amount", "clerk", read_amount, clerk_amount)
            clerk_gstin = clean_gstin(row.get("clerk_gstin"))
            if read_gstin != clerk_gstin:
                tool_ok, clerk_ok = gstin_checksum_ok(read_gstin), gstin_checksum_ok(clerk_gstin)
                if not clerk_gstin:
                    note = "clerk left it blank; tool value " + ("passes" if tool_ok else "FAILS") + " the check digit"
                elif not read_gstin:
                    note = "tool left it blank"
                elif tool_ok and not clerk_ok:
                    note = "tool right, clerk typo (clerk value fails check digit)"
                elif clerk_ok and not tool_ok:
                    note = "tool misread (tool value fails check digit)"
                elif not tool_ok and not clerk_ok:
                    note = "both fail the check digit"
                else:
                    note = "both pass: check the photo"
                add(row, "gstin", "clerk", read_gstin, clerk_gstin, note)
        if not same_vendor(read_vendor, row.get("vendor_claim", "")):
            add(row, "vendor", "rep", read_vendor, row.get("vendor_claim", ""))
        if read_date != clean_date(row.get("expense_date", "")):
            add(row, "receipt_date", "rep", read_date, clean_date(row.get("expense_date", "")))
        claimed = clean_amount(row.get("amount_claimed", ""))
        if read_amount is None or claimed is None or abs(read_amount - claimed) > AMOUNT_TOLERANCE:
            add(row, "total_amount", "rep", read_amount, claimed)

    
        if read_gstin and not gstin_checksum_ok(read_gstin):
            add(row, "gstin_format", "format", read_gstin, "", f"{len(read_gstin)} characters; fails the GSTIN check digit")

    result = pd.DataFrame(found, columns=["claim_id", "field", "compared_with", "ocr_value", "other_value", "note"])
    for column in ["ocr_value", "other_value"]:
        result[column] = result[column].map(_as_text)
    return result


MAIN_FIELDS = ["vendor", "receipt_date", "receipt_number", "total_amount"]
MIN_SAFE_CONF = 0.75


CLERK_FAULT_NOTES = ("clerk dropped a character", "tool right, clerk typo", "clerk left it blank; tool value passes")


def accuracy_table(mismatches, clerk_ids):
    total = len(clerk_ids)
    if total == 0 or "compared_with" not in mismatches.columns:
        return None
    part = mismatches[(mismatches["compared_with"] == "clerk") & mismatches["claim_id"].isin(clerk_ids)].copy()
    notes = part["note"] if "note" in part.columns else pd.Series("", index=part.index)
    part["clerk_fault"] = notes.fillna("").astype(str).str.startswith(CLERK_FAULT_NOTES)
    pct = lambda wrong: round(100 * (total - wrong) / total, 1)
    rows = []
    for field in MAIN_FIELDS + ["gstin"]:
        found = part[part["field"] == field]
        differ, fault = found["claim_id"].nunique(), found.loc[found["clerk_fault"], "claim_id"].nunique()
        rows.append({"field": field, "claims compared": total, "differ": differ, "likely clerk error": fault,
                     "agreement %": pct(differ), "without clerk errors %": pct(differ - fault)})
    main = part[part["field"].isin(MAIN_FIELDS)]
    any_main, tool_side = main["claim_id"].nunique(), main.loc[~main["clerk_fault"], "claim_id"].nunique()
    rows.append({"field": "all 4 main fields right", "claims compared": total, "differ": any_main,
                 "likely clerk error": any_main - tool_side, "agreement %": pct(any_main),
                 "without clerk errors %": pct(tool_side)})
    return pd.DataFrame(rows)


def accuracy_by_category(differences, train_with_clerk):
    if "compared_with" not in differences.columns or train_with_clerk.empty:
        return pd.DataFrame()
    part = differences[differences["compared_with"] == "clerk"]
    rows = []
    for category, group in train_with_clerk.groupby("category"):
        ids, n = group["claim_id"], len(group)
        row = {"category": category, "claims": n}
        for field in MAIN_FIELDS + ["gstin"]:
            differ = part.loc[(part["field"] == field) & part["claim_id"].isin(ids), "claim_id"].nunique()
            row[field] = round(100 * (n - differ) / n, 1)
        wrong = part.loc[part["field"].isin(MAIN_FIELDS) & part["claim_id"].isin(ids), "claim_id"].nunique()
        row["4 main fields right"] = round(100 * (n - wrong) / n, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def risky_approvals(table):
    ok = table[table["decision"] == "approve"].copy()
    if ok.empty:                      
        return ok.assign(why="")[["claim_id", "category", "amount_claimed", "total_amount", "min_conf", "why"]]
    blank = ok[MAIN_FIELDS].fillna("").astype(str).apply(lambda col: col.str.strip() == "").any(axis=1)
    amount_column = "bill_amount" if "bill_amount" in ok.columns else "total_amount"
    bill = pd.Series([clean_amount(x) for x in ok[amount_column]], index=ok.index, dtype="float64")
    claimed = pd.Series([clean_amount(x) for x in ok["amount_claimed"]], index=ok.index, dtype="float64")
    why = pd.DataFrame({
        "a main field is blank": blank,
        "photo not ok": ok["image_status"].fillna("") != "ok",
        "low confidence": pd.Series([clean_amount(x) or 0.0 for x in ok["min_conf"]], index=ok.index, dtype="float64") < MIN_SAFE_CONF,
        "Rs 2,000 or more": bill >= 2000,
        "differs from claim": (bill - claimed).abs() > AMOUNT_TOLERANCE,
    }, index=ok.index).fillna(False)
    ok["why"] = why.apply(lambda row: "; ".join(row.index[row.values.astype(bool)]), axis=1)
    cols = ["claim_id", "category", "amount_claimed", "total_amount", "min_conf", "why"]
    return ok.loc[ok["why"] != "", cols]


def run_mismatch_analysis(ocr_path=None, train_path=None, test_path=None, out_path=None):
    using_default_ocr = ocr_path is None
    ocr_path = ocr_path or config.OCR_RESULTS
    train_path = train_path or config.TRAIN_CSV
    test_path = test_path or config.TEST_CSV
    out_path = out_path or CLERK_COMPARE_REPORT
    ocr = pd.read_csv(ocr_path, dtype=str, keep_default_na=False)
    if using_default_ocr:
        fixed = load_decision_table().set_index("claim_id")
        ocr["gstin"] = ocr["claim_id"].map(fixed["bill_gstin"]).fillna(ocr["gstin"])
        ocr["total_amount"] = ocr["claim_id"].map(fixed["bill_amount"]).fillna(ocr["total_amount"])
    train = pd.read_csv(train_path, dtype=str).assign(split="train")
    test = pd.read_csv(test_path, dtype=str).assign(split="test")
    claims = pd.concat([train, test], ignore_index=True).rename(columns={"vendor": "vendor_claim"})
    claims = claims.drop(columns=[c for c in ["emp_id"] if c in claims.columns])
    result = find_mismatches(ocr, claims)
    result.to_csv(out_path, index=False)
    return result


def read_text(path):
    return path.read_text(encoding="utf-8") if path.exists() else "Report not found. Click Run Analysis."


def _files_key():
    paths = [config.CLAIM_DECISIONS, config.QUALITY_REPORT, config.OCR_RESULTS, config.MISMATCH_REPORT]
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in paths)


@st.cache_data(show_spinner=False)
def _load(files_key):
    decisions = pd.read_csv(config.CLAIM_DECISIONS, dtype=str, keep_default_na=False)
    quality = pd.read_csv(config.QUALITY_REPORT, dtype=str, keep_default_na=False)
    ocr = pd.read_csv(config.OCR_RESULTS, dtype=str, keep_default_na=False)
    mismatches = pd.read_csv(config.MISMATCH_REPORT, dtype=str, keep_default_na=False)
    for column in CONF_COLUMNS:
        ocr[column] = pd.to_numeric(ocr[column], errors="coerce")
    ocr["min_conf"] = ocr[CONF_COLUMNS].min(axis=1)

    quality_part = quality[["claim_id", "resolved_image_path", "image_status", "blur_status", "blur_score",
                            "brightness_score", "contrast_score", "orientation_status"]]
    ocr_part = ocr[["claim_id", "vendor", "receipt_date", "receipt_number", "gstin", "total_amount",
                    "min_conf"] + CONF_COLUMNS]

    overlap = (set(quality_part.columns) | set(ocr_part.columns)) - {"claim_id"}
    decisions = decisions.drop(columns=[c for c in decisions.columns if c in overlap])
    table = decisions.merge(quality_part, on="claim_id", how="left").merge(ocr_part, on="claim_id", how="left")
    return table, quality, ocr, mismatches


def load_data():
    return _load(_files_key())




def run_analysis():
    progress = st.progress(0, text="Starting...")
    try:
        progress.progress(10, text="Checking the data files...")
        run_dataset_analysis()
        progress.progress(40, text="Checking photo quality...")
        run_image_quality_analysis()
        progress.progress(60, text="Comparing the read values with the clerk and rep values...")
        run_mismatch_analysis()
        progress.progress(75, text="Applying the decision rules...")
        run_decisions()
        progress.progress(100, text="Done")
        st.cache_data.clear()
        st.success("Analysis finished. The numbers below come from the new reports.")
    except Exception as error:
        st.error("Analysis stopped: " + str(error)[:300])
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


def clear_filters():
    for key, value in FILTER_DEFAULTS.items():
        st.session_state[key] = value


def active_filters():
    found = []
    for key, default in FILTER_DEFAULTS.items():
        value = st.session_state.get(key, default)
        if value != default:
            text = ", ".join(str(v) or "(none)" for v in value) if isinstance(value, list) else value
            found.append(f"{FILTER_LABELS[key]}: {text}")
    return found


def _keep_valid(key, options):
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in options]


def sidebar_filters(table):
    st.sidebar.header("Filters")
    st.sidebar.caption("Leave them empty to see every claim. Filters combine, so each one narrows the list.")
    st.sidebar.button("Clear filters", on_click=clear_filters)

    cat_options = sorted(table["category"].unique())
    img_options = sorted(table["image_status"].fillna("").unique())
    blur_options = sorted(table["blur_status"].fillna("").unique())
    dec_options = sorted(table["decision"].unique())
    for key, options in [("f_cat", cat_options), ("f_img", img_options), ("f_blur", blur_options),
                         ("f_dec", dec_options)]:
        _keep_valid(key, options)

    claim_text = st.sidebar.text_input("Claim ID contains", key="f_claim")
    split = st.sidebar.selectbox("Claims set", ["All", "test", "train"], key="f_split")
    categories = st.sidebar.multiselect("Category", cat_options, key="f_cat")
    image_status = st.sidebar.multiselect("Image status", img_options, key="f_img", format_func=lambda v: v or "(none)")
    blur_status = st.sidebar.multiselect("Blur status", blur_options, key="f_blur", format_func=lambda v: v or "(none)")
    decisions = st.sidebar.multiselect("Decision", dec_options, key="f_dec")
    min_conf = st.sidebar.slider("Lowest OCR confidence at least", 0.0, 1.0, 0.0, 0.05, key="f_conf")

    shown = table
    if claim_text:
        shown = shown[shown["claim_id"].str.contains(claim_text, case=False, regex=False)]
    if split != "All":
        shown = shown[shown["split"] == split]
    if categories:
        shown = shown[shown["category"].isin(categories)]
    if image_status:
        shown = shown[shown["image_status"].fillna("").isin(image_status)]
    if blur_status:
        shown = shown[shown["blur_status"].fillna("").isin(blur_status)]
    if decisions:
        shown = shown[shown["decision"].isin(decisions)]
    if min_conf > 0:
        shown = shown[shown["min_conf"].fillna(0) >= min_conf]

    st.sidebar.write(f"Showing {len(shown)} of {len(table)} claims")
    return shown


def no_claims(shown):
    if shown.empty:
        st.info("No claims match the filters. Click Clear filters in the sidebar to see everything.")
        return True
    return False


@safe
def show_summary(shown):
    active = active_filters()
    if active:
        message = f"{len(active)} filter(s) active: " + " | ".join(active)
        (st.warning if shown.empty else st.info)(message)
        st.button("Clear all filters", on_click=clear_filters, key="clear_top")

    counts = shown["decision"].value_counts()
    approve, review = int(counts.get("approve", 0)), int(counts.get("review", 0))
    first, second, third, fourth = st.columns(4)
    first.metric("Claims shown", len(shown))
    second.metric("Approve", approve)
    third.metric("Review", review)
    fourth.metric("Review rate", f"{review / len(shown) * 100:.0f}%" if len(shown) else "-")

    minutes = approve * MINUTES_PER_CLAIM
    rupees = minutes * CLERK_MONTHLY_COST / (CLERK_HOURS_PER_MONTH * 60)
    fifth, sixth = st.columns(2)
    fifth.metric("Clerk time saved (estimate)", f"{minutes / 60:.1f} hours")
    sixth.metric("Clerk cost avoided (estimate)", f"Rs {rupees:,.0f}")
    st.caption(f"Estimate for the claims shown: approved claims x {MINUTES_PER_CLAIM} min at Rs "
               f"{CLERK_MONTHLY_COST:,}/month over {CLERK_HOURS_PER_MONTH} hours (hours is an assumption). "
               "It does not subtract the GPT cost per photo or the cost of wrongly approved claims.")



@safe
def show_overview():
    st.subheader("Dataset overview")
    claims = pd.concat([load_claims(config.TRAIN_CSV, "train"), load_claims(config.TEST_CSV, "test")],
                       ignore_index=True)
    left, middle, right = st.columns(3)
    left.metric("Total claims", len(claims))
    middle.metric("Train claims", int((claims["split"] == "train").sum()))
    right.metric("Test claims", int((claims["split"] == "test").sum()))

    st.write("Claims by category")
    st.bar_chart(claims["category"].value_counts())

    missing = claims.isna().sum()
    missing = missing[missing > 0]
    st.write("Columns with missing values")
    st.dataframe(missing.rename("missing").to_frame())

    st.write("Receipt mapping (claim to photo)")
    mapping = pd.read_csv(config.MAPPING_REPORT, dtype=str, keep_default_na=False)
    st.dataframe(mapping.head(20))
    with st.expander("Full dataset analysis report"):
        st.text(read_text(DATASET_REPORT))


@safe
def show_image_quality(quality_all, shown):
    st.subheader("Image quality")
    if no_claims(shown):
        return
    quality = quality_all[quality_all["claim_id"].isin(shown["claim_id"])]
    paths = quality["resolved_image_path"][quality["resolved_image_path"] != ""]
    left, middle, right = st.columns(3)
    left.metric("Unique images", paths.nunique())
    middle.metric("Missing images", int((quality["image_status"] == "missing").sum()))
    right.metric("Unreadable images", int((quality["image_status"] == "unreadable").sum()))

    st.write("Blur status (a blur score is a hint, not proof that a photo is unreadable)")
    blur_counts = quality["blur_status"].value_counts()
    chart_col, table_col = st.columns([3, 1])
    chart_col.bar_chart(blur_counts)
    table_col.dataframe(blur_counts.rename("photos").to_frame())

    scores = pd.to_numeric(quality["blur_score"], errors="coerce").dropna()
    if len(scores):
        st.write("Blur score spread")
        spread = pd.cut(scores, 10).value_counts().sort_index()
        # Zero-padded labels: the chart sorts labels as text, so "5.5" would otherwise land after "439"
        spread.index = [f"{int(i.left):04d} to {int(i.right):04d}" for i in spread.index]
        st.bar_chart(spread)

    st.write("Photos that need manual review")
    if "review_required" in quality.columns:
        needs_review = quality[quality["review_required"].fillna("").astype(str).str.lower() == "true"]
    else: 
        needs_review = quality[(quality["image_status"] != "ok") | (quality["blur_status"] == "blurry")]
    if needs_review.empty:
        st.caption("None in the shown claims.")
    else:
        st.dataframe(needs_review)


@safe
def show_ocr_results(shown):
    st.subheader("OCR results")
    if no_claims(shown):
        return
    fields = ["vendor", "receipt_date", "receipt_number", "gstin", "total_amount"]
    blank = {field: int((shown[field].fillna("").astype(str).str.strip() == "").sum()) for field in fields}
    st.write("Blank fields in the shown claims (a blank GSTIN is normal on small bills)")
    st.dataframe(pd.Series(blank, name="blank").to_frame())
    st.caption("Sorted by lowest confidence first, so the reads most likely to be wrong are at the top.")
    st.dataframe(shown[["claim_id", "category", "image_status"] + fields + ["min_conf"]]
                 .sort_values("min_conf", na_position="last"))


@safe
def show_mismatches(mismatches, shown):
    st.subheader("Mismatch analysis")
    st.caption("A difference does not always mean an OCR error. The rep or the clerk may have typed it wrong.")
    if no_claims(shown):
        return
    part = mismatches[mismatches["claim_id"].isin(shown["claim_id"])]
    if part.empty:
        st.success("No differences found in the shown claims.")
        return
    st.write("Differences by field")
    st.bar_chart(part["field"].value_counts())
    st.dataframe(part)
    show_receipt_preview(shown[shown["claim_id"].isin(part["claim_id"])], key="mismatch")


@safe
def show_decisions(shown):
    st.subheader("Claim decisions")
    if no_claims(shown):
        return
    approved = int((shown["decision"] == "approve").sum())
    try:
        risky = risky_approvals(shown)
    except Exception as error:
        st.warning(f"The safety check could not run ({type(error).__name__}: {str(error)[:150]}). The decisions below are still shown.")
        risky = None
    if risky is None:
        pass
    elif risky.empty:
        st.success(f"Safety check passed: none of the {approved} approved claims has a blank main field, an unreadable "
                   "photo, low confidence, an amount of Rs 2,000 or more, or an amount different from the claim.")
    else:
        st.error(f"Safety check FAILED: {len(risky)} of {approved} approved claims should have gone to review.")
        st.dataframe(risky)
    if "reason" in shown.columns:
        repaired = shown[(shown["decision"] == "approve") & shown["reason"].fillna("").astype(str).str.contains("extra 7 from")]
        if len(repaired):
            st.info(f"{len(repaired)} approved claim(s) use a taxi amount corrected for an extra '7' (the '/-' after the "
                    "amount, misread): " + ", ".join(repaired["claim_id"]) + ". The safety check above uses the corrected amount.")
    st.write("Approve and review by category")
    st.bar_chart(pd.crosstab(shown["category"], shown["decision"]))

    all_flags = shown["flags"] if "flags" in shown.columns else pd.Series("", index=shown.index)
    flags = all_flags.fillna("").astype(str).str.split(";").explode()
    flags = flags[flags != ""].value_counts()
    st.write("Why claims go to review (a claim can have several reasons)")
    if len(flags):
        st.bar_chart(flags)
    else:
        st.caption("No review reasons in the shown claims.")

    st.write("Duplicate flags")
    duplicates = shown[all_flags.fillna("").astype(str).str.contains("duplicate")]
    if duplicates.empty:
        st.caption("No duplicate flags in the shown claims.")
    else:
        st.dataframe(_existing(duplicates, ["claim_id", "emp_id", "reason"]))

    st.write("All shown decisions")
    st.dataframe(_existing(shown, ["claim_id", "split", "category", "amount_claimed", "bill_amount", "decision", "reason"]))


def _compare_key():
    paths = [config.OCR_RESULTS, config.TRAIN_CSV, config.TEST_CSV]
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in paths)


@st.cache_data(show_spinner=False)
def _clerk_differences(key):
    return run_mismatch_analysis()  


@safe
def show_accuracy():
    st.write("How often the tool agrees with the clerk (train claims only)")
    train = load_claims(config.TRAIN_CSV, "train")
    if "clerk_vendor" not in train.columns:
        st.warning("The train claims have no clerk columns, so agreement cannot be worked out.")
        return
    has_clerk = train[train["clerk_vendor"].fillna("").astype(str).str.strip() != ""]
    differences = _clerk_differences(_compare_key())
    table = accuracy_table(differences, has_clerk["claim_id"])
    if table is None:
        st.warning("No train claims with clerk values were found.")
        return
    st.dataframe(table, hide_index=True)
    st.caption("Target: 95% on every field. GSTIN and amount are the repaired values the decisions use "
               "(the raw reads stay in ocr_results.csv; the evaluation report below compares raw and repaired). "
               "'Likely clerk error' counts only differences the notes explain (clerk "
               "dropped a character, clerk typo, clerk skipped a valid GSTIN). Other clerk mistakes, such as "
               "CLM-30421 (bill 250, clerk 240) and CLM-30469 (bill 335, clerk 435), are not detected, so the last "
               "column is still a lower bound. Check the photo before counting any difference.")
    printed = load_decision_table()["bill_gstin"].fillna("").astype(str).str.strip().str.upper()
    printed = printed[printed != ""]
    wrong = int((~printed.map(gstin_checksum_ok)).sum())
    st.write("GSTINs after repair, tested with the GSTIN check digit (needs no clerk, covers all claims)")
    a, b, c = st.columns(3)
    a.metric("GSTINs read", len(printed))
    b.metric("Fail the check digit (misread)", wrong)
    c.metric("Pass", f"{100 * (len(printed) - wrong) / len(printed):.1f}%" if len(printed) else "-")
    gstin_diffs = differences[(differences["compared_with"] == "clerk") & (differences["field"] == "gstin")]
    if len(gstin_diffs):
        st.write("Who is right in the GSTIN differences (decided by the check digit)")
        verdicts = gstin_diffs["note"].replace("", "(no verdict)").value_counts().rename("claims")
        st.dataframe(verdicts.to_frame())
    st.write("Agreement by category (%)")
    st.dataframe(accuracy_by_category(differences, has_clerk), hide_index=True)

    st.write("Where the tool and the clerk differ")
    clerk_part = differences[differences["compared_with"] == "clerk"]
    field = st.selectbox("Field", ["all"] + sorted(clerk_part["field"].unique()), key="clerk_diff_field")
    st.dataframe(clerk_part if field == "all" else clerk_part[clerk_part["field"] == field], hide_index=True)
    st.download_button("Download ocr_vs_clerk_differences.csv", differences.to_csv(index=False).encode("utf-8"),
                       file_name="ocr_vs_clerk_differences.csv", key="clerk_diff_download")


@safe
def show_check_sample():
    st.write("Hand-check sample: the real test of accuracy on new claims")
    st.caption("The numbers above use train claims, which the rules and prompts were tuned on, so they are optimistic. "
               "Download this sheet, open each photo, and mark every field right (Y) or wrong (N).")
    table = load_data()[0]
    pool = table[(table["split"] == "test") & (table["image_status"] == "ok")]
    if pool.empty:
        st.info("No test claims with a readable photo.")
        return
    n = st.slider("Test claims to check by hand", 10, 100, 40, 5, key="sample_n")
    sample = pool.sample(min(n, len(pool)), random_state=7).sort_values("claim_id")  
    sheet = _existing(sample, ["claim_id", "category", "resolved_image_path", "vendor", "receipt_date",
                               "receipt_number", "gstin", "total_amount", "decision"]).copy()
    sheet["gstin_check_digit"] = sample["gstin"].fillna("").astype(str).map(
        lambda g: "" if not g.strip() else ("pass" if gstin_checksum_ok(g) else "FAIL"))
    for column in ["vendor_ok", "date_ok", "number_ok", "gstin_ok", "amount_ok", "notes"]:
        sheet[column] = ""
    st.dataframe(sheet, hide_index=True)
    st.download_button("Download hand_check_sheet.csv", sheet.to_csv(index=False).encode("utf-8"),
                       file_name="hand_check_sheet.csv", key="hand_check_download")


@safe
def show_reports():
    st.subheader("Reports and downloads")
    show_accuracy()
    show_check_sample()
    st.write("Evaluation report (OCR accuracy compared with clerk-typed values)")
    st.caption("The clerk values are not perfect (for example CLM-30421: the bill says 250, the clerk typed 240). "
               "Check the photo before counting a difference as an OCR error.")
    st.text(read_text(EVALUATION_REPORT))
    for name in ["receipt_mapping_report.csv", "image_quality_report.csv", "ocr_results.csv",
                 "ocr_mismatches.csv", "claim_decisions.csv", "dataset_analysis_report.txt",
                 "evaluation_report.txt"]:
        path = config.OUTPUT_DIR / name
        if path.exists():
            st.download_button("Download " + name, path.read_bytes(), file_name=name, key=name)
    if config.PREDICTIONS_CSV.exists():
        st.download_button("Download predictions.csv", config.PREDICTIONS_CSV.read_bytes(),
                           file_name="predictions.csv", key="predictions")


def _existing(table, columns):
    return table[[c for c in columns if c in table.columns]]


def _text(row, column):
    value = row.get(column)
    return "" if pd.isna(value) else str(value)


@safe
def show_receipt_preview(shown, key="preview"):
    st.subheader("Receipt preview")
    if no_claims(shown):
        return
    options = shown["claim_id"].tolist()
    claim_filter = st.session_state.get("f_claim", "").strip()
    if st.session_state.get(key + "_filter") != claim_filter:   
        st.session_state[key + "_filter"] = claim_filter
        matches = [c for c in options if claim_filter and claim_filter.lower() in c.lower()]
        if matches:
            st.session_state[key + "_claim"] = matches[0]
    if st.session_state.get(key + "_claim") not in options:  
        st.session_state.pop(key + "_claim", None)
    claim_id = st.selectbox("Pick a claim", options, key=key + "_claim")
    rotate = st.radio("Rotate photo (view only, the file is not changed)", [0, 90, 180, 270],
                      horizontal=True, key=key + "_rotate")
    row = shown[shown["claim_id"] == claim_id].iloc[0]
    left, right = st.columns(2)
    path = _text(row, "resolved_image_path")
    if path and os.path.exists(path):
        try:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")   
            image.thumbnail((1400, 1400))                                        
            if rotate:
                image = image.rotate(-rotate, expand=True) 
            left.image(image, width=420)
        except Exception:
            left.warning("The photo could not be opened (damaged file or a format that is not supported).")
    else:
        left.warning("No photo for this claim (the upload may have failed).")
    right.write(f"Decision: {_text(row, 'decision')}")
    quality_bits = [f"{label} {_text(row, col)}" for label, col in
                    [("image", "image_status"), ("blur", "blur_status"), ("blur score", "blur_score"),
                     ("orientation", "orientation_status")] if _text(row, col)]
    if quality_bits:
        right.write("Photo quality: " + ", ".join(quality_bits))
    if _text(row, "blur_status") == "blurry" or _text(row, "image_status") not in ("", "ok"):
        right.warning("This photo may be too poor for the tool to read. If fields are blank, a person must check it.")
    right.write(f"Reason: {_text(row, 'reason')}")
    right.write(f"Vendor: {_text(row, 'vendor')}")
    right.write(f"Bill date: {_text(row, 'receipt_date')}")
    right.write(f"Bill number: {_text(row, 'receipt_number')}")
    right.write(f"GSTIN: {_text(row, 'gstin') or '(none)'}")
    gstin = _text(row, "gstin").strip().upper()
    if gstin and not gstin_checksum_ok(gstin):
        right.warning(f"This GSTIN has {len(gstin)} characters and fails the GSTIN check digit, so it is probably "
                      "misread. Check the photo.")
    raw_amount, used_amount = _text(row, "total_amount"), _text(row, "bill_amount")
    if used_amount and clean_amount(used_amount) is not None and clean_amount(raw_amount) is not None \
            and abs(clean_amount(used_amount) - clean_amount(raw_amount)) > 0.5:
        right.write(f"Total amount: {clean_amount(used_amount):g} (read from the photo as {raw_amount}; the extra 7 "
                    "from '/-' was removed because 360/- matches the claim)")
    else:
        right.write(f"Total amount: {raw_amount}")
    right.write(f"Amount claimed by rep: {_text(row, 'amount_claimed')}")