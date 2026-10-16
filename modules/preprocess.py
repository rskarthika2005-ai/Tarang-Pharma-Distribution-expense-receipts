import cv2
import numpy as np
import pandas as pd
from PIL import Image

from source import config

EXIF_ORIENTATION_TAG = 274
ROTATED_EXIF_VALUES = {3: "rotated_180", 6: "rotated_90_cw", 8: "rotated_90_ccw"}


def load_gray_image(image_path):
    """Open a photo and return it as a grey-level numpy array (original file untouched)."""
    with Image.open(image_path) as photo:
        exif_value = photo.getexif().get(EXIF_ORIENTATION_TAG, 1)
        gray = np.array(photo.convert("L"))
    return gray, exif_value


def measure_blur(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_orientation(width, height, exif_value):
    if exif_value in ROTATED_EXIF_VALUES:
        return ROTATED_EXIF_VALUES[exif_value]
    if width > height * 1.3:
        return "landscape_check"
    return "upright"


def label_blur(blur_score):
    return "blurry" if blur_score < config.BLUR_THRESHOLD else "sharp"


def analyze_one_image(image_path):
    result = {"image_status": "ok", "error_message": ""}
    try:
        gray, exif_value = load_gray_image(image_path)
    except Exception as error:
        return {"image_status": "unreadable", "error_message": str(error)[:120]}

    height, width = gray.shape
    blur_score = measure_blur(gray)
    result.update({
        "width": width,
        "height": height,
        "blur_score": round(blur_score, 1),
        "blur_status": label_blur(blur_score),
        "brightness_score": round(float(gray.mean()), 1),
        "contrast_score": round(float(gray.std()), 1),
        "orientation_status": check_orientation(width, height, exif_value),
    })
    return result


def needs_manual_review(row):
    """True when the photo may be too poor to trust automatic reading."""
    if row["image_status"] != "ok":
        return True
    return (
        row["blur_status"] == "blurry"
        or row["brightness_score"] < config.DARK_THRESHOLD
        or row["brightness_score"] > config.BRIGHT_THRESHOLD
        or row["contrast_score"] < config.LOW_CONTRAST_THRESHOLD
        or min(row["width"], row["height"]) < config.MIN_SIDE_PIXELS
    )


def run_image_quality_analysis():
    """Measure every unique photo once, then write output/image_quality_report.csv."""
    mapping = pd.read_csv(config.MAPPING_REPORT, dtype=str, keep_default_na=False)

    measured = {}
    for path in mapping.loc[mapping["image_status"] == "found", "resolved_image_path"].unique():
        measured[path] = analyze_one_image(path)

    rows = []
    for _, claim in mapping.iterrows():
        row = {
            "claim_id": claim["claim_id"],
            "image_file": claim["image_file_recorded"],
            "resolved_image_path": claim["resolved_image_path"],
        }
        if claim["image_status"] == "found":
            row.update(measured[claim["resolved_image_path"]])
        else:
            row.update({"image_status": "missing", "error_message": "photo not found in data/images"})
        rows.append(row)

    report = pd.DataFrame(rows)
    report["review_required"] = report.apply(needs_manual_review_safe, axis=1)

    columns = ["claim_id", "image_file", "resolved_image_path", "image_status", "width", "height",
               "blur_score", "blur_status", "brightness_score", "contrast_score",
               "orientation_status", "review_required", "error_message"]
    report = report.reindex(columns=columns)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    report.to_csv(config.QUALITY_REPORT, index=False)
    return report


def needs_manual_review_safe(row):
    if row["image_status"] != "ok":
        return True
    return needs_manual_review(row)


def summarize_quality(report):
    unique = report.drop_duplicates("resolved_image_path")
    unique = unique[unique["resolved_image_path"] != ""]
    ok = report[report["image_status"] == "ok"]
    lines = [
        "IMAGE QUALITY SUMMARY",
        f"Claims: {len(report)} | unique photos found: {len(unique)} | "
        f"missing: {(report['image_status'] == 'missing').sum()} | "
        f"unreadable: {(report['image_status'] == 'unreadable').sum()}",
        f"Blurry: {(ok['blur_status'] == 'blurry').sum()} | "
        f"orientation hints: {ok['orientation_status'].value_counts().to_dict()}",
        f"Claims needing manual review: {int(report['review_required'].sum())}",
    ]
    for column in ("blur_score", "brightness_score", "contrast_score"):
        if len(ok):
            values = ok[column].astype(float)
            lines.append(f"{column}: min {values.min():.0f}, 10th pct {values.quantile(.1):.0f}, "
                         f"median {values.median():.0f}, max {values.max():.0f}")
    return "\n".join(lines)
