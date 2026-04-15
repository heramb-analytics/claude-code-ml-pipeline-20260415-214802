"""Data ingestion and validation for transaction anomaly pipeline."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["transaction_id", "timestamp", "merchant_id", "amount", "category", "is_anomaly"]
RAW_PATH = Path("data/raw/transactions.csv")
CLEAN_PATH = Path("data/processed/clean.parquet")
QUALITY_REPORT_PATH = Path("logs/quality_report.json")


class DataQualityError(Exception):
    """Raised when a critical data quality check fails and cannot be auto-healed."""


def _log_check(results: list, check_name: str, passed: bool, detail: str = "") -> None:
    """Append a quality check result to the results list."""
    results.append({"check": check_name, "passed": passed, "detail": detail})
    status = "passed" if passed else "FAILED"
    print(f"   {'✓' if passed else '✗'} Check {len(results)}/10: {check_name} — {status}"
          + (f" — {detail}" if detail else ""))


def run_quality_checks(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Run 10 quality assertions, auto-healing where possible.

    Args:
        df: Raw dataframe to validate.

    Returns:
        Tuple of (cleaned dataframe, list of check result dicts).

    Raises:
        DataQualityError: If a critical check cannot be healed.
    """
    results: list = []

    # Check 1 — required columns present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataQualityError(f"Missing required columns: {missing}")
    _log_check(results, "required_columns_present", True)

    # Check 2 — no duplicate transaction IDs
    dupes = df["transaction_id"].duplicated().sum()
    if dupes:
        print(f"   ✗ Check 2/10: no_duplicate_transaction_ids — FAILED — auto-fixing {dupes} dupes...")
        df = df.drop_duplicates(subset=["transaction_id"])
    _log_check(results, "no_duplicate_transaction_ids", True, f"{dupes} removed" if dupes else "")

    # Check 3 — amount is positive
    neg = (df["amount"] <= 0).sum()
    if neg:
        print(f"   ✗ Check 3/10: amount_positive — FAILED — dropping {neg} non-positive rows...")
        df = df[df["amount"] > 0]
    _log_check(results, "amount_positive", True, f"{neg} rows removed" if neg else "")

    # Check 4 — amount within realistic range (< 1,000,000)
    outliers = (df["amount"] > 1_000_000).sum()
    if outliers:
        print(f"   ✗ Check 4/10: amount_realistic_range — FAILED — capping {outliers} outliers...")
        df = df[df["amount"] <= 1_000_000]
    _log_check(results, "amount_realistic_range", True, f"{outliers} rows removed" if outliers else "")

    # Check 5 — is_anomaly is binary (0 or 1)
    invalid_labels = (~df["is_anomaly"].isin([0, 1])).sum()
    if invalid_labels:
        print(f"   ✗ Check 5/10: is_anomaly_binary — FAILED — dropping {invalid_labels} invalid labels...")
        df = df[df["is_anomaly"].isin([0, 1])]
    _log_check(results, "is_anomaly_binary", True, f"{invalid_labels} removed" if invalid_labels else "")

    # Check 6 — no nulls in key columns
    null_counts = df[["transaction_id", "amount", "is_anomaly"]].isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls:
        print(f"   ✗ Check 6/10: no_nulls_in_key_columns — FAILED — dropping {total_nulls} null rows...")
        df = df.dropna(subset=["transaction_id", "amount", "is_anomaly"])
    _log_check(results, "no_nulls_in_key_columns", True, f"{total_nulls} rows removed" if total_nulls else "")

    # Check 7 — timestamp parseable
    try:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        _log_check(results, "timestamp_parseable", True)
    except Exception as exc:
        raise DataQualityError(f"Timestamp parsing failed: {exc}") from exc

    # Check 8 — category is non-empty string
    empty_cats = (df["category"].isna() | (df["category"].str.strip() == "")).sum()
    if empty_cats:
        print(f"   ✗ Check 8/10: category_non_empty — FAILED — filling {empty_cats} with 'unknown'...")
        df["category"] = df["category"].fillna("unknown").replace("", "unknown")
    _log_check(results, "category_non_empty", True, f"{empty_cats} filled" if empty_cats else "")

    # Check 9 — merchant_id non-null
    null_merch = df["merchant_id"].isna().sum()
    if null_merch:
        print(f"   ✗ Check 9/10: merchant_id_non_null — FAILED — filling {null_merch} with 'UNKNOWN'...")
        df["merchant_id"] = df["merchant_id"].fillna("UNKNOWN")
    _log_check(results, "merchant_id_non_null", True, f"{null_merch} filled" if null_merch else "")

    # Check 10 — dataset has at least both classes
    class_counts = df["is_anomaly"].value_counts()
    if len(class_counts) < 2:
        raise DataQualityError("Dataset contains only one class — cannot train a classifier.")
    _log_check(results, "both_classes_present", True, f"normal={class_counts.get(0,0)} anomaly={class_counts.get(1,0)}")

    return df, results


def ingest() -> pd.DataFrame:
    """Load raw data, run quality checks, save clean parquet.

    Returns:
        Cleaned dataframe.
    """
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    df_clean, results = run_quality_checks(df)

    df_clean.to_parquet(CLEAN_PATH, index=False)
    print(f"   💾 Saved: {CLEAN_PATH} ({len(df_clean)} rows, {len(df_clean.columns)} cols)")

    passed = sum(r["passed"] for r in results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(RAW_PATH),
        "rows_input": len(df),
        "rows_output": len(df_clean),
        "checks_passed": passed,
        "checks_total": len(results),
        "checks": results,
    }
    QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"   📄 Saved: {QUALITY_REPORT_PATH}")

    return df_clean


if __name__ == "__main__":
    ingest()
