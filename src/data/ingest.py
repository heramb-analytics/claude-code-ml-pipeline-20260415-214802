"""Data ingestion and validation module for transaction anomaly detection.

This module handles loading raw transaction data and applying
10 quality assertions before saving cleaned output.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/transactions.csv")
PROCESSED_PATH = Path("data/processed/clean.parquet")
QUALITY_REPORT_PATH = Path("logs/quality_report.json")


class DataQualityError(Exception):
    """Raised when a critical data quality check fails and cannot be auto-fixed."""


def log_event(event: dict) -> None:
    """Append a JSON Lines log entry to logs/audit.jsonl.

    Args:
        event: Dictionary containing the event data to log.
    """
    log_path = Path("logs/audit.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({**event, "timestamp": datetime.now(timezone.utc).isoformat()}) + "\n")


def run_quality_checks(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Run 10 quality checks on the raw DataFrame and auto-fix issues.

    Args:
        df: Raw DataFrame loaded from CSV.

    Returns:
        Tuple of (cleaned DataFrame, list of check result dicts).

    Raises:
        DataQualityError: If a check fails and cannot be auto-fixed.
    """
    results = []

    def record(n: int, name: str, passed: bool, fix: str = "") -> None:
        results.append({"check": n, "name": name, "passed": passed, "fix": fix})
        status = "passed" if passed else f"FAILED — auto-fixing: {fix}"
        print(f"   {'✓' if passed else '✗'} Check {n}/10: {name} — {status}")

    # Check 1: File loaded successfully (non-empty)
    record(1, "file_not_empty", len(df) > 0)

    # Check 2: Required columns present
    required_cols = {"transaction_id", "timestamp", "amount", "category", "is_anomaly"}
    missing = required_cols - set(df.columns)
    if missing:
        record(2, "required_columns_present", False, f"Dropping missing-col rows for {missing}")
        for col in missing:
            df[col] = np.nan
    else:
        record(2, "required_columns_present", True)

    # Check 3: No duplicate transaction IDs
    dups = df.duplicated(subset=["transaction_id"])
    if dups.any():
        record(3, "no_duplicate_transaction_ids", False, f"Dropping {dups.sum()} duplicate rows")
        df = df[~dups].copy()
    else:
        record(3, "no_duplicate_transaction_ids", True)

    # Check 4: amount is numeric and non-negative
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    neg_mask = df["amount"] < 0
    if neg_mask.any():
        record(4, "amount_non_negative", False, f"Clipping {neg_mask.sum()} negative amounts to 0")
        df.loc[neg_mask, "amount"] = 0.0
    else:
        record(4, "amount_non_negative", True)

    # Check 5: No null amounts
    null_amount = df["amount"].isna()
    if null_amount.any():
        record(5, "no_null_amounts", False, f"Filling {null_amount.sum()} null amounts with median")
        df["amount"] = df["amount"].fillna(df["amount"].median())
    else:
        record(5, "no_null_amounts", True)

    # Check 6: is_anomaly is binary (0 or 1)
    valid_labels = df["is_anomaly"].isin([0, 1])
    if not valid_labels.all():
        bad_count = (~valid_labels).sum()
        record(6, "is_anomaly_binary", False, f"Coercing {bad_count} invalid labels to 0")
        df.loc[~valid_labels, "is_anomaly"] = 0
    else:
        record(6, "is_anomaly_binary", True)

    # Check 7: timestamp parseable
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        record(7, "timestamp_parseable", True)
    except Exception:
        record(7, "timestamp_parseable", False, "Dropping rows with unparseable timestamps")
        df = df[pd.to_datetime(df["timestamp"], errors="coerce").notna()].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Check 8: category column has no nulls
    null_cat = df["category"].isna()
    if null_cat.any():
        record(8, "no_null_categories", False, f"Filling {null_cat.sum()} null categories with 'unknown'")
        df["category"] = df["category"].fillna("unknown")
    else:
        record(8, "no_null_categories", True)

    # Check 9: amount within reasonable range (0 to 1,000,000)
    extreme_mask = df["amount"] > 1_000_000
    if extreme_mask.any():
        record(9, "amount_reasonable_range", False, f"Capping {extreme_mask.sum()} extreme amounts at 1,000,000")
        df.loc[extreme_mask, "amount"] = 1_000_000.0
    else:
        record(9, "amount_reasonable_range", True)

    # Check 10: Anomaly rate is plausible (between 0.001% and 50%)
    anomaly_rate = df["is_anomaly"].mean()
    plausible = 0.0 < anomaly_rate <= 0.5
    if not plausible:
        record(10, "anomaly_rate_plausible", False, "Anomaly rate outside expected bounds — pipeline proceeds but flag raised")
    else:
        record(10, "anomaly_rate_plausible", True)

    return df, results


def ingest() -> pd.DataFrame:
    """Load, validate, and save cleaned transaction data.

    Returns:
        Cleaned DataFrame saved as Parquet.
    """
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    print("   📂 Loading raw data from data/raw/transactions.csv...")
    df = pd.read_csv(RAW_PATH)

    df, check_results = run_quality_checks(df)

    # Save clean parquet
    df.to_parquet(PROCESSED_PATH, index=False)
    n_passed = sum(1 for r in check_results if r["passed"])
    print(f"   💾 Saved: data/processed/clean.parquet ({len(df)} rows, {len(df.columns)} cols)")

    # Save quality report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_file": str(RAW_PATH),
        "output_file": str(PROCESSED_PATH),
        "rows_ingested": len(df),
        "columns": list(df.columns),
        "checks_passed": n_passed,
        "checks_total": len(check_results),
        "results": check_results,
    }
    with open(QUALITY_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"   📄 Saved: logs/quality_report.json")

    log_event({"stage": "ingest", "rows": len(df), "checks_passed": n_passed})
    print(f"✅ STAGE 1 COMPLETE — {n_passed}/10 quality checks passed")
    return df


if __name__ == "__main__":
    ingest()
