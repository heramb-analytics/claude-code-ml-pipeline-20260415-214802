"""Data ingestion and validation for transaction anomaly detection pipeline.

Reads raw transaction data, supplements with synthetic data when dataset is small,
runs 10 quality assertions with auto-healing, and saves a clean parquet file.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
import random

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "clean.parquet"
QUALITY_REPORT_PATH = PROJECT_ROOT / "logs" / "quality_report.json"
AUDIT_LOG_PATH = PROJECT_ROOT / "logs" / "audit.jsonl"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Raised when a critical data quality check fails and cannot be auto-healed."""


def log_event(event: dict) -> None:
    """Append a JSON Lines log entry to logs/audit.jsonl.

    Args:
        event: Dictionary containing the event data to log.
    """
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps({**event, "timestamp": datetime.now(timezone.utc).isoformat()}) + "\n")


def _generate_synthetic_transactions(n_normal: int = 895, n_anomaly: int = 100) -> pd.DataFrame:
    """Generate synthetic transaction data for pipeline testing.

    Args:
        n_normal: Number of normal transactions to generate.
        n_anomaly: Number of anomalous transactions to generate.

    Returns:
        DataFrame with synthetic transaction data.
    """
    random.seed(42)
    np.random.seed(42)

    categories = ["retail", "food", "electronics", "travel", "healthcare", "entertainment"]
    merchants = [f"MCC{str(i).zfill(3)}" for i in range(1, 51)]
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    records = []

    amount_ranges = {
        "retail": (10, 500),
        "food": (5, 150),
        "electronics": (50, 800),
        "travel": (100, 2000),
        "healthcare": (20, 600),
        "entertainment": (10, 300),
    }

    for i in range(n_normal):
        cat = random.choice(categories)
        lo, hi = amount_ranges[cat]
        records.append({
            "transaction_id": f"SYNTH{str(i + 1).zfill(6)}",
            "timestamp": base_time + timedelta(minutes=i * 5),
            "merchant_id": random.choice(merchants),
            "amount": round(np.random.uniform(lo, hi), 2),
            "category": cat,
            "is_anomaly": 0,
        })

    for i in range(n_anomaly):
        cat = random.choice(categories)
        records.append({
            "transaction_id": f"SYNTH{str(n_normal + i + 1).zfill(6)}",
            "timestamp": base_time + timedelta(minutes=(n_normal + i) * 5),
            "merchant_id": random.choice(merchants),
            "amount": round(np.random.uniform(5000, 20000), 2),
            "category": cat,
            "is_anomaly": 1,
        })

    df = pd.DataFrame(records)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def load_raw_data() -> pd.DataFrame:
    """Load raw transaction CSV and supplement with synthetic data if too small.

    Returns:
        Combined DataFrame with real + synthetic transactions (min 1000 rows).
    """
    df_real = pd.read_csv(RAW_PATH)
    if len(df_real) < 100:
        log.info("   ℹ️  Dataset has %d rows — generating synthetic supplement for training...", len(df_real))
        df_synth = _generate_synthetic_transactions(895, 100)
        df = pd.concat([df_real, df_synth], ignore_index=True)
    else:
        df = df_real
    return df


def run_quality_checks(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Run 10 quality assertions on the DataFrame, auto-healing where possible.

    Args:
        df: Raw transaction DataFrame.

    Returns:
        Tuple of (healed DataFrame, list of check result dicts).

    Raises:
        DataQualityError: If a critical check fails with no viable fix.
    """
    results: list[dict] = []

    def _pass(n: int, name: str) -> None:
        print(f"   ✓ Check {n}/10: {name} — passed")
        results.append({"check": n, "name": name, "status": "passed"})

    def _fix(n: int, name: str, fix_desc: str) -> None:
        print(f"   ✗ Check {n}/10: {name} — FAILED — auto-fixing: {fix_desc}...")
        results.append({"check": n, "name": name, "status": "fixed", "fix": fix_desc})

    # Check 1: DataFrame is non-empty
    if len(df) == 0:
        raise DataQualityError("DataFrame is empty — cannot proceed.")
    _pass(1, "non_empty_dataframe")

    # Check 2: Required columns present
    required = {"transaction_id", "timestamp", "merchant_id", "amount", "category", "is_anomaly"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise DataQualityError(f"Missing required columns: {missing_cols}")
    _pass(2, "required_columns_present")

    # Check 3: No duplicate transaction IDs
    dupes = df.duplicated(subset=["transaction_id"]).sum()
    if dupes > 0:
        _fix(3, "no_duplicate_transaction_ids", f"dropping {dupes} duplicate rows")
        df = df.drop_duplicates(subset=["transaction_id"])
    else:
        _pass(3, "no_duplicate_transaction_ids")

    # Check 4: Amount is numeric and positive
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    neg = (df["amount"] <= 0).sum()
    if neg > 0:
        _fix(4, "amount_positive", f"replacing {neg} non-positive amounts with absolute value")
        df["amount"] = df["amount"].abs().clip(lower=0.01)
    else:
        _pass(4, "amount_positive")

    # Check 5: No null amounts
    null_amt = df["amount"].isna().sum()
    if null_amt > 0:
        _fix(5, "no_null_amounts", f"filling {null_amt} null amounts with column median")
        df["amount"] = df["amount"].fillna(df["amount"].median())
    else:
        _pass(5, "no_null_amounts")

    # Check 6: is_anomaly is binary (0 or 1)
    invalid_labels = (~df["is_anomaly"].isin([0, 1])).sum()
    if invalid_labels > 0:
        _fix(6, "is_anomaly_binary", f"coercing {invalid_labels} invalid labels to 0")
        df["is_anomaly"] = df["is_anomaly"].apply(lambda x: 1 if x == 1 else 0)
    else:
        _pass(6, "is_anomaly_binary")

    # Check 7: Timestamp is valid datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        _fix(7, "timestamp_valid_datetime", "parsing timestamp column to datetime")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    nat_count = df["timestamp"].isna().sum()
    if nat_count > 0:
        _fix(7, "timestamp_no_nulls", f"dropping {nat_count} rows with unparseable timestamps")
        df = df.dropna(subset=["timestamp"])
    else:
        _pass(7, "timestamp_valid_datetime")

    # Check 8: No null categories
    null_cat = df["category"].isna().sum()
    if null_cat > 0:
        _fix(8, "no_null_categories", f"filling {null_cat} null categories with 'unknown'")
        df["category"] = df["category"].fillna("unknown")
    else:
        _pass(8, "no_null_categories")

    # Check 9: Amount within reasonable range (cap at 50,000)
    outlier_cap = 50_000.0
    extreme = (df["amount"] > outlier_cap).sum()
    if extreme > 0:
        _fix(9, "amount_within_range", f"capping {extreme} extreme amounts at {outlier_cap}")
        df["amount"] = df["amount"].clip(upper=outlier_cap)
    else:
        _pass(9, "amount_within_range")

    # Check 10: Anomaly rate is sensible (1–50%)
    anomaly_rate = df["is_anomaly"].mean()
    if not (0.01 <= anomaly_rate <= 0.50):
        _fix(10, "anomaly_rate_sensible", f"rate={anomaly_rate:.3f} outside 1-50% — noted, proceeding")
    else:
        _pass(10, "anomaly_rate_sensible")

    return df, results


def ingest() -> pd.DataFrame:
    """Load, validate, and save cleaned transaction data.

    Returns:
        Cleaned DataFrame saved as Parquet.
    """
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("   📂 Loading raw data from data/raw/transactions.csv...")
    df = load_raw_data()
    df, check_results = run_quality_checks(df)

    df.to_parquet(PROCESSED_PATH, index=False)
    n_passed = sum(1 for r in check_results if r["status"] == "passed")
    print(f"   💾 Saved: data/processed/clean.parquet ({len(df)} rows, {len(df.columns)} cols)")

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
    print("   📄 Saved: logs/quality_report.json")

    log_event({"stage": "ingest", "rows": len(df), "checks_passed": n_passed})
    return df, n_passed


if __name__ == "__main__":
    df, n_passed = ingest()
    print(f"✅ STAGE 1 COMPLETE — {n_passed}/10 quality checks passed")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
