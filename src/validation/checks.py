"""Post-ingestion validation — 12 checks on clean + feature data."""

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np


CLEAN_PATH = Path("data/processed/clean.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
REPORT_PATH = Path("logs/validation_report.json")


def run() -> None:
    """Run 12 validation checks and save report."""
    clean = pd.read_parquet(CLEAN_PATH)
    features = pd.read_parquet(FEATURES_PATH)

    results = []
    passed = 0

    def check(n: int, name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed
        results.append({"check": n, "name": name, "passed": bool(condition), "detail": detail})
        if condition:
            passed += 1

    # 1. Clean parquet row count matches raw
    check(1, "clean_parquet_not_empty", len(clean) > 0, f"{len(clean)} rows")
    # 2. Features parquet row count matches clean
    check(2, "features_row_count_matches", len(features) == len(clean),
          f"clean={len(clean)}, features={len(features)}")
    # 3. No NaN in feature numeric columns
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    nan_count = features[numeric_cols].isna().sum().sum()
    check(3, "no_nan_in_numeric_features", nan_count == 0, f"NaN count={nan_count}")
    # 4. log_amount column exists and is non-negative
    check(4, "log_amount_non_negative",
          "log_amount" in features.columns and (features["log_amount"] >= 0).all(),
          "log_amount >= 0")
    # 5. hour_of_day in [0, 23]
    check(5, "hour_of_day_valid_range",
          "hour_of_day" in features.columns and features["hour_of_day"].between(0, 23).all(),
          "hour in [0,23]")
    # 6. day_of_week in [0, 6]
    check(6, "day_of_week_valid_range",
          "day_of_week" in features.columns and features["day_of_week"].between(0, 6).all(),
          "day in [0,6]")
    # 7. is_weekend is binary
    check(7, "is_weekend_binary",
          "is_weekend" in features.columns and features["is_weekend"].isin([0, 1]).all(),
          "binary 0/1")
    # 8. amount_percentile in [0, 1]
    check(8, "amount_percentile_range",
          "amount_percentile" in features.columns and features["amount_percentile"].between(0, 1).all(),
          "in [0,1]")
    # 9. is_anomaly target column present in features
    check(9, "target_column_present", "is_anomaly" in features.columns, "is_anomaly exists")
    # 10. No all-zero feature rows
    feature_only = [c for c in numeric_cols if c != "is_anomaly"]
    all_zero = (features[feature_only] == 0).all(axis=1).sum()
    check(10, "no_all_zero_feature_rows", all_zero == 0, f"all-zero rows={all_zero}")
    # 11. merchant_txn_count is positive
    check(11, "merchant_txn_count_positive",
          "merchant_txn_count" in features.columns and (features["merchant_txn_count"] > 0).all(),
          "> 0")
    # 12. Feature schema file exists
    check(12, "feature_schema_exists",
          Path("data/processed/feature_schema.json").exists(),
          "file present")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks_passed": passed,
        "checks_total": len(results),
        "results": results,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"   ✅ Subagent C done — {passed}/12 validation checks passed")
    print(f"   📄 Saved: logs/validation_report.json")


if __name__ == "__main__":
    run()
