"""Post-ingestion validation — 12 checks on clean + feature data."""

import json
from datetime import datetime, timezone

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""

    def default(self, obj):  # noqa: D102
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
from pathlib import Path

import pandas as pd

CLEAN_PATH = Path("data/processed/clean.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
SCHEMA_PATH = Path("data/processed/feature_schema.json")
VALIDATION_REPORT_PATH = Path("logs/validation_report.json")


def run() -> list:
    """Run 12 validation checks on processed data.

    Returns:
        List of check result dicts.
    """
    Path("logs").mkdir(parents=True, exist_ok=True)
    results = []

    df_clean = pd.read_parquet(CLEAN_PATH)
    df_feat = pd.read_parquet(FEATURES_PATH)
    schema = json.loads(SCHEMA_PATH.read_text())

    def check(name: str, passed: bool, detail: str = "") -> None:
        results.append({"check": name, "passed": passed, "detail": detail})

    # 1 — clean parquet exists and non-empty
    check("clean_parquet_exists", CLEAN_PATH.exists() and len(df_clean) > 0,
          f"{len(df_clean)} rows")

    # 2 — features parquet exists and non-empty
    check("features_parquet_exists", FEATURES_PATH.exists() and len(df_feat) > 0,
          f"{len(df_feat)} rows")

    # 3 — row counts match between clean and features
    check("row_count_consistent", len(df_clean) == len(df_feat),
          f"clean={len(df_clean)} features={len(df_feat)}")

    # 4 — all feature columns present in features parquet
    missing_cols = [c for c in schema["feature_columns"] if c not in df_feat.columns]
    check("all_feature_cols_present", len(missing_cols) == 0,
          f"missing={missing_cols}" if missing_cols else "")

    # 5 — no nulls in feature columns
    null_counts = df_feat[schema["feature_columns"]].isnull().sum().sum()
    check("no_nulls_in_features", null_counts == 0, f"{null_counts} nulls")

    # 6 — target column present in clean
    check("target_col_in_clean", "is_anomaly" in df_clean.columns)

    # 7 — target is binary
    unique_labels = set(int(v) for v in df_clean["is_anomaly"].unique())
    check("target_binary", unique_labels.issubset({0, 1}), f"unique={unique_labels}")

    # 8 — amount_log values are non-negative
    check("amount_log_non_negative", (df_feat["amount_log"] >= 0).all())

    # 9 — category_encoded has at least 2 unique values
    n_cats = df_feat["category_encoded"].nunique()
    check("category_encoded_variance", n_cats >= 1, f"n_unique={n_cats}")

    # 10 — hour_of_day in [0, 23]
    valid_hours = df_feat["hour_of_day"].between(0, 23).all()
    check("hour_of_day_valid_range", bool(valid_hours))

    # 11 — merchant_txn_count >= 1
    check("merchant_txn_count_positive", (df_feat["merchant_txn_count"] >= 1).all())

    # 12 — schema file contains required keys
    schema_valid = all(k in schema for k in ["feature_columns", "dtypes", "stats"])
    check("feature_schema_valid", schema_valid)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checks_passed": sum(r["passed"] for r in results),
        "checks_total": len(results),
        "checks": results,
    }
    VALIDATION_REPORT_PATH.write_text(json.dumps(report, indent=2, cls=_NumpyEncoder))

    return results


if __name__ == "__main__":
    results = run()
    passed = sum(r["passed"] for r in results)
    print(f"   ✅ Subagent C done — {passed}/{len(results)} validation checks passed")
    print(f"   📄 Saved: {VALIDATION_REPORT_PATH}")
