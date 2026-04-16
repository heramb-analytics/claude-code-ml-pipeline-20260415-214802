"""Feature engineering module for transaction anomaly detection.

Generates numerical features from cleaned transaction data.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np


CLEAN_PATH = Path("data/processed/clean.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
SCHEMA_PATH = Path("data/processed/feature_schema.json")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features from cleaned transaction DataFrame.

    Args:
        df: Cleaned DataFrame from Stage 1.

    Returns:
        DataFrame with engineered features.
    """
    df = df.copy()

    # Temporal features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month

    # Amount features
    df["log_amount"] = np.log1p(df["amount"])
    df["amount_zscore"] = (df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-9)
    df["amount_percentile"] = df["amount"].rank(pct=True)
    df["is_large_txn"] = (df["amount"] > df["amount"].quantile(0.75)).astype(int)
    df["is_round_amount"] = (df["amount"] % 100 == 0).astype(int)

    # Category features
    category_dummies = pd.get_dummies(df["category"], prefix="cat").astype(int)
    df = pd.concat([df, category_dummies], axis=1)

    # Merchant features
    merchant_txn_count = df.groupby("merchant_id")["transaction_id"].transform("count")
    df["merchant_txn_count"] = merchant_txn_count

    # Amount deviation from merchant average
    merchant_avg = df.groupby("merchant_id")["amount"].transform("mean")
    df["amount_vs_merchant_avg"] = df["amount"] - merchant_avg

    # Velocity feature: transactions per hour slot
    df["hour_txn_volume"] = df.groupby("hour_of_day")["transaction_id"].transform("count")

    return df


def save_features(df: pd.DataFrame) -> list[str]:
    """Save feature DataFrame and schema.

    Args:
        df: Feature-engineered DataFrame.

    Returns:
        List of feature column names.
    """
    # Feature columns (exclude raw/id columns)
    exclude = {"transaction_id", "timestamp", "merchant_id", "category", "is_anomaly"}
    feature_cols = [c for c in df.columns if c not in exclude]

    df.to_parquet(FEATURES_PATH, index=False)

    schema = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_features": len(feature_cols),
        "feature_columns": feature_cols,
        "all_columns": list(df.columns),
        "rows": len(df),
    }
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)

    return feature_cols


def run() -> None:
    """Execute feature engineering pipeline."""
    df = pd.read_parquet(CLEAN_PATH)
    df_features = engineer_features(df)
    feature_cols = save_features(df_features)
    print(f"   ✅ Subagent A done — {len(feature_cols)} features engineered")
    print(f"   💾 Saved: data/processed/features.parquet")
    print(f"   📄 Saved: data/processed/feature_schema.json")


if __name__ == "__main__":
    run()
