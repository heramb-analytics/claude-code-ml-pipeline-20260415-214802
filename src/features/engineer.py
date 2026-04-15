"""Feature engineering for transaction anomaly detection pipeline."""

import json
from pathlib import Path

import pandas as pd

CLEAN_PATH = Path("data/processed/clean.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
SCHEMA_PATH = Path("data/processed/feature_schema.json")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from cleaned transaction data.

    Args:
        df: Cleaned transaction dataframe.

    Returns:
        Dataframe with engineered features.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Temporal features
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_business_hours"] = df["hour_of_day"].between(9, 17).astype(int)

    # Amount features
    df["amount_log"] = df["amount"].apply(lambda x: __import__("math").log1p(x))
    amount_mean = df["amount"].mean()
    amount_std = df["amount"].std() if df["amount"].std() > 0 else 1.0
    df["amount_zscore"] = (df["amount"] - amount_mean) / amount_std
    df["is_high_value"] = (df["amount"] > df["amount"].quantile(0.75)).astype(int)

    # Category encoding
    df["category_encoded"] = df["category"].astype("category").cat.codes

    # Merchant features
    merchant_txn_counts = df.groupby("merchant_id")["transaction_id"].transform("count")
    df["merchant_txn_count"] = merchant_txn_counts

    return df


def save_schema(df: pd.DataFrame, feature_cols: list) -> None:
    """Save feature schema to JSON.

    Args:
        df: Feature dataframe.
        feature_cols: List of feature column names used for modeling.
    """
    schema = {
        "feature_columns": feature_cols,
        "dtypes": {col: str(df[col].dtype) for col in feature_cols},
        "stats": {
            col: {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
            for col in feature_cols
            if pd.api.types.is_numeric_dtype(df[col])
        },
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2))


def run() -> tuple[pd.DataFrame, list]:
    """Load clean data, engineer features, save outputs.

    Returns:
        Tuple of (feature dataframe, feature column list).
    """
    df = pd.read_parquet(CLEAN_PATH)
    df_feat = engineer_features(df)

    feature_cols = [
        "hour_of_day", "day_of_week", "is_weekend", "is_business_hours",
        "amount_log", "amount_zscore", "is_high_value",
        "category_encoded", "merchant_txn_count",
    ]

    df_feat.to_parquet(FEATURES_PATH, index=False)
    save_schema(df_feat, feature_cols)

    return df_feat, feature_cols


if __name__ == "__main__":
    df_feat, feature_cols = run()
    print(f"   ✅ Subagent A done — {len(feature_cols)} features engineered")
    print(f"   💾 Saved: {FEATURES_PATH}")
    print(f"   📄 Saved: {SCHEMA_PATH}")
