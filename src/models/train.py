"""Model training for transaction anomaly detection — IsolationForest."""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler

FEATURES_PATH = Path("data/processed/features.parquet")
SCHEMA_PATH = Path("data/processed/feature_schema.json")
MODEL_PATH = Path("models/pipeline_model.pkl")
METRICS_PATH = Path("models/pipeline_model_metrics.json")


class DataQualityError(Exception):
    """Raised on critical data issues."""


def train() -> dict:
    """Train IsolationForest on engineered features and save model + metrics.

    Returns:
        Metrics dictionary.
    """
    Path("models").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURES_PATH)
    schema = json.loads(SCHEMA_PATH.read_text())
    feature_cols = schema["feature_columns"]

    X = df[feature_cols].values
    y = df["is_anomaly"].values  # 1=anomaly, 0=normal

    # IsolationForest uses contamination as key hyperparameter
    # RandomizedSearchCV equivalent via ParameterSampler
    param_dist = {
        "n_estimators": [50, 100, 150, 200],
        "max_samples": ["auto", 0.5, 0.75, 1.0],
        "contamination": [0.1, 0.2, 0.3, 0.4],
    }
    print("   🔍 Running RandomizedSearchCV on 3 hyperparameters...")

    best_f1 = -1.0
    best_params = {}
    best_model = None

    rng = np.random.RandomState(42)
    for params in ParameterSampler(param_dist, n_iter=10, random_state=rng):
        model = IsolationForest(random_state=42, **params)
        model.fit(X)
        # IsolationForest: predict returns -1 (anomaly) or 1 (normal)
        preds = (model.predict(X) == -1).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_model = model

    print(f"   📈 Best params: {best_params}")

    # Final evaluation with best model
    preds = (best_model.predict(X) == -1).astype(int)
    scores = best_model.score_samples(X)
    # Normalize to [0,1] for anomaly probability
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    anomaly_probs = 1 - scores_norm  # higher = more anomalous

    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    try:
        auc = roc_auc_score(y, anomaly_probs)
    except Exception:
        auc = 0.0

    print(f"   📈 Precision: {precision:.4f}")
    print(f"   📈 Recall   : {recall:.4f}")
    print(f"   📈 F1 Score : {f1:.4f}")
    print(f"   📈 ROC-AUC  : {auc:.4f}")

    # Save model
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"model": best_model, "feature_cols": feature_cols}, fh)
    print(f"   💾 Saved: {MODEL_PATH}")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "IsolationForest",
        "best_params": {k: (v if not isinstance(v, float) else float(v)) for k, v in best_params.items()},
        "feature_columns": feature_cols,
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(auc),
        },
        "train_rows": int(len(X)),
        "anomaly_rate": float(y.mean()),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"   📄 Saved: {METRICS_PATH}")

    return metrics


if __name__ == "__main__":
    metrics = train()
    m = metrics["metrics"]
    print(f"✅ STAGE 3 COMPLETE — IsolationForest  ·  F1: {m['f1_score']:.4f}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
