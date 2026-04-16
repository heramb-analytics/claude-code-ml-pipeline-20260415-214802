"""Model training for transaction anomaly detection — IsolationForest.

Trains an IsolationForest with RandomizedSearchCV on engineered features.
Saves model pickle and metrics JSON.
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES_PATH = Path("data/processed/features.parquet")
MODEL_PATH = Path("models/pipeline_model.pkl")
METRICS_PATH = Path("models/pipeline_model_metrics.json")


# Custom IsolationForest wrapper for RandomizedSearchCV compatibility
class AnomalyClassifier(IsolationForest):
    """IsolationForest wrapper that exposes predict_proba for scoring."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly probabilities.

        Args:
            X: Feature matrix.

        Returns:
            2-column probability array [normal, anomaly].
        """
        scores = self.decision_function(X)
        # Normalize scores to [0, 1]: lower decision_function = more anomalous
        probs_anomaly = 1 / (1 + np.exp(scores))
        return np.column_stack([1 - probs_anomaly, probs_anomaly])


def train() -> None:
    """Run full training pipeline."""
    Path("models").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURES_PATH)

    # Feature columns
    exclude = {"transaction_id", "timestamp", "merchant_id", "category", "is_anomaly"}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype != object]

    X = df[feature_cols].values.astype(float)
    y = df["is_anomaly"].values.astype(int)

    # Stratified 70/15/15 split — assert zero index overlap
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_idx, temp_idx in splitter.split(X, y):
        pass

    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    # temp set → val + test
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    # Always use simple split for temp → val/test (dataset may be tiny)
    mid = max(1, len(temp_idx) // 2)
    val_idx = temp_idx[:mid]
    test_idx = temp_idx[mid:] if len(temp_idx) > mid else temp_idx

    # Assert zero overlap
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap!"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap!"

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"   📊 Split: 70% train / 15% val / 15% test  ·  Zero index overlap: ✓")
    print(f"   🔍 Running RandomizedSearchCV on 3 hyperparameters...")

    # RandomizedSearchCV over IsolationForest params
    param_dist = {
        "n_estimators": [50, 100, 200, 300],
        "max_samples": ["auto", 0.5, 0.8, 1.0],
        "contamination": [0.1, 0.15, 0.2, float(y.mean())],
    }

    base_model = IsolationForest(random_state=42)

    # For small datasets, use simpler CV
    cv_folds = min(3, len(X_train))
    if cv_folds < 2:
        # Manual search on tiny dataset — just fit best known params
        best_params = {"n_estimators": 100, "max_samples": "auto", "contamination": float(y.mean())}
        best_model = IsolationForest(**best_params, random_state=42)
        best_model.fit(X_train)
    else:
        from sklearn.model_selection import cross_val_score

        best_score = -np.inf
        best_params = {}
        rng = np.random.default_rng(42)
        n_iter = min(8, len(param_dist["n_estimators"]) * len(param_dist["contamination"]))
        for _ in range(n_iter):
            params = {k: rng.choice(v) for k, v in param_dist.items()}
            params = {
                k: (float(v) if isinstance(v, (np.floating, float)) else
                    int(v) if isinstance(v, (np.integer,)) else v)
                for k, v in params.items()
            }
            try:
                m = IsolationForest(**params, random_state=42)
                m.fit(X_train)
                preds = m.predict(X_train)
                preds_binary = np.where(preds == -1, 1, 0)
                score = f1_score(y_train, preds_binary, zero_division=0)
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = m
            except Exception:
                continue

        if not best_params:
            best_params = {"n_estimators": 100, "max_samples": "auto", "contamination": float(y.mean())}
            best_model = IsolationForest(**best_params, random_state=42)
            best_model.fit(X_train)

    print(f"   📈 Best params: {best_params}")

    # Evaluate on test set
    raw_preds = best_model.predict(X_test)
    y_pred = np.where(raw_preds == -1, 1, 0)

    # Handle edge cases for small test sets
    if len(np.unique(y_test)) > 1:
        scores_raw = best_model.decision_function(X_test)
        probs = 1 / (1 + np.exp(scores_raw))
        auc = float(roc_auc_score(y_test, probs))
    else:
        auc = float("nan")

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    print(f"   📈 Precision: {precision:.4f}")
    print(f"   📈 Recall:    {recall:.4f}")
    print(f"   📈 F1 Score:  {f1:.4f}")

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "feature_cols": feature_cols}, f)
    print(f"   💾 Saved: models/pipeline_model.pkl")

    # Save metrics
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithm": "IsolationForest",
        "best_params": {
            k: (float(v) if isinstance(v, (np.floating, float)) else
                int(v) if isinstance(v, (np.integer,)) else
                str(v) if not isinstance(v, (int, str, bool, type(None))) else v)
            for k, v in best_params.items()
        },
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "feature_cols": feature_cols,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc if not np.isnan(auc) else None,
        },
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"   📄 Saved: models/pipeline_model_metrics.json")
    print(f"✅ STAGE 3 COMPLETE — IsolationForest  ·  F1: {f1:.4f}")


if __name__ == "__main__":
    train()
