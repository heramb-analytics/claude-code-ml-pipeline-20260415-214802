"""Unit tests for the transaction anomaly detection pipeline."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODEL_PATH = Path("models/pipeline_model.pkl")
METRICS_PATH = Path("models/pipeline_model_metrics.json")
FEATURES_PATH = Path("data/processed/features.parquet")
SCHEMA_PATH = Path("data/processed/feature_schema.json")
CLEAN_PATH = Path("data/processed/clean.parquet")
QUALITY_REPORT_PATH = Path("logs/quality_report.json")


@pytest.fixture(scope="module")
def model_bundle():
    """Load the trained model bundle dict."""
    return pickle.loads(MODEL_PATH.read_bytes())


@pytest.fixture(scope="module")
def schema():
    """Load the feature schema."""
    return json.loads(SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def metrics():
    """Load the model metrics."""
    return json.loads(METRICS_PATH.read_text())


@pytest.fixture(scope="module")
def sample_features(schema):
    """Return a sample feature array from the features parquet."""
    df = pd.read_parquet(FEATURES_PATH)
    return df[schema["feature_columns"]].values[:1]


def test_model_loads():
    """Model pickle file exists and loads without error."""
    assert MODEL_PATH.exists(), "models/pipeline_model.pkl not found"
    bundle = pickle.loads(MODEL_PATH.read_bytes())
    assert "model" in bundle, "model bundle missing 'model' key"
    assert bundle["model"] is not None


def test_metrics_file_exists():
    """Metrics JSON file exists and contains required inner keys."""
    assert METRICS_PATH.exists(), "models/pipeline_model_metrics.json not found"
    m = json.loads(METRICS_PATH.read_text())
    assert "algorithm" in m
    assert "best_params" in m
    assert "metrics" in m
    for key in ("precision", "recall", "f1_score", "roc_auc"):
        assert key in m["metrics"], f"Missing metrics key: {key}"


def test_predict_returns_binary(model_bundle, sample_features):
    """Model predict() maps to binary 0/1 labels."""
    clf = model_bundle["model"]
    raw = clf.predict(sample_features)  # returns -1 or 1
    preds = (raw == -1).astype(int)
    assert set(preds).issubset({0, 1}), f"Unexpected labels: {set(preds)}"


def test_score_samples_shape(model_bundle, sample_features):
    """score_samples returns an array of shape (n_samples,)."""
    clf = model_bundle["model"]
    scores = clf.score_samples(sample_features)
    assert scores.shape == (1,), f"Expected shape (1,), got {scores.shape}"


def test_anomaly_scores_finite(model_bundle, sample_features):
    """Anomaly scores are finite (no NaN / Inf)."""
    clf = model_bundle["model"]
    scores = clf.score_samples(sample_features)
    assert np.all(np.isfinite(scores)), "Anomaly scores contain NaN or Inf"


def test_clean_parquet_schema():
    """clean.parquet has the expected columns."""
    df = pd.read_parquet(CLEAN_PATH)
    required = {"transaction_id", "timestamp", "merchant_id", "amount", "category", "is_anomaly"}
    assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"


def test_feature_schema_valid(schema):
    """Feature schema has required keys and non-empty feature list."""
    assert "feature_columns" in schema
    assert "dtypes" in schema
    assert len(schema["feature_columns"]) > 0


def test_quality_report_all_checks_passed():
    """Quality report records 10/10 checks passed."""
    report = json.loads(QUALITY_REPORT_PATH.read_text())
    assert report["checks_passed"] == report["checks_total"], (
        f"Only {report['checks_passed']}/{report['checks_total']} quality checks passed"
    )
