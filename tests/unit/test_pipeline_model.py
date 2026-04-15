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
def model():
    """Load the trained model."""
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
    model = pickle.loads(MODEL_PATH.read_bytes())
    assert model is not None


def test_metrics_file_exists():
    """Metrics JSON file exists and contains required keys."""
    assert METRICS_PATH.exists(), "models/pipeline_model_metrics.json not found"
    metrics = json.loads(METRICS_PATH.read_text())
    for key in ("algorithm", "accuracy", "f1_score", "roc_auc", "best_params"):
        assert key in metrics, f"Missing key: {key}"


def test_predict_returns_binary(model, sample_features):
    """Model predict() returns only 0 or 1."""
    preds = model.predict(sample_features)
    assert set(preds).issubset({0, 1}), f"Unexpected labels: {set(preds)}"


def test_predict_proba_shape(model, sample_features):
    """predict_proba returns shape (n_samples, 2)."""
    proba = model.predict_proba(sample_features)
    assert proba.shape == (1, 2), f"Expected (1, 2), got {proba.shape}"


def test_predict_proba_sums_to_one(model, sample_features):
    """Predicted probabilities sum to 1 for each sample."""
    proba = model.predict_proba(sample_features)
    assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities do not sum to 1"


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
