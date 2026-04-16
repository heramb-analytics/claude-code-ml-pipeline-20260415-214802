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
CLEAN_PATH = Path("data/processed/clean.parquet")


@pytest.fixture(scope="module")
def model_bundle():
    """Load the saved model bundle."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def features_df():
    """Load the features DataFrame."""
    return pd.read_parquet(FEATURES_PATH)


def test_model_load(model_bundle):
    """Test that model bundle loads correctly with expected keys."""
    assert "model" in model_bundle
    assert "feature_cols" in model_bundle
    assert model_bundle["model"] is not None


def test_predict_schema(model_bundle, features_df):
    """Test that predict returns correct shape and binary values."""
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    X = features_df[feature_cols].values.astype(float)
    raw_preds = model.predict(X)
    # IsolationForest returns -1 (anomaly) or 1 (normal)
    assert raw_preds.shape == (len(X),)
    assert set(raw_preds).issubset({-1, 1})


def test_model_metrics_file_exists():
    """Test that metrics JSON file exists and has required keys."""
    assert METRICS_PATH.exists(), f"Metrics file missing: {METRICS_PATH}"
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    assert "algorithm" in metrics
    assert "metrics" in metrics
    assert "f1_score" in metrics["metrics"]


def test_feature_columns_match(model_bundle, features_df):
    """Test that saved feature columns exist in the features DataFrame."""
    feature_cols = model_bundle["feature_cols"]
    for col in feature_cols:
        assert col in features_df.columns, f"Feature column missing: {col}"


def test_clean_parquet_exists():
    """Test that clean.parquet was saved by Stage 1."""
    assert CLEAN_PATH.exists(), f"Clean parquet missing: {CLEAN_PATH}"
    df = pd.read_parquet(CLEAN_PATH)
    assert len(df) > 0
    assert "is_anomaly" in df.columns


def test_no_nan_in_features(model_bundle, features_df):
    """Test that feature matrix has no NaN values."""
    feature_cols = model_bundle["feature_cols"]
    X = features_df[feature_cols]
    assert not X.isna().any().any(), "NaN values found in feature matrix"


def test_decision_function_returns_scores(model_bundle, features_df):
    """Test that decision_function returns numeric scores."""
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    X = features_df[feature_cols].values.astype(float)
    scores = model.decision_function(X)
    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()


def test_prediction_output_type(model_bundle, features_df):
    """Test end-to-end prediction produces expected binary output."""
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    X = features_df[feature_cols].values.astype(float)
    raw_preds = model.predict(X)
    y_pred = np.where(raw_preds == -1, 1, 0)
    assert y_pred.dtype in (np.int64, np.int32, int, np.intp)
    assert set(y_pred).issubset({0, 1})
