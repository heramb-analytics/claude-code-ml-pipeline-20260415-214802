"""FastAPI application for transaction anomaly detection pipeline."""

import json
import pickle
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ── Model loading ────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/pipeline_model.pkl")
METRICS_PATH = Path("models/pipeline_model_metrics.json")

_model_bundle: dict | None = None
_metrics: dict | None = None
_recent_predictions: deque = deque(maxlen=10)

START_TIME = datetime.now(timezone.utc)


def load_model() -> dict:
    """Load model bundle from disk (cached after first load).

    Returns:
        Dict with 'model' and 'feature_cols' keys.
    """
    global _model_bundle
    if _model_bundle is None:
        with open(MODEL_PATH, "rb") as f:
            _model_bundle = pickle.load(f)
    return _model_bundle


def load_metrics() -> dict:
    """Load metrics JSON (cached after first load).

    Returns:
        Metrics dictionary.
    """
    global _metrics
    if _metrics is None:
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
    return _metrics


# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Transaction Anomaly Detection API",
    description="IsolationForest-based anomaly detection for financial transactions",
    version="1.0.0",
)


# ── Request/Response schemas ─────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    """Input schema for a single transaction prediction."""

    amount: float = 150.0
    hour_of_day: int = 9
    day_of_week: int = 0
    is_weekend: int = 0
    month: int = 1
    merchant_txn_count: int = 2
    amount_vs_merchant_avg: float = 0.0
    hour_txn_volume: int = 1


class PredictionResponse(BaseModel):
    """Prediction output with anomaly score and label."""

    request_id: str
    timestamp: str
    is_anomaly: int
    anomaly_score: float
    confidence: float
    input: dict[str, Any]


# ── Helpers ──────────────────────────────────────────────────────────────────
def build_feature_vector(req: TransactionRequest, feature_cols: list[str]) -> np.ndarray:
    """Convert request into feature vector aligned to model's expected columns.

    Args:
        req: Transaction request object.
        feature_cols: List of feature column names model was trained on.

    Returns:
        1-D numpy array of feature values.
    """
    # Build a row dict with all possible features set to defaults
    row = {
        "amount": req.amount,
        "hour_of_day": req.hour_of_day,
        "day_of_week": req.day_of_week,
        "is_weekend": req.is_weekend,
        "month": req.month,
        "log_amount": np.log1p(req.amount),
        "amount_zscore": 0.0,
        "amount_percentile": 0.5,
        "is_large_txn": int(req.amount > 500),
        "is_round_amount": int(req.amount % 100 == 0),
        "merchant_txn_count": req.merchant_txn_count,
        "amount_vs_merchant_avg": req.amount_vs_merchant_avg,
        "hour_txn_volume": req.hour_txn_volume,
        # category dummies — default 0
        "cat_electronics": 0,
        "cat_food": 0,
        "cat_retail": 0,
    }
    return np.array([row.get(c, 0.0) for c in feature_cols], dtype=float)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the HTML dashboard UI."""
    metrics = load_metrics()
    bundle = load_model()
    uptime = (datetime.now(timezone.utc) - START_TIME).seconds

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Transaction Anomaly Detection</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .pulse {{ animation: pulse 2s infinite; }}
    @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.5}} }}
  </style>
</head>
<body class="bg-gray-900 text-white min-h-screen p-6">
  <div class="max-w-5xl mx-auto">
    <!-- Header -->
    <div class="flex items-center justify-between mb-8">
      <div>
        <h1 class="text-3xl font-bold text-indigo-400">🔍 Transaction Anomaly Detection</h1>
        <p class="text-gray-400 mt-1">IsolationForest · Real-time Fraud Scoring</p>
      </div>
      <div class="flex items-center gap-2">
        <div class="w-3 h-3 bg-green-400 rounded-full pulse"></div>
        <span class="text-green-400 text-sm font-medium">LIVE · Uptime {uptime}s</span>
      </div>
    </div>

    <!-- Metrics cards -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
      <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <p class="text-gray-400 text-xs uppercase tracking-wide">Algorithm</p>
        <p class="text-white font-bold text-lg mt-1">{metrics.get('algorithm','IsolationForest')}</p>
      </div>
      <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <p class="text-gray-400 text-xs uppercase tracking-wide">F1 Score</p>
        <p class="text-green-400 font-bold text-lg mt-1">{metrics['metrics'].get('f1_score', 0):.4f}</p>
      </div>
      <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <p class="text-gray-400 text-xs uppercase tracking-wide">Precision</p>
        <p class="text-blue-400 font-bold text-lg mt-1">{metrics['metrics'].get('precision', 0):.4f}</p>
      </div>
      <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <p class="text-gray-400 text-xs uppercase tracking-wide">Recall</p>
        <p class="text-yellow-400 font-bold text-lg mt-1">{metrics['metrics'].get('recall', 0):.4f}</p>
      </div>
    </div>

    <!-- Prediction form -->
    <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 mb-8">
      <h2 class="text-xl font-semibold mb-4 text-indigo-300">🔮 Predict Transaction</h2>
      <form id="predForm" class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <label class="text-gray-400 text-xs block mb-1">Amount ($)</label>
          <input type="number" step="0.01" name="amount" value="9999.99" class="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-indigo-400"/>
        </div>
        <div>
          <label class="text-gray-400 text-xs block mb-1">Hour of Day</label>
          <input type="number" min="0" max="23" name="hour_of_day" value="2" class="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-indigo-400"/>
        </div>
        <div>
          <label class="text-gray-400 text-xs block mb-1">Day of Week</label>
          <input type="number" min="0" max="6" name="day_of_week" value="6" class="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-indigo-400"/>
        </div>
        <div>
          <label class="text-gray-400 text-xs block mb-1">Merchant Txn Count</label>
          <input type="number" name="merchant_txn_count" value="1" class="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-indigo-400"/>
        </div>
        <div class="col-span-2 md:col-span-4 flex gap-3 mt-2">
          <button type="submit" class="bg-indigo-600 hover:bg-indigo-500 text-white font-medium px-6 py-2 rounded-lg transition-colors">
            ▶ Predict
          </button>
          <div id="badge" class="hidden px-4 py-2 rounded-lg font-bold text-sm flex items-center"></div>
        </div>
      </form>
    </div>

    <!-- Recent predictions table -->
    <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
      <h2 class="text-xl font-semibold mb-4 text-indigo-300">📋 Last 10 Predictions</h2>
      <div id="predTable" class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-gray-400 border-b border-gray-700">
              <th class="text-left pb-2">Request ID</th>
              <th class="text-left pb-2">Amount</th>
              <th class="text-left pb-2">Score</th>
              <th class="text-left pb-2">Result</th>
              <th class="text-left pb-2">Time</th>
            </tr>
          </thead>
          <tbody id="predRows" class="divide-y divide-gray-700">
            <tr><td colspan="5" class="text-gray-500 text-center py-4">No predictions yet</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Footer links -->
    <div class="mt-6 flex gap-4 text-sm text-gray-500">
      <a href="/docs" class="hover:text-indigo-400">📖 Swagger Docs</a>
      <a href="/metrics" class="hover:text-indigo-400">📊 Metrics</a>
      <a href="/health" class="hover:text-indigo-400">❤️ Health</a>
    </div>
  </div>

  <script>
    const predictions = [];

    document.getElementById('predForm').addEventListener('submit', async (e) => {{
      e.preventDefault();
      const fd = new FormData(e.target);
      const payload = {{
        amount: parseFloat(fd.get('amount')),
        hour_of_day: parseInt(fd.get('hour_of_day')),
        day_of_week: parseInt(fd.get('day_of_week')),
        merchant_txn_count: parseInt(fd.get('merchant_txn_count')),
        is_weekend: parseInt(fd.get('day_of_week')) >= 5 ? 1 : 0,
        month: new Date().getMonth() + 1,
        amount_vs_merchant_avg: 0,
        hour_txn_volume: 1
      }};

      try {{
        const res = await fetch('/predict', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify(payload)
        }});
        const data = await res.json();
        const badge = document.getElementById('badge');
        badge.classList.remove('hidden');
        if (data.is_anomaly === 1) {{
          badge.className = 'px-4 py-2 rounded-lg font-bold text-sm flex items-center bg-red-900 text-red-300 border border-red-700';
          badge.textContent = '🚨 ANOMALY DETECTED (score: ' + data.anomaly_score.toFixed(4) + ')';
        }} else {{
          badge.className = 'px-4 py-2 rounded-lg font-bold text-sm flex items-center bg-green-900 text-green-300 border border-green-700';
          badge.textContent = '✅ NORMAL (score: ' + data.anomaly_score.toFixed(4) + ')';
        }}

        predictions.unshift(data);
        if (predictions.length > 10) predictions.pop();
        renderTable();
      }} catch(err) {{
        console.error(err);
      }}
    }});

    function renderTable() {{
      const tbody = document.getElementById('predRows');
      if (!predictions.length) return;
      tbody.innerHTML = predictions.map(p => `
        <tr class="text-sm">
          <td class="py-2 text-gray-400 font-mono text-xs">${{p.request_id.slice(0,8)}}...</td>
          <td class="py-2">${{p.input.amount?.toFixed(2) ?? '—'}}</td>
          <td class="py-2">${{p.anomaly_score.toFixed(4)}}</td>
          <td class="py-2">${{p.is_anomaly === 1
            ? '<span class="bg-red-900 text-red-300 px-2 py-0.5 rounded text-xs">ANOMALY</span>'
            : '<span class="bg-green-900 text-green-300 px-2 py-0.5 rounded text-xs">NORMAL</span>'}}</td>
          <td class="py-2 text-gray-400 text-xs">${{new Date(p.timestamp).toLocaleTimeString()}}</td>
        </tr>
      `).join('');
    }}
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: TransactionRequest) -> PredictionResponse:
    """Predict whether a transaction is anomalous.

    Args:
        req: Transaction feature inputs.

    Returns:
        Prediction result with request_id, timestamp, and anomaly score.
    """
    bundle = load_model()
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    x = build_feature_vector(req, feature_cols).reshape(1, -1)
    raw_pred = model.predict(x)[0]
    score = float(model.decision_function(x)[0])
    confidence = float(1 / (1 + np.exp(score)))
    is_anomaly = int(raw_pred == -1)

    result = PredictionResponse(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        is_anomaly=is_anomaly,
        anomaly_score=round(score, 6),
        confidence=round(confidence, 4),
        input=req.model_dump(),
    )
    _recent_predictions.appendleft(result.model_dump())
    return result


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint.

    Returns:
        Status, uptime, and model info.
    """
    bundle = load_model()
    return JSONResponse({
        "status": "ok",
        "uptime_seconds": (datetime.now(timezone.utc) - START_TIME).seconds,
        "model": "IsolationForest",
        "feature_count": len(bundle["feature_cols"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/metrics")
async def metrics() -> JSONResponse:
    """Return model performance metrics.

    Returns:
        Metrics JSON loaded from models/pipeline_model_metrics.json.
    """
    m = load_metrics()
    return JSONResponse({
        **m,
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/predictions")
async def recent_predictions() -> JSONResponse:
    """Return last 10 predictions.

    Returns:
        List of recent prediction results.
    """
    return JSONResponse({
        "predictions": list(_recent_predictions),
        "count": len(_recent_predictions),
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
