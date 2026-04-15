"""FastAPI application for transaction anomaly detection pipeline."""

import json
import pickle
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/pipeline_model.pkl")
METRICS_PATH = Path("models/pipeline_model_metrics.json")
SCHEMA_PATH = Path("data/processed/feature_schema.json")

app = FastAPI(title="Transaction Anomaly Detection API", version="1.0.0")

# In-memory store for last 10 predictions
_recent_predictions: deque = deque(maxlen=10)
_model = None
_metrics: dict = {}
_feature_cols: list = []


def _load_model() -> None:
    """Load model and metadata into module-level globals."""
    global _model, _metrics, _feature_cols
    _model = pickle.loads(MODEL_PATH.read_bytes())
    _metrics = json.loads(METRICS_PATH.read_text())
    _feature_cols = json.loads(SCHEMA_PATH.read_text())["feature_columns"]


@app.on_event("startup")
async def startup_event() -> None:
    """Load model on application startup."""
    _load_model()


class PredictRequest(BaseModel):
    """Prediction request schema."""

    hour_of_day: float = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    day_of_week: float = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    is_weekend: int = Field(..., ge=0, le=1)
    is_business_hours: int = Field(..., ge=0, le=1)
    amount_log: float = Field(..., description="log1p(amount)")
    amount_zscore: float = Field(..., description="Z-score normalised amount")
    is_high_value: int = Field(..., ge=0, le=1)
    category_encoded: int = Field(..., ge=0)
    merchant_txn_count: int = Field(..., ge=1)


class PredictResponse(BaseModel):
    """Prediction response schema."""

    request_id: str
    timestamp: str
    prediction: int
    label: str
    confidence: float


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    """Run anomaly detection inference on a single transaction.

    Args:
        req: Feature values for the transaction.

    Returns:
        Prediction result with label and confidence score.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = [[getattr(req, col) for col in _feature_cols]]
    pred = int(_model.predict(features)[0])
    proba = float(_model.predict_proba(features)[0][pred])
    label = "ANOMALY" if pred == 1 else "NORMAL"
    request_id = str(uuid.uuid4())[:8]
    ts = datetime.now(timezone.utc).isoformat()

    result = PredictResponse(
        request_id=request_id,
        timestamp=ts,
        prediction=pred,
        label=label,
        confidence=round(proba, 4),
    )
    _recent_predictions.appendleft(result.model_dump())
    return result


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return service health status.

    Returns:
        Health status dict.
    """
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Return model metrics.

    Returns:
        Metrics dict loaded from models/pipeline_model_metrics.json.
    """
    return _metrics


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the HTML prediction dashboard.

    Returns:
        HTML response with the interactive dashboard.
    """
    recent_rows = ""
    for p in list(_recent_predictions):
        badge_class = "bg-red-100 text-red-800" if p["prediction"] == 1 else "bg-green-100 text-green-800"
        recent_rows += f"""
        <tr class="border-b">
          <td class="px-4 py-2 font-mono text-xs">{p['request_id']}</td>
          <td class="px-4 py-2 text-xs">{p['timestamp'][:19]}</td>
          <td class="px-4 py-2"><span class="px-2 py-1 rounded text-xs font-bold {badge_class}">{p['label']}</span></td>
          <td class="px-4 py-2 text-xs">{p['confidence']:.1%}</td>
        </tr>"""

    m = _metrics
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Transaction Anomaly Detection</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 min-h-screen">

  <!-- Header -->
  <header class="bg-blue-900 text-white px-8 py-4 flex items-center justify-between shadow">
    <div>
      <h1 class="text-2xl font-bold">Transaction Anomaly Detection</h1>
      <p class="text-blue-200 text-sm">Built with Claude Code · ML Pipeline v1.0</p>
    </div>
    <div class="flex items-center gap-2">
      <span id="status-dot" class="w-3 h-3 rounded-full bg-gray-400 inline-block"></span>
      <span id="status-text" class="text-sm">Checking...</span>
    </div>
  </header>

  <main class="max-w-5xl mx-auto px-6 py-8 space-y-8">

    <!-- Metrics Cards -->
    <section class="grid grid-cols-3 gap-4">
      <div class="bg-white rounded-xl shadow p-5 text-center">
        <p class="text-xs text-gray-500 uppercase font-semibold">Algorithm</p>
        <p class="text-lg font-bold text-blue-700 mt-1">{m.get('algorithm','—').replace('Classifier','')}</p>
      </div>
      <div class="bg-white rounded-xl shadow p-5 text-center">
        <p class="text-xs text-gray-500 uppercase font-semibold">Accuracy</p>
        <p class="text-2xl font-bold text-green-600 mt-1">{m.get('accuracy', 0):.1%}</p>
      </div>
      <div class="bg-white rounded-xl shadow p-5 text-center">
        <p class="text-xs text-gray-500 uppercase font-semibold">ROC-AUC</p>
        <p class="text-2xl font-bold text-purple-600 mt-1">{m.get('roc_auc', 0):.2f}</p>
      </div>
    </section>

    <!-- Prediction Form + Result -->
    <section class="grid grid-cols-2 gap-6">
      <div class="bg-white rounded-xl shadow p-6">
        <h2 class="text-lg font-semibold mb-4">Run Prediction</h2>
        <form id="predict-form" class="space-y-3">
          <div class="grid grid-cols-2 gap-3">
            <label class="block"><span class="text-xs text-gray-500">Hour of Day</span>
              <input name="hour_of_day" type="number" value="9" min="0" max="23" class="w-full border rounded px-2 py-1 text-sm mt-1"/>
            </label>
            <label class="block"><span class="text-xs text-gray-500">Day of Week</span>
              <input name="day_of_week" type="number" value="1" min="0" max="6" class="w-full border rounded px-2 py-1 text-sm mt-1"/>
            </label>
            <label class="block"><span class="text-xs text-gray-500">Is Weekend</span>
              <input name="is_weekend" type="number" value="0" min="0" max="1" class="w-full border rounded px-2 py-1 text-sm mt-1"/>
            </label>
            <label class="block"><span class="text-xs text-gray-500">Business Hours</span>
              <input name="is_business_hours" type="number" value="1" min="0" max="1" class="w-full border rounded px-2 py-1 text-sm mt-1"/>
            </label>
            <label class="block"><span class="text-xs text-gray-500">Amount Log</span>
              <input name="amount_log" type="number" value="5.01" step="0.01" class="w-full border rounded px-2 py-1 text-sm mt-1"/>
            </label>
            <label class="block"><span class="text-xs text-gray-500">Amount Z-score</span>
              <input name="amount_zscore" type="number" value="0.0" step="0.1" class="w-full border rounded px-2 py-1 text-sm mt-1"/>
            </label>
            <label class="block"><span class="text-xs text-gray-500">High Value</span>
              <input name="is_high_value" type="number" value="0" min="0" max="1" class="w-full border rounded px-2 py-1 text-sm mt-1"/>
            </label>
            <label class="block"><span class="text-xs text-gray-500">Category Encoded</span>
              <input name="category_encoded" type="number" value="2" min="0" class="w-full border rounded px-2 py-1 text-sm mt-1"/>
            </label>
            <label class="block col-span-2"><span class="text-xs text-gray-500">Merchant Txn Count</span>
              <input name="merchant_txn_count" type="number" value="3" min="1" class="w-full border rounded px-2 py-1 text-sm mt-1"/>
            </label>
          </div>
          <button type="submit" class="w-full bg-blue-700 hover:bg-blue-800 text-white font-semibold py-2 rounded-lg mt-2 transition">
            Predict
          </button>
        </form>
      </div>

      <div class="bg-white rounded-xl shadow p-6 flex flex-col items-center justify-center">
        <h2 class="text-lg font-semibold mb-4 self-start">Result</h2>
        <div id="result-panel" class="text-center hidden">
          <div id="result-badge" class="text-3xl font-bold px-8 py-4 rounded-2xl mb-3"></div>
          <p class="text-sm text-gray-500">Confidence: <span id="confidence-val" class="font-semibold text-gray-700"></span></p>
          <p class="text-xs text-gray-400 mt-1">Request ID: <span id="request-id"></span></p>
        </div>
        <div id="result-placeholder" class="text-gray-400 text-sm">Submit a transaction to see the result.</div>
      </div>
    </section>

    <!-- Recent Predictions Table -->
    <section class="bg-white rounded-xl shadow p-6">
      <h2 class="text-lg font-semibold mb-4">Last 10 Predictions</h2>
      <div class="overflow-x-auto">
        <table class="w-full text-left">
          <thead><tr class="text-xs text-gray-500 uppercase border-b">
            <th class="px-4 py-2">Request ID</th>
            <th class="px-4 py-2">Timestamp</th>
            <th class="px-4 py-2">Label</th>
            <th class="px-4 py-2">Confidence</th>
          </tr></thead>
          <tbody id="predictions-table">{recent_rows if recent_rows else '<tr><td colspan="4" class="px-4 py-4 text-center text-gray-400 text-sm">No predictions yet.</td></tr>'}</tbody>
        </table>
      </div>
    </section>

  </main>

  <script>
    // Health check polling
    async function checkHealth() {{
      try {{
        const r = await fetch('/health');
        const d = await r.json();
        const dot = document.getElementById('status-dot');
        const txt = document.getElementById('status-text');
        if (d.status === 'ok') {{
          dot.className = 'w-3 h-3 rounded-full bg-green-400 inline-block';
          txt.textContent = 'Live';
        }}
      }} catch(e) {{
        document.getElementById('status-dot').className = 'w-3 h-3 rounded-full bg-red-400 inline-block';
        document.getElementById('status-text').textContent = 'Offline';
      }}
    }}
    checkHealth();
    setInterval(checkHealth, 5000);

    // Prediction form
    document.getElementById('predict-form').addEventListener('submit', async (e) => {{
      e.preventDefault();
      const fd = new FormData(e.target);
      const body = {{}};
      fd.forEach((v, k) => body[k] = parseFloat(v));
      body.is_weekend = parseInt(body.is_weekend);
      body.is_business_hours = parseInt(body.is_business_hours);
      body.is_high_value = parseInt(body.is_high_value);
      body.category_encoded = parseInt(body.category_encoded);
      body.merchant_txn_count = parseInt(body.merchant_txn_count);

      const r = await fetch('/predict', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(body)}});
      const d = await r.json();
      document.getElementById('result-panel').classList.remove('hidden');
      document.getElementById('result-placeholder').classList.add('hidden');
      const badge = document.getElementById('result-badge');
      badge.textContent = d.label;
      badge.className = d.prediction === 1
        ? 'text-3xl font-bold px-8 py-4 rounded-2xl mb-3 bg-red-100 text-red-700'
        : 'text-3xl font-bold px-8 py-4 rounded-2xl mb-3 bg-green-100 text-green-700';
      document.getElementById('confidence-val').textContent = (d.confidence * 100).toFixed(1) + '%';
      document.getElementById('request-id').textContent = d.request_id;

      // Prepend to table
      const tbody = document.getElementById('predictions-table');
      const badgeCls = d.prediction === 1 ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800';
      const row = `<tr class="border-b">
        <td class="px-4 py-2 font-mono text-xs">${{d.request_id}}</td>
        <td class="px-4 py-2 text-xs">${{d.timestamp.slice(0,19)}}</td>
        <td class="px-4 py-2"><span class="px-2 py-1 rounded text-xs font-bold ${{badgeCls}}">${{d.label}}</span></td>
        <td class="px-4 py-2 text-xs">${{(d.confidence*100).toFixed(1)}}%</td>
      </tr>`;
      if (tbody.querySelector('td[colspan]')) tbody.innerHTML = '';
      tbody.insertAdjacentHTML('afterbegin', row);
    }});
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)
