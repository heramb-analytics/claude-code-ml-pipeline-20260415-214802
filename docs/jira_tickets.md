# JIRA Tickets — Transaction Anomaly Pipeline
# Project Key: TXAP (create at https://your-instance.atlassian.net)
# Generated: 2026-04-15

---

## Epic
**Summary:** Transaction Anomaly ML Pipeline — End-to-End Automation v1.0
**Type:** Epic
**Labels:** ml-pipeline, automated, claude-code

---

## TXAP-1 — Data Ingestion & Validation
**Type:** Task  
**Story Points:** 3  
**Status:** Done  
**Labels:** data-engineering, automated  

### What was done
- Loaded `transactions.csv` from `data/raw/`
- Applied 10 automated quality assertions
- Output: `data/processed/clean.parquet` (5 rows, 6 columns)

### Quality checks passed
- No nulls in key columns ✓
- Amount positive ✓
- Amount realistic range ✓
- is_anomaly binary ✓
- No duplicate transaction IDs ✓
- Required columns present ✓
- Schema matches expected ✓
- Timestamp parseable ✓
- Category values valid ✓
- Row count above minimum ✓

### Files created
- `src/data/ingest.py`
- `data/processed/clean.parquet`
- `logs/quality_report.json`

---

## TXAP-2 — Feature Engineering
**Type:** Task  
**Story Points:** 3  
**Status:** Done  
**Labels:** feature-engineering, automated  

### Features engineered (9 total)

| Feature | Type | Description |
|---|---|---|
| hour_of_day | int32 | Hour extracted from timestamp |
| day_of_week | int32 | Day of week (0=Mon) |
| is_weekend | int64 | 1 if Sat or Sun |
| is_business_hours | int64 | 1 if 9am–5pm |
| amount_log | float64 | log(amount + 1) |
| amount_zscore | float64 | Standardised amount |
| is_high_value | int64 | 1 if amount > 1000 |
| category_encoded | int8 | Label-encoded category |
| merchant_txn_count | int64 | Frequency of merchant |

### Files created
- `src/features/engineer.py`
- `data/processed/features.parquet`
- `data/processed/feature_schema.json`

---

## TXAP-3 — Model Training
**Type:** Task  
**Story Points:** 5  
**Status:** Done  
**Labels:** ml, model-training, automated  

### Model Card

| Attribute | Value |
|---|---|
| Algorithm | XGBoostClassifier |
| n_estimators | 150 |
| max_depth | 4 |
| learning_rate | 0.05 |
| Train samples | 5 |
| Test samples | 5 |
| Accuracy | 0.60 |
| F1 Score | 0.00 |
| Precision | 0.00 |
| Recall | 0.00 |
| ROC-AUC | 0.50 |

### Files created
- `src/models/pipeline_model.py`
- `models/pipeline_model.pkl`
- `models/pipeline_model_metrics.json`

---

## TXAP-4 — FastAPI + Dashboard UI
**Type:** Task  
**Story Points:** 5  
**Status:** Done  
**Labels:** api, deployment, ui, automated  

### Endpoints

| Method | Path | Description |
|---|---|---|
| POST | /predict | Run model inference |
| GET | /health | Service health check |
| GET | /metrics | Model metrics JSON |
| GET | / | HTML dashboard |

### Dashboard features
- Live status indicator (auto-refreshes every 5s)
- Prediction form with all 9 features
- Result badge (ANOMALY/NORMAL + confidence)
- Metrics cards + last-10 predictions table
- Tailwind CDN styling

### URL
http://localhost:8000

### Files created
- `src/api/main.py`

---

## TXAP-5 — Playwright E2E Tests + Screenshots
**Type:** Task  
**Story Points:** 3  
**Status:** Done  
**Labels:** testing, playwright, automated  

### Test results

| Test | Status |
|---|---|
| Dashboard loads | ✅ |
| Health endpoint OK | ✅ |
| Predict endpoint returns valid prediction | ✅ |
| Swagger docs accessible | ✅ |
| Metrics endpoint returns JSON | ✅ |
| Prediction result badge visible | ✅ |

### Screenshots saved
- `reports/screenshots/01_dashboard_home.png`
- `reports/screenshots/02_form_filled.png`
- `reports/screenshots/03_prediction_result.png`
- `reports/screenshots/04_swagger_docs.png`
- `reports/screenshots/05_metrics_endpoint.png`
- `reports/screenshots/06_health_endpoint.png`

### Files created
- `tests/e2e/test_api.py`

---

## TXAP-6 — Nightly Scheduler
**Type:** Task  
**Story Points:** 2  
**Status:** In Progress  
**Labels:** scheduler, monitoring, automated  

### Scheduled jobs

| Job | Schedule | Action |
|---|---|---|
| Retrain | 02:00 daily | Validate new data → retrain if >500 rows |
| Drift check | Every 6h | Compare anomaly rate → create JIRA ticket if >20% deviation |

### Files created
- `src/scheduler/nightly_job.py`
