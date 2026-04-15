# Transaction Anomaly Detection Pipeline

End-to-end ML pipeline for transaction anomaly detection, built autonomously with Claude Code.

## What Was Built

| Stage | Description | Output |
|-------|-------------|--------|
| 0 | Data Discovery | 5 rows · 6 cols · binary classification |
| 1 | Ingest & Validate | `data/processed/clean.parquet` · 10/10 checks |
| 2 | Features + EDA | 9 features · 5 charts · 12/12 validation checks |
| 3 | Model Training | XGBoost · Accuracy: 60% · ROC-AUC: 0.50 |
| 4 | Unit Tests | 8/8 passed |
| 5 | FastAPI + Dashboard | http://localhost:8000 |
| 6 | Playwright E2E | 6/6 tests · 6 screenshots |
| 7 | Git + GitHub | This repo |

## Quick Start

```bash
# Install dependencies
pip3 install pandas numpy scikit-learn xgboost fastapi uvicorn pytest playwright pytest-playwright pyarrow matplotlib seaborn

# Install Playwright browser
python3 -m playwright install chromium

# Run the API
uvicorn src.api.main:app --port 8000

# Open dashboard
open http://localhost:8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Interactive dashboard |
| `POST` | `/predict` | Run anomaly inference |
| `GET` | `/health` | Service health check |
| `GET` | `/metrics` | Model metrics |
| `GET` | `/docs` | Swagger UI |

## Run Tests

```bash
# Unit tests
pytest tests/unit/ -v

# E2E tests (requires running API on port 8000)
pytest tests/e2e/ -v
```

## Project Structure

```
src/
  api/main.py          FastAPI app + dashboard
  data/ingest.py       Data ingestion + 10 quality checks
  features/engineer.py Feature engineering (9 features)
  models/pipeline_model.py XGBoost training
  validation/checks.py 12 post-ingestion checks
  scheduler/nightly_job.py Drift + retrain jobs
tests/
  unit/                8 unit tests
  e2e/                 6 Playwright E2E tests
models/
  pipeline_model.pkl   Trained model
  pipeline_model_metrics.json Performance metrics
reports/
  figures/             5 EDA charts
  screenshots/         6 Playwright screenshots
```

---
*Built with [Claude Code](https://claude.ai/code)*
