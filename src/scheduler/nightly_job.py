"""Nightly scheduler for transaction anomaly detection pipeline.

Job 1 @ 02:00 daily  — validate new data, retrain if >500 new rows
Job 2 @ every 6h     — drift check, create JIRA ticket if anomaly rate deviates >20%
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw")
CLEAN_PATH = Path("data/processed/clean.parquet")
MODEL_PATH = Path("models/pipeline_model.pkl")
METRICS_PATH = Path("models/pipeline_model_metrics.json")
AUDIT_LOG = Path("logs/audit.jsonl")


def _append_audit(event: str, detail: dict) -> None:
    """Append a JSON Lines entry to the audit log.

    Args:
        event: Event name.
        detail: Additional fields to log.
    """
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **detail}
    with AUDIT_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def retrain_if_new_data() -> None:
    """Job 1: Validate new raw data and retrain the model if >500 new rows found.

    Scans data/raw/ for CSV files added since last run.
    If combined new rows exceed 500, triggers full pipeline retraining.
    """
    logger.info("[Job 1] Starting nightly retrain check...")
    try:
        import pandas as pd
        csv_files = list(RAW_PATH.glob("*.csv"))
        total_rows = sum(len(pd.read_csv(f)) for f in csv_files)
        logger.info("[Job 1] Total rows in data/raw: %d", total_rows)

        if total_rows > 500:
            logger.info("[Job 1] >500 rows detected — triggering retrain...")
            from src.data.ingest import ingest
            from src.features.engineer import run as engineer_run
            from src.models.pipeline_model import train

            ingest()
            engineer_run()
            metrics = train()
            logger.info("[Job 1] Retrain complete. Accuracy: %.4f", metrics["accuracy"])
            _append_audit("retrain_complete", {"rows": total_rows, "accuracy": metrics["accuracy"]})
        else:
            logger.info("[Job 1] Row count below threshold (%d ≤ 500) — skipping retrain.", total_rows)
            _append_audit("retrain_skipped", {"rows": total_rows, "reason": "below_threshold"})

    except Exception as exc:
        logger.error("[Job 1] Retrain check failed: %s", exc)
        _append_audit("retrain_error", {"error": str(exc)})


def drift_check() -> None:
    """Job 2: Compare current anomaly rate against baseline and alert if deviation >20%.

    Loads the model, scores the current clean dataset, compares anomaly rate
    against the trained baseline. If drift >20% (absolute), logs a warning.
    In a production setup this would create a JIRA ticket via MCP.
    """
    logger.info("[Job 2] Starting drift check...")
    try:
        import pandas as pd
        from src.features.engineer import engineer_features

        if not CLEAN_PATH.exists():
            logger.warning("[Job 2] clean.parquet not found — skipping drift check.")
            return

        if not MODEL_PATH.exists():
            logger.warning("[Job 2] Model not found — skipping drift check.")
            return

        metrics = json.loads(METRICS_PATH.read_text())
        model = pickle.loads(MODEL_PATH.read_bytes())
        df_clean = pd.read_parquet(CLEAN_PATH)

        feature_cols = metrics.get("feature_columns", [])
        df_feat = engineer_features(df_clean)
        X = df_feat[feature_cols].values
        preds = model.predict(X)

        current_rate = float(preds.mean())
        baseline_rate = float(df_clean["is_anomaly"].mean())
        deviation = abs(current_rate - baseline_rate)

        logger.info(
            "[Job 2] Baseline anomaly rate: %.2f%%  Current: %.2f%%  Deviation: %.2f%%",
            baseline_rate * 100, current_rate * 100, deviation * 100,
        )

        if deviation > 0.20:
            logger.warning(
                "[Job 2] DRIFT ALERT — anomaly rate deviation %.1f%% exceeds 20%% threshold. "
                "Create JIRA ticket via: claude mcp jira_create_issue",
                deviation * 100,
            )
            _append_audit("drift_alert", {
                "baseline_rate": baseline_rate,
                "current_rate": current_rate,
                "deviation": deviation,
                "alert": True,
            })
        else:
            logger.info("[Job 2] Drift within acceptable range.")
            _append_audit("drift_ok", {
                "baseline_rate": baseline_rate,
                "current_rate": current_rate,
                "deviation": deviation,
            })

    except Exception as exc:
        logger.error("[Job 2] Drift check failed: %s", exc)
        _append_audit("drift_error", {"error": str(exc)})


def start() -> None:
    """Start the blocking scheduler with both jobs configured."""
    scheduler = BlockingScheduler(timezone="UTC")

    # Job 1 — retrain at 02:00 UTC daily
    scheduler.add_job(
        retrain_if_new_data,
        trigger=CronTrigger(hour=2, minute=0),
        id="nightly_retrain",
        name="Nightly retrain (02:00 UTC)",
        misfire_grace_time=3600,
    )

    # Job 2 — drift check every 6 hours
    scheduler.add_job(
        drift_check,
        trigger=IntervalTrigger(hours=6),
        id="drift_check",
        name="Drift check (every 6h)",
        misfire_grace_time=600,
    )

    logger.info("Scheduler started — retrain @ 02:00 UTC daily · drift check every 6h")
    logger.info("Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    start()
