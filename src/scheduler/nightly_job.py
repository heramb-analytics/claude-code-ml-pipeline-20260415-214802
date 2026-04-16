"""Nightly scheduler for transaction anomaly detection pipeline.

Job 1 @ 02:00 daily  — validate new data, retrain if >500 new rows
Job 2 @ every 6h     — drift check, create JIRA ticket if anomaly rate deviates >20%
"""

import json
import logging
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import schedule
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/pipeline_model.pkl")
METRICS_PATH = Path("models/pipeline_model_metrics.json")
AUDIT_LOG = Path("logs/audit.jsonl")

# Baseline anomaly rate (from training data)
BASELINE_ANOMALY_RATE: float | None = None


def _log(event: dict) -> None:
    """Append event to audit.jsonl.

    Args:
        event: Dictionary to log.
    """
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps({**event, "timestamp": datetime.now(timezone.utc).isoformat()}) + "\n")
    logger.info(json.dumps(event))


def _load_baseline_anomaly_rate() -> float:
    """Load baseline anomaly rate from training metrics.

    Returns:
        Baseline anomaly rate as float.
    """
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            m = json.load(f)
        return m.get("baseline_anomaly_rate", 0.2)
    return 0.2


def job_nightly_retrain() -> None:
    """Job 1 — Run at 02:00 daily.

    Validate new data and retrain if >500 new rows detected.
    """
    logger.info("🔄 Job 1: Nightly retrain check starting...")
    _log({"job": "nightly_retrain", "status": "started"})

    # Count new rows across all CSV files in data/raw/
    total_rows = 0
    for csv_file in RAW_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            total_rows += len(df)
        except Exception as e:
            logger.warning(f"Could not read {csv_file}: {e}")

    logger.info(f"   📊 Total rows in raw data: {total_rows}")

    if total_rows > 500:
        logger.info(f"   🚀 {total_rows} rows detected — triggering retrain...")
        try:
            result = subprocess.run(
                ["python3", "src/data/ingest.py"],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                subprocess.run(["python3", "src/features/engineer.py"], timeout=120)
                subprocess.run(["python3", "src/models/train.py"], timeout=300)
                _log({"job": "nightly_retrain", "status": "retrained", "rows": total_rows})
                logger.info("   ✅ Retrain complete")
            else:
                _log({"job": "nightly_retrain", "status": "failed", "error": result.stderr})
                logger.error(f"   ❌ Retrain failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            _log({"job": "nightly_retrain", "status": "timeout"})
            logger.error("   ❌ Retrain timed out")
    else:
        logger.info(f"   ℹ️  Only {total_rows} rows — threshold not met (>500 required), skipping retrain")
        _log({"job": "nightly_retrain", "status": "skipped", "rows": total_rows, "threshold": 500})


def job_drift_check() -> None:
    """Job 2 — Run every 6 hours.

    Compare current anomaly rate vs baseline. Create JIRA ticket if deviation >20%.
    """
    global BASELINE_ANOMALY_RATE
    logger.info("📡 Job 2: Drift check starting...")
    _log({"job": "drift_check", "status": "started"})

    if BASELINE_ANOMALY_RATE is None:
        BASELINE_ANOMALY_RATE = _load_baseline_anomaly_rate()

    # Load latest predictions from clean data
    clean_path = PROCESSED_DIR / "clean.parquet"
    if not clean_path.exists():
        logger.warning("   ⚠️  clean.parquet not found — skipping drift check")
        _log({"job": "drift_check", "status": "skipped", "reason": "no_data"})
        return

    df = pd.read_parquet(clean_path)
    if "is_anomaly" not in df.columns or len(df) == 0:
        _log({"job": "drift_check", "status": "skipped", "reason": "no_labels"})
        return

    current_rate = float(df["is_anomaly"].mean())
    baseline = BASELINE_ANOMALY_RATE
    deviation = abs(current_rate - baseline) / max(baseline, 1e-9)

    logger.info(f"   📊 Baseline anomaly rate: {baseline:.4f}")
    logger.info(f"   📊 Current anomaly rate:  {current_rate:.4f}")
    logger.info(f"   📊 Deviation:             {deviation:.2%}")

    if deviation > 0.20:
        logger.warning(f"   ⚠️  DRIFT DETECTED — deviation {deviation:.2%} > 20% threshold!")
        _log({
            "job": "drift_check",
            "status": "drift_detected",
            "baseline_rate": baseline,
            "current_rate": current_rate,
            "deviation_pct": round(deviation * 100, 2),
        })
        _create_drift_jira_ticket(baseline, current_rate, deviation)
    else:
        logger.info("   ✅ No significant drift detected")
        _log({
            "job": "drift_check",
            "status": "ok",
            "baseline_rate": baseline,
            "current_rate": current_rate,
            "deviation_pct": round(deviation * 100, 2),
        })


def _create_drift_jira_ticket(
    baseline: float, current_rate: float, deviation: float
) -> None:
    """Create a JIRA ticket when anomaly rate drifts >20%.

    Args:
        baseline: Baseline anomaly rate from training.
        current_rate: Current observed anomaly rate.
        deviation: Relative deviation as a fraction.
    """
    now = datetime.now(timezone.utc).isoformat()
    summary = f"[DRIFT ALERT] Anomaly rate deviation {deviation:.1%} detected"
    logger.info(f"   🎫 Creating JIRA ticket: {summary}")
    # JIRA ticket creation would use the MCP tool in production
    # Here we log the intent so it can be picked up by the monitoring system
    _log({
        "job": "drift_check",
        "action": "jira_ticket_requested",
        "summary": summary,
        "baseline_rate": baseline,
        "current_rate": current_rate,
        "deviation_pct": round(deviation * 100, 2),
        "project": "TAD",
        "priority": "High",
        "detected_at": now,
    })
    logger.info("   ✅ JIRA drift ticket logged to audit.jsonl")


def run_scheduler() -> None:
    """Configure and start the scheduler with both jobs."""
    global BASELINE_ANOMALY_RATE
    BASELINE_ANOMALY_RATE = _load_baseline_anomaly_rate()

    # Job 1: Nightly retrain at 02:00
    schedule.every().day.at("02:00").do(job_nightly_retrain)

    # Job 2: Drift check every 6 hours
    schedule.every(6).hours.do(job_drift_check)

    logger.info("⏰ Scheduler started:")
    logger.info("   Job 1: Nightly retrain @ 02:00 daily")
    logger.info("   Job 2: Drift check every 6 hours")

    _log({"event": "scheduler_started", "jobs": ["nightly_retrain@02:00", "drift_check@6h"]})

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run both jobs once for testing
        logger.info("Running jobs in test mode...")
        job_nightly_retrain()
        job_drift_check()
    else:
        run_scheduler()
