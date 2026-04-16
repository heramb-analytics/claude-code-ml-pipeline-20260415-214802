"""EDA report — generates 5 charts into reports/figures/."""

from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FEATURES_PATH = Path("data/processed/features.parquet")
FIGURES_DIR = Path("reports/figures")


def run() -> None:
    """Generate 5 EDA charts from feature data."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEATURES_PATH)

    # Chart 1: Amount distribution by anomaly label
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, grp in df.groupby("is_anomaly"):
        ax.hist(grp["amount"], bins=20, alpha=0.7, label=f"is_anomaly={int(label)}")
    ax.set_title("Amount Distribution by Anomaly Label")
    ax.set_xlabel("Amount ($)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_amount_distribution.png", dpi=120)
    plt.close(fig)
    print("   📊 Chart 1/5: Amount Distribution — saved")

    # Chart 2: Transaction count by category
    fig, ax = plt.subplots(figsize=(8, 5))
    cat_counts = df["category"].value_counts() if "category" in df.columns else pd.Series(dtype=int)
    if len(cat_counts) == 0:
        # reconstruct from dummies
        cat_cols = [c for c in df.columns if c.startswith("cat_")]
        cat_counts = pd.Series({c.replace("cat_", ""): df[c].sum() for c in cat_cols})
    cat_counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Transaction Count by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_category_counts.png", dpi=120)
    plt.close(fig)
    print("   📊 Chart 2/5: Category Counts — saved")

    # Chart 3: Hourly transaction volume
    fig, ax = plt.subplots(figsize=(8, 5))
    hourly = df.groupby("hour_of_day").size()
    ax.bar(hourly.index, hourly.values, color="coral")
    ax.set_title("Transaction Volume by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Transaction Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_hourly_volume.png", dpi=120)
    plt.close(fig)
    print("   📊 Chart 3/5: Hourly Volume — saved")

    # Chart 4: Anomaly rate by category
    fig, ax = plt.subplots(figsize=(8, 5))
    cat_cols = [c for c in df.columns if c.startswith("cat_")]
    anomaly_by_cat = {}
    for c in cat_cols:
        subset = df[df[c] == 1]
        if len(subset) > 0:
            anomaly_by_cat[c.replace("cat_", "")] = subset["is_anomaly"].mean()
    if anomaly_by_cat:
        pd.Series(anomaly_by_cat).plot(kind="bar", ax=ax, color="tomato")
    ax.set_title("Anomaly Rate by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Anomaly Rate")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_anomaly_rate_by_category.png", dpi=120)
    plt.close(fig)
    print("   📊 Chart 4/5: Anomaly Rate by Category — saved")

    # Chart 5: Log-amount vs Z-score scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = df["is_anomaly"].map({0: "steelblue", 1: "red"})
    ax.scatter(df["log_amount"], df["amount_zscore"], c=colors, alpha=0.8, edgecolors="k", linewidths=0.5)
    ax.set_title("Log Amount vs Amount Z-Score (red=anomaly)")
    ax.set_xlabel("Log Amount")
    ax.set_ylabel("Amount Z-Score")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_log_amount_vs_zscore.png", dpi=120)
    plt.close(fig)
    print("   📊 Chart 5/5: Log Amount vs Z-Score — saved")

    print("   ✅ Subagent B done — 5 charts saved to reports/figures/")


if __name__ == "__main__":
    run()
