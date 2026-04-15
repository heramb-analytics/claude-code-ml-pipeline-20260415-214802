"""EDA report — generates 5 charts into reports/figures/."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CLEAN_PATH = Path("data/processed/clean.parquet")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run() -> None:
    """Generate 5 EDA charts and save to reports/figures/."""
    df = pd.read_parquet(CLEAN_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    sns.set_theme(style="whitegrid")

    # Chart 1 — target distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["is_anomaly"].value_counts()
    ax.bar(["Normal (0)", "Anomaly (1)"], [counts.get(0, 0), counts.get(1, 0)],
           color=["#2563EB", "#DC2626"])
    ax.set_title("Target Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_target_distribution.png", dpi=120)
    plt.close(fig)

    # Chart 2 — feature correlations (numeric only)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlations")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_feature_correlations.png", dpi=120)
    plt.close(fig)

    # Chart 3 — missing values
    fig, ax = plt.subplots(figsize=(6, 4))
    null_pct = df.isnull().mean() * 100
    null_pct.plot(kind="bar", ax=ax, color="#7C3AED")
    ax.set_title("Missing Values (%)")
    ax.set_ylabel("% Null")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_missing_values.png", dpi=120)
    plt.close(fig)

    # Chart 4 — amount distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    df["amount"].hist(bins=10, ax=ax, color="#059669", edgecolor="white")
    ax.set_title("Amount Distribution")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_amount_distribution.png", dpi=120)
    plt.close(fig)

    # Chart 5 — amount by anomaly class
    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column="amount", by="is_anomaly", ax=ax,
               boxprops=dict(color="#1E3A8A"))
    ax.set_title("Amount by Anomaly Class")
    ax.set_xlabel("is_anomaly")
    ax.set_ylabel("Amount")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_amount_by_class.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
    print("   ✅ Subagent B done — 5 charts saved to reports/figures/")
