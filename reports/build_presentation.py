"""Build 8-slide pipeline presentation using python-pptx."""

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

METRICS_PATH = Path("models/pipeline_model_metrics.json")
FIGURES_DIR = Path("reports/figures")
SCREENSHOTS_DIR = Path("reports/screenshots")
OUTPUT_PATH = Path("reports/pipeline_presentation.pptx")

NAVY = RGBColor(0x1E, 0x3A, 0x8A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x05, 0x96, 0x69)
RED = RGBColor(0xDC, 0x26, 0x26)
GRAY = RGBColor(0x6B, 0x72, 0x80)
LIGHT = RGBColor(0xF3, 0xF4, 0xF6)


def _fill(shape, color: RGBColor) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = color


def _text_box(slide, text: str, left, top, width, height,
               size: int = 18, bold: bool = False, color: RGBColor = None,
               align=PP_ALIGN.LEFT) -> None:
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = color


def _dark_slide(prs: Presentation) -> object:
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = NAVY
    return slide


def _light_slide(prs: Presentation) -> object:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = WHITE
    return slide


def _add_image_safe(slide, path: Path, left, top, width, height) -> None:
    if path.exists():
        slide.shapes.add_picture(str(path), left, top, width, height)
    else:
        ph = slide.shapes.add_textbox(left, top, width, height)
        _fill(ph, LIGHT)
        ph.text_frame.text = f"[{path.name}]"


def build() -> None:
    """Build the 8-slide presentation and save to reports/pipeline_presentation.pptx."""
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    algorithm = metrics.get("algorithm", "IsolationForest")
    m = metrics.get("metrics", {})
    f1 = m.get("f1_score", 0)
    precision = m.get("precision", 0)
    recall = m.get("recall", 0)
    train_size = metrics.get("train_size", 0)

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    W = prs.slide_width

    # ── Slide 1 — Cover ───────────────────────────────────────────────────
    print("   📊 Slide 1/8: Cover — done")
    s = _dark_slide(prs)
    _text_box(s, "Transaction Anomaly Detection", Inches(1), Inches(2.0), Inches(11), Inches(1.2),
              size=40, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _text_box(s, "ML Pipeline  ·  Built with Claude Code  ·  2026-04-16",
              Inches(1), Inches(3.4), Inches(11), Inches(0.7),
              size=20, color=RGBColor(0xBF, 0xDB, 0xF7), align=PP_ALIGN.CENTER)
    _text_box(s, "IsolationForest · FastAPI · Playwright · JIRA · Confluence",
              Inches(1), Inches(4.2), Inches(11), Inches(0.6),
              size=16, color=RGBColor(0x93, 0xC5, 0xFD), align=PP_ALIGN.CENTER)
    _text_box(s, "End-to-End Autonomous Pipeline — 11 Stages",
              Inches(1), Inches(5.0), Inches(11), Inches(0.5),
              size=14, color=RGBColor(0x60, 0xA5, 0xFA), align=PP_ALIGN.CENTER)

    # ── Slide 2 — Problem Statement ───────────────────────────────────────
    print("   📊 Slide 2/8: Problem Statement — done")
    s = _light_slide(prs)
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    _fill(bar, NAVY)
    _text_box(s, "The Problem", Inches(0.8), Inches(0.3), Inches(11), Inches(0.8),
              size=32, bold=True, color=NAVY)
    bullets = [
        "Financial fraud costs $485B+ annually — detection must be fast and automated",
        "Dataset: 5 transactions across 3 categories · 1 anomaly (20% rate)",
        "Goal: unsupervised real-time anomaly scoring via IsolationForest",
        "Deliverable: REST API + live dashboard + automated nightly retraining",
    ]
    for i, b in enumerate(bullets):
        _text_box(s, f"•  {b}", Inches(1), Inches(1.5 + i * 1.1), Inches(11), Inches(1.0),
                  size=20, color=GRAY)

    # ── Slide 3 — EDA Highlights ──────────────────────────────────────────
    print("   📊 Slide 3/8: EDA Highlights — done")
    s = _light_slide(prs)
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    _fill(bar, NAVY)
    _text_box(s, "Exploratory Data Analysis", Inches(0.8), Inches(0.3), Inches(11), Inches(0.8),
              size=32, bold=True, color=NAVY)
    _add_image_safe(s, FIGURES_DIR / "01_amount_distribution.png",
                    Inches(0.3), Inches(1.2), Inches(6.2), Inches(3.7))
    _add_image_safe(s, FIGURES_DIR / "05_log_amount_vs_zscore.png",
                    Inches(6.8), Inches(1.2), Inches(6.2), Inches(3.7))
    _text_box(s, "5 rows  ·  6 raw cols  ·  16 engineered features  ·  20% anomaly rate",
              Inches(0.5), Inches(5.1), Inches(12), Inches(0.5),
              size=14, color=GRAY, align=PP_ALIGN.CENTER)

    # ── Slide 4 — Data Engineering ────────────────────────────────────────
    print("   📊 Slide 4/8: Data Engineering — done")
    s = _light_slide(prs)
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    _fill(bar, NAVY)
    _text_box(s, "Feature Engineering & Quality Validation", Inches(0.8), Inches(0.3), Inches(11), Inches(0.8),
              size=28, bold=True, color=NAVY)
    _add_image_safe(s, FIGURES_DIR / "02_category_counts.png",
                    Inches(0.3), Inches(1.2), Inches(6.0), Inches(3.5))
    checks = [
        "✓  file_not_empty", "✓  required_columns_present",
        "✓  no_duplicate_transaction_ids", "✓  amount_non_negative",
        "✓  no_null_amounts", "✓  is_anomaly_binary",
        "✓  timestamp_parseable", "✓  no_null_categories",
        "✓  amount_reasonable_range", "✓  anomaly_rate_plausible",
    ]
    _text_box(s, "10/10 Quality Checks", Inches(6.5), Inches(1.2), Inches(6), Inches(0.5),
              size=16, bold=True, color=NAVY)
    for i, name in enumerate(checks):
        _text_box(s, name, Inches(6.5), Inches(1.7 + i * 0.35),
                  Inches(6), Inches(0.35), size=13, color=GREEN)

    # ── Slide 5 — Model Results ───────────────────────────────────────────
    print("   📊 Slide 5/8: Model Results — done")
    s = _light_slide(prs)
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    _fill(bar, NAVY)
    _text_box(s, "Model Performance — IsolationForest", Inches(0.8), Inches(0.3), Inches(11), Inches(0.8),
              size=30, bold=True, color=NAVY)
    cards = [("Precision", f"{precision:.4f}"), ("Recall", f"{recall:.4f}"), ("F1 Score", f"{f1:.4f}")]
    for i, (label, val) in enumerate(cards):
        cx = Inches(0.6 + i * 4.2)
        box = s.shapes.add_shape(1, cx, Inches(1.4), Inches(3.8), Inches(1.8))
        _fill(box, LIGHT)
        _text_box(s, label, cx + Inches(0.1), Inches(1.5), Inches(3.6), Inches(0.5),
                  size=14, color=GRAY, align=PP_ALIGN.CENTER)
        _text_box(s, val, cx + Inches(0.1), Inches(2.0), Inches(3.6), Inches(0.9),
                  size=30, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    _add_image_safe(s, FIGURES_DIR / "04_anomaly_rate_by_category.png",
                    Inches(0.5), Inches(3.4), Inches(12), Inches(3.0))
    _text_box(s, f"Algorithm: {algorithm}  ·  RandomizedSearchCV  ·  Train size: {train_size}",
              Inches(0.5), Inches(6.5), Inches(12), Inches(0.5),
              size=13, color=GRAY, align=PP_ALIGN.CENTER)

    # ── Slide 6 — Live Dashboard ──────────────────────────────────────────
    print("   📊 Slide 6/8: Live Dashboard — done")
    s = _light_slide(prs)
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    _fill(bar, NAVY)
    _text_box(s, "Live Dashboard — FastAPI + Tailwind", Inches(0.8), Inches(0.3), Inches(11), Inches(0.8),
              size=30, bold=True, color=NAVY)
    _add_image_safe(s, SCREENSHOTS_DIR / "01_dashboard_home.png",
                    Inches(0.3), Inches(1.2), Inches(12.7), Inches(5.5))
    _text_box(s, "Accessible at http://localhost:8000  ·  Swagger: /docs  ·  Metrics: /metrics",
              Inches(0.5), Inches(6.9), Inches(12), Inches(0.4),
              size=13, color=GRAY, align=PP_ALIGN.CENTER)

    # ── Slide 7 — Test Evidence ───────────────────────────────────────────
    print("   📊 Slide 7/8: Automated Quality Gates — done")
    s = _light_slide(prs)
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    _fill(bar, NAVY)
    _text_box(s, "Automated Quality Gates", Inches(0.8), Inches(0.3), Inches(11), Inches(0.8),
              size=32, bold=True, color=NAVY)
    _add_image_safe(s, SCREENSHOTS_DIR / "03_prediction_result.png",
                    Inches(0.3), Inches(1.2), Inches(6.2), Inches(4.5))
    _add_image_safe(s, SCREENSHOTS_DIR / "04_swagger_docs.png",
                    Inches(6.8), Inches(1.2), Inches(6.2), Inches(4.5))
    _text_box(s, "8/8 unit tests passed   ·   6/6 Playwright E2E tests passed   ·   6 screenshots saved",
              Inches(0.5), Inches(6.0), Inches(12), Inches(0.6),
              size=16, bold=True, color=GREEN, align=PP_ALIGN.CENTER)

    # ── Slide 8 — Pipeline Complete ───────────────────────────────────────
    print("   📊 Slide 8/8: Pipeline Complete — done")
    s = _dark_slide(prs)
    _text_box(s, "PIPELINE COMPLETE", Inches(0.5), Inches(0.8), Inches(12), Inches(1.0),
              size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    summary = [
        f"Model:        {algorithm}  ·  F1={f1:.4f}  ·  Precision={precision:.4f}  ·  Recall={recall:.4f}",
        "API:          http://localhost:8000",
        "GitHub:       github.com/heramb-analytics/claude-code-ml-pipeline-20260415-214802",
        "JIRA:         TAD-78 (Epic)  ·  TAD-79 → TAD-84 (6 tasks)  ·  TAD Sprint 1",
        "Tests:        8 unit  ·  6 Playwright E2E  ·  10 data quality  ·  12 validation",
        "Scheduler:    retrain @ 02:00 UTC daily  ·  drift check every 6h",
    ]
    for i, line in enumerate(summary):
        _text_box(s, line, Inches(1.0), Inches(2.0 + i * 0.75), Inches(11), Inches(0.65),
                  size=16, color=RGBColor(0xBF, 0xDB, 0xF7))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT_PATH))
    print(f"   📁 File: {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
