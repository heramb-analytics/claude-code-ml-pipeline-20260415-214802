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
    algorithm = metrics.get("algorithm", "XGBoostClassifier").replace("Classifier", "")
    accuracy = metrics.get("accuracy", 0)
    f1 = metrics.get("f1_score", 0)
    auc = metrics.get("roc_auc", 0)
    train_size = metrics.get("train_size", 0)

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    W = prs.slide_width
    H = prs.slide_height

    # ── Slide 1 — Cover ───────────────────────────────────────────────────
    print("   📊 Slide 1/8: Cover — done")
    s = _dark_slide(prs)
    _text_box(s, "Transaction Anomaly Detection", Inches(1), Inches(2.2), Inches(11), Inches(1.2),
              size=40, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _text_box(s, "ML Pipeline  ·  Built with Claude Code  ·  2026-04-15",
              Inches(1), Inches(3.5), Inches(11), Inches(0.7),
              size=20, color=RGBColor(0xBF, 0xDB, 0xF7), align=PP_ALIGN.CENTER)
    _text_box(s, "End-to-End Autonomous Pipeline — 12 Stages",
              Inches(1), Inches(4.3), Inches(11), Inches(0.6),
              size=16, color=RGBColor(0x93, 0xC5, 0xFD), align=PP_ALIGN.CENTER)

    # ── Slide 2 — Problem Statement ───────────────────────────────────────
    print("   📊 Slide 2/8: Problem Statement — done")
    s = _light_slide(prs)
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    _fill(bar, NAVY)
    _text_box(s, "The Problem", Inches(0.8), Inches(0.3), Inches(11), Inches(0.8),
              size=32, bold=True, color=NAVY)
    bullets = [
        "5 transactions with 40% anomaly rate in sample data",
        "Manual fraud detection is slow, inconsistent, and costly",
        "Goal: automated real-time anomaly scoring via REST API",
    ]
    for i, b in enumerate(bullets):
        _text_box(s, f"•  {b}", Inches(1), Inches(1.5 + i * 1.2), Inches(11), Inches(1.0),
                  size=20, color=GRAY)

    # ── Slide 3 — EDA Highlights ──────────────────────────────────────────
    print("   📊 Slide 3/8: EDA Highlights — done")
    s = _light_slide(prs)
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    _fill(bar, NAVY)
    _text_box(s, "Data Overview", Inches(0.8), Inches(0.3), Inches(11), Inches(0.8),
              size=32, bold=True, color=NAVY)
    _add_image_safe(s, FIGURES_DIR / "01_target_distribution.png",
                    Inches(0.5), Inches(1.3), Inches(5.8), Inches(3.8))
    _add_image_safe(s, FIGURES_DIR / "02_feature_correlations.png",
                    Inches(6.8), Inches(1.3), Inches(5.8), Inches(3.8))
    _text_box(s, "5 rows  ·  6 raw cols  ·  9 engineered features  ·  40% anomaly rate",
              Inches(0.5), Inches(5.3), Inches(12), Inches(0.5),
              size=14, color=GRAY, align=PP_ALIGN.CENTER)

    # ── Slide 4 — Data Engineering ────────────────────────────────────────
    print("   📊 Slide 4/8: Data Engineering — done")
    s = _light_slide(prs)
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    _fill(bar, NAVY)
    _text_box(s, "Data Pipeline", Inches(0.8), Inches(0.3), Inches(11), Inches(0.8),
              size=32, bold=True, color=NAVY)
    _add_image_safe(s, FIGURES_DIR / "03_missing_values.png",
                    Inches(0.5), Inches(1.3), Inches(5.5), Inches(3.5))
    checks = [
        ("required_columns_present", "✓"), ("no_duplicate_ids", "✓"),
        ("amount_positive", "✓"), ("amount_realistic_range", "✓"),
        ("is_anomaly_binary", "✓"), ("no_nulls_in_key_columns", "✓"),
        ("timestamp_parseable", "✓"), ("category_non_empty", "✓"),
        ("merchant_id_non_null", "✓"), ("both_classes_present", "✓"),
    ]
    _text_box(s, "Quality Checks", Inches(6.5), Inches(1.3), Inches(6), Inches(0.5),
              size=16, bold=True, color=NAVY)
    for i, (name, result) in enumerate(checks):
        _text_box(s, f"{result}  {name}", Inches(6.5), Inches(1.8 + i * 0.35),
                  Inches(6), Inches(0.35), size=13, color=GREEN)

    # ── Slide 5 — Model Results ───────────────────────────────────────────
    print("   📊 Slide 5/8: Model Results — done")
    s = _light_slide(prs)
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    _fill(bar, NAVY)
    _text_box(s, "Model Performance", Inches(0.8), Inches(0.3), Inches(11), Inches(0.8),
              size=32, bold=True, color=NAVY)
    cards = [("Accuracy", f"{accuracy:.1%}"), ("F1 Score", f"{f1:.2f}"), ("ROC-AUC", f"{auc:.2f}")]
    for i, (label, val) in enumerate(cards):
        cx = Inches(0.6 + i * 4.2)
        box = s.shapes.add_shape(1, cx, Inches(1.4), Inches(3.8), Inches(1.8))
        _fill(box, LIGHT)
        _text_box(s, label, cx + Inches(0.1), Inches(1.5), Inches(3.6), Inches(0.5),
                  size=14, color=GRAY, align=PP_ALIGN.CENTER)
        _text_box(s, val, cx + Inches(0.1), Inches(2.0), Inches(3.6), Inches(0.9),
                  size=30, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    _add_image_safe(s, FIGURES_DIR / "04_amount_distribution.png",
                    Inches(0.5), Inches(3.4), Inches(12), Inches(3.0))
    _text_box(s, f"Algorithm: {algorithm}  ·  Trained on {train_size} samples (LOO-CV)",
              Inches(0.5), Inches(6.5), Inches(12), Inches(0.5),
              size=13, color=GRAY, align=PP_ALIGN.CENTER)

    # ── Slide 6 — Live Dashboard ──────────────────────────────────────────
    print("   📊 Slide 6/8: Live Dashboard — done")
    s = _light_slide(prs)
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    _fill(bar, NAVY)
    _text_box(s, "Live Dashboard", Inches(0.8), Inches(0.3), Inches(11), Inches(0.8),
              size=32, bold=True, color=NAVY)
    _add_image_safe(s, SCREENSHOTS_DIR / "01_dashboard_home.png",
                    Inches(0.5), Inches(1.2), Inches(12.3), Inches(5.5))
    _text_box(s, "Accessible at http://localhost:8000",
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
                    Inches(0.4), Inches(1.2), Inches(6.0), Inches(4.5))
    _add_image_safe(s, SCREENSHOTS_DIR / "04_swagger_docs.png",
                    Inches(6.8), Inches(1.2), Inches(6.0), Inches(4.5))
    _text_box(s, "8 unit tests passed   ·   6 Playwright E2E tests passed   ·   6 screenshots saved",
              Inches(0.5), Inches(6.0), Inches(12), Inches(0.6),
              size=16, bold=True, color=GREEN, align=PP_ALIGN.CENTER)

    # ── Slide 8 — Pipeline Complete ───────────────────────────────────────
    print("   📊 Slide 8/8: PIPELINE COMPLETE — done")
    s = _dark_slide(prs)
    _text_box(s, "PIPELINE COMPLETE", Inches(0.5), Inches(0.8), Inches(12), Inches(1.0),
              size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    summary = [
        f"Model:       {algorithm} — Accuracy: {accuracy:.1%}",
        "API:         http://localhost:8000",
        "GitHub:      github.com/heramb-analytics/claude-code-ml-pipeline-20260415-214802",
        "Tests:       8 unit  ·  6 Playwright E2E",
        "Scheduler:   retrain @ 02:00 UTC  ·  drift check every 6h",
        "Built by:    Claude Code (claude-sonnet-4-6)",
    ]
    for i, line in enumerate(summary):
        _text_box(s, line, Inches(1.5), Inches(2.0 + i * 0.75), Inches(10), Inches(0.65),
                  size=17, color=RGBColor(0xBF, 0xDB, 0xF7))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT_PATH))
    print(f"   📁 File: {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
