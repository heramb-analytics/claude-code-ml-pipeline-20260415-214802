"""Playwright E2E tests for transaction anomaly detection API."""

import re
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:8000"
SCREENSHOTS_DIR = Path("reports/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save(page: Page, filename: str) -> None:
    """Save a screenshot and print confirmation.

    Args:
        page: Playwright page object.
        filename: Target filename (saved under reports/screenshots/).
    """
    path = SCREENSHOTS_DIR / filename
    page.screenshot(path=str(path))
    size = path.stat().st_size
    print(f"   ✅ Saved: reports/screenshots/{filename} ({size:,} bytes)")


def test_01_dashboard_home(page: Page) -> None:
    """Dashboard loads and displays the header title."""
    print("   📸 Taking screenshot 1/6: 01_dashboard_home.png...")
    page.goto(BASE_URL)
    expect(page.locator("h1")).to_contain_text("Transaction Anomaly Detection")
    _save(page, "01_dashboard_home.png")


def test_02_form_filled(page: Page) -> None:
    """Prediction form can be filled with high-value anomaly values."""
    print("   📸 Taking screenshot 2/6: 02_form_filled.png...")
    page.goto(BASE_URL)
    page.fill("input[name='amount_log']", "9.2")
    page.fill("input[name='amount_zscore']", "3.5")
    page.fill("input[name='is_high_value']", "1")
    _save(page, "02_form_filled.png")


def test_03_prediction_result(page: Page) -> None:
    """Submitting the form shows a prediction result badge."""
    print("   📸 Taking screenshot 3/6: 03_prediction_result.png...")
    page.goto(BASE_URL)
    page.fill("input[name='amount_log']", "9.2")
    page.fill("input[name='amount_zscore']", "3.5")
    page.fill("input[name='is_high_value']", "1")
    page.click("button[type='submit']")
    page.wait_for_selector("#result-badge", state="visible", timeout=5000)
    badge_text = page.locator("#result-badge").inner_text()
    assert badge_text in ("ANOMALY", "NORMAL"), f"Unexpected badge: {badge_text}"
    _save(page, "03_prediction_result.png")


def test_04_swagger_docs(page: Page) -> None:
    """Swagger UI is accessible at /docs."""
    print("   📸 Taking screenshot 4/6: 04_swagger_docs.png...")
    page.goto(f"{BASE_URL}/docs")
    page.wait_for_load_state("networkidle", timeout=8000)
    # Swagger UI renders title in .title class or as a heading
    title_loc = page.locator(".title, h2, h1").first
    expect(title_loc).to_be_visible(timeout=8000)
    _save(page, "04_swagger_docs.png")


def test_05_metrics_endpoint(page: Page) -> None:
    """Metrics endpoint returns JSON with accuracy field."""
    print("   📸 Taking screenshot 5/6: 05_metrics_endpoint.png...")
    page.goto(f"{BASE_URL}/metrics")
    content = page.content()
    assert "accuracy" in content, "accuracy key not found in /metrics response"
    _save(page, "05_metrics_endpoint.png")


def test_06_health_endpoint(page: Page) -> None:
    """Health endpoint returns ok status."""
    print("   📸 Taking screenshot 6/6: 06_health_endpoint.png...")
    page.goto(f"{BASE_URL}/health")
    content = page.content()
    assert "ok" in content, "'ok' not found in /health response"
    _save(page, "06_health_endpoint.png")
