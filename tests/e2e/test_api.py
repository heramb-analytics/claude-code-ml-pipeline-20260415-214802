"""Playwright E2E tests for transaction anomaly detection API."""

import re
from pathlib import Path

import pytest
import requests
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:8000"
SCREENSHOTS_DIR = Path("reports/screenshots")


@pytest.fixture(autouse=True)
def ensure_screenshots_dir():
    """Ensure screenshots directory exists."""
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def test_dashboard_home(page: Page) -> None:
    """Test dashboard home page loads with correct title and live status."""
    print("   📸 Taking screenshot 1/6: 01_dashboard_home.png...")
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    expect(page.locator("h1")).to_contain_text("Transaction Anomaly Detection")
    page.screenshot(path=str(SCREENSHOTS_DIR / "01_dashboard_home.png"), full_page=True)
    size = (SCREENSHOTS_DIR / "01_dashboard_home.png").stat().st_size
    print(f"   ✅ Saved: reports/screenshots/01_dashboard_home.png ({size} bytes)")


def test_form_filled(page: Page) -> None:
    """Test filling in the prediction form with anomalous values."""
    print("   📸 Taking screenshot 2/6: 02_form_filled.png...")
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.fill("input[name='amount']", "9999.99")
    page.fill("input[name='hour_of_day']", "2")
    page.fill("input[name='day_of_week']", "6")
    page.fill("input[name='merchant_txn_count']", "1")
    page.screenshot(path=str(SCREENSHOTS_DIR / "02_form_filled.png"), full_page=True)
    size = (SCREENSHOTS_DIR / "02_form_filled.png").stat().st_size
    print(f"   ✅ Saved: reports/screenshots/02_form_filled.png ({size} bytes)")


def test_prediction_result(page: Page) -> None:
    """Test submitting the form and getting a prediction badge."""
    print("   📸 Taking screenshot 3/6: 03_prediction_result.png...")
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.fill("input[name='amount']", "9999.99")
    page.fill("input[name='hour_of_day']", "3")
    page.fill("input[name='day_of_week']", "6")
    page.fill("input[name='merchant_txn_count']", "1")
    page.click("button[type='submit']")
    page.wait_for_selector("#badge:not(.hidden)", timeout=10000)
    page.screenshot(path=str(SCREENSHOTS_DIR / "03_prediction_result.png"), full_page=True)
    size = (SCREENSHOTS_DIR / "03_prediction_result.png").stat().st_size
    print(f"   ✅ Saved: reports/screenshots/03_prediction_result.png ({size} bytes)")


def test_swagger_docs(page: Page) -> None:
    """Test Swagger UI loads correctly."""
    print("   📸 Taking screenshot 4/6: 04_swagger_docs.png...")
    page.goto(f"{BASE_URL}/docs")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector(".swagger-ui", timeout=10000)
    page.screenshot(path=str(SCREENSHOTS_DIR / "04_swagger_docs.png"), full_page=True)
    size = (SCREENSHOTS_DIR / "04_swagger_docs.png").stat().st_size
    print(f"   ✅ Saved: reports/screenshots/04_swagger_docs.png ({size} bytes)")


def test_metrics_endpoint(page: Page) -> None:
    """Test /metrics endpoint returns JSON with algorithm field."""
    print("   📸 Taking screenshot 5/6: 05_metrics_endpoint.png...")
    page.goto(f"{BASE_URL}/metrics")
    page.wait_for_load_state("networkidle")
    content = page.content()
    assert "IsolationForest" in content or "algorithm" in content
    page.screenshot(path=str(SCREENSHOTS_DIR / "05_metrics_endpoint.png"), full_page=True)
    size = (SCREENSHOTS_DIR / "05_metrics_endpoint.png").stat().st_size
    print(f"   ✅ Saved: reports/screenshots/05_metrics_endpoint.png ({size} bytes)")


def test_health_endpoint(page: Page) -> None:
    """Test /health endpoint returns ok status."""
    print("   📸 Taking screenshot 6/6: 06_health_endpoint.png...")
    page.goto(f"{BASE_URL}/health")
    page.wait_for_load_state("networkidle")
    content = page.content()
    assert "ok" in content
    page.screenshot(path=str(SCREENSHOTS_DIR / "06_health_endpoint.png"), full_page=True)
    size = (SCREENSHOTS_DIR / "06_health_endpoint.png").stat().st_size
    print(f"   ✅ Saved: reports/screenshots/06_health_endpoint.png ({size} bytes)")
