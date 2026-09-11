"""Unit tests for FastAPI serving layer (Phase 10).

Runs offline using Starlette TestClient.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestServingEndpoints:
    def test_health_check(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data

    def test_get_regions(self, client: TestClient):
        response = client.get("/api/regions")
        assert response.status_code == 200
        data = response.json()
        assert "tiers" in data
        assert "Large" in data["tiers"]
        assert "Medium" in data["tiers"]
        assert data["total_regions"] == 52

    def test_serve_ui(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "Energy Demand Forecasting" in response.text
        assert "<canvas id=\"forecastChart\">" in response.text

    def test_predict_validation_error_when_missing_respondent(self, client: TestClient):
        response = client.post("/api/predict", json={})
        assert response.status_code == 422
