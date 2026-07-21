"""Integration tests for FastAPI endpoints using TestClient."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_health_endpoints():
    """Test GET /health and GET /health/ready endpoints."""
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    data_health = resp_health.json()
    assert data_health["status"] == "ok"
    assert "version" in data_health

    resp_ready = client.get("/health/ready")
    assert resp_ready.status_code == 200
    data_ready = resp_ready.json()
    assert "models_available" in data_ready


def test_api_market_data_endpoint():
    """Test GET /market/{symbol} endpoint."""
    resp = client.get("/market/AAPL")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "AAPL"
    assert data["price"] > 0.0
    assert "data_freshness_status" in data


def test_api_prediction_endpoint():
    """Test GET /prediction/{symbol} endpoint."""
    resp = client.get("/prediction/AAPL")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "AAPL"
    assert "1d" in data["predicted_returns"]
    assert "1d" in data["predicted_prices"]
    assert data["confidence_level"] in ["HIGH", "MEDIUM", "LOW"]


def test_api_sentiment_endpoint():
    """Test GET /sentiment/{symbol} endpoint."""
    resp = client.get("/sentiment/AAPL")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "AAPL"
    assert "news_count" in data


def test_api_analysis_post_endpoint():
    """Test POST /analysis endpoint."""
    payload = {
        "symbol": "AAPL",
        "question": "Why is the model forecast positive?",
        "period": "1mo",
    }
    resp = client.post("/analysis", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "confidence_level" in data
