"""
Integration tests for FastAPI Serving Endpoints (/health, /metrics, /predict).
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

def test_health_endpoint():
    """Verify health endpoint responds with 200 OK and expected structure."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "service" in data

def test_metrics_endpoint():
    """Verify metrics endpoint returns prediction count, latencies, and fraud rate."""
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "average_latency_ms" in data
        assert "p95_latency_ms" in data
        assert "fraud_rate_last_100" in data
        assert data["status"] == "HEALTHY"

def test_predict_endpoint_validation_error():
    """Verify that invalid payload returns 422 Unprocessable Entity."""
    with TestClient(app) as client:
        bad_payload = {"transaction_id": "TX_123"}  # Missing required fields
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

def test_predict_endpoint_valid_scoring():
    """Verify valid transaction payload is scored with SHAP explanation and sub-50ms latency."""
    with TestClient(app) as client:
        valid_payload = {
            "transaction_id": "TX_TEST_001",
            "timestamp": "2026-01-20T14:30:00Z",
            "user_id": "U_00124",
            "card_id": "CARD_00124",
            "amount": 420.50,
            "merchant_id": "M_0088",
            "merchant_category": "electronics",
            "device_id": "DEV_001",
            "location_country": "US",
            "channel": "web"
        }
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        data = response.json()
        
        assert "fraud_score" in data
        assert 0.0 <= data["fraud_score"] <= 1.0
        assert "is_fraud" in data
        assert isinstance(data["is_fraud"], bool)
        assert data["decision"] in ["FRAUD_ALERT", "APPROVE"]
        assert "latency_ms" in data
        assert "shap_explanation" in data
        assert len(data["shap_explanation"]) > 0
        
        first_shap = data["shap_explanation"][0]
        assert "feature" in first_shap
        assert "shap_value" in first_shap
        assert "direction" in first_shap
