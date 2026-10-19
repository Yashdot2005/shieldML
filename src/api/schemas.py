"""
Pydantic Request and Response Schemas for the ShieldML Serving API.
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class TransactionPayload(BaseModel):
    transaction_id: str = Field(..., example="TX_892301BA")
    timestamp: str = Field(..., example="2026-01-15T14:32:00Z")
    user_id: str = Field(..., example="U_00124")
    card_id: str = Field(..., example="CARD_00124")
    amount: float = Field(..., gt=0, example=345.50)
    merchant_id: str = Field(..., example="M_0042")
    merchant_category: str = Field(..., example="electronics")
    device_id: str = Field(..., example="DEV_00891")
    location_country: str = Field(default="US", example="US")
    channel: str = Field(default="web", example="web")

class ShapFeatureContribution(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    direction: Literal["RISK_INCREASING", "RISK_DECREASING"]

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_score: float = Field(..., description="Calibrated probability of fraud in [0.0, 1.0]")
    is_fraud: bool = Field(..., description="Binary classification at optimized F2 threshold")
    decision: Literal["FRAUD_ALERT", "APPROVE"]
    threshold_applied: float
    latency_ms: float
    model_version: str
    shap_explanation: List[ShapFeatureContribution]

class MetricsResponse(BaseModel):
    total_predictions: int
    window_size: int
    average_latency_ms: float
    p95_latency_ms: float
    fraud_rate_last_100: float
    total_fraud_detected: int
    uptime_seconds: float
    status: str
