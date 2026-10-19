"""
Schema validation and constants for raw and processed transaction records.
"""
from typing import Optional, List
import pandas as pd
from pydantic import BaseModel, Field

class RawTransaction(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction UUID")
    timestamp: str = Field(..., description="ISO 8601 formatted timestamp string")
    user_id: str = Field(..., description="User identifier")
    card_id: str = Field(..., description="Card token identifier")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_category: str = Field(..., description="Merchant category code/name")
    device_id: str = Field(..., description="Device fingerprint hash")
    location_country: str = Field(default="US", description="Two-letter country code")
    channel: str = Field(default="web", description="Transaction channel (web, mobile, pos, contactless)")
    is_fraud: Optional[int] = Field(default=None, description="Ground truth fraud label: 0 or 1")

EXPECTED_RAW_COLUMNS: List[str] = [
    "transaction_id",
    "timestamp",
    "user_id",
    "card_id",
    "amount",
    "merchant_id",
    "merchant_category",
    "device_id",
    "location_country",
    "channel",
    "is_fraud"
]

def validate_dataframe_schema(df: pd.DataFrame, require_target: bool = True) -> bool:
    """Validate DataFrame against required columns and types."""
    required = [c for c in EXPECTED_RAW_COLUMNS if require_target or c != "is_fraud"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    
    if df["amount"].lt(0).any():
        raise ValueError("Dataset contains negative transaction amounts.")
        
    return True
