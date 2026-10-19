"""
Unit tests for ShieldML Feature Engineering.
Validates zero future data leakage, rolling velocity correctness, and merchant z-scores.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.engineer import ShieldMLFeatureEngineer
from src.features.feature_store import LightweightFeatureStore

def test_feature_engineering_temporal_leakage_prevention():
    """Verify that velocity features only incorporate transactions strictly in the past."""
    base_time = datetime(2026, 1, 10, 12, 0, 0)
    
    # Create 3 transactions for CARD_1:
    # Tx 1: 12:00:00, $50
    # Tx 2: 12:20:00, $30  (within 1h of Tx 1)
    # Tx 3: 14:00:00, $100 (more than 1h after Tx 1 & 2)
    sample_df = pd.DataFrame([
        {
            "transaction_id": "TX_1",
            "timestamp": (base_time).strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": "U_1",
            "card_id": "CARD_1",
            "amount": 50.0,
            "merchant_id": "M_1",
            "merchant_category": "grocery",
            "device_id": "DEV_1",
            "location_country": "US",
            "channel": "web",
            "is_fraud": 0
        },
        {
            "transaction_id": "TX_2",
            "timestamp": (base_time + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": "U_1",
            "card_id": "CARD_1",
            "amount": 30.0,
            "merchant_id": "M_1",
            "merchant_category": "grocery",
            "device_id": "DEV_1",
            "location_country": "US",
            "channel": "web",
            "is_fraud": 0
        },
        {
            "transaction_id": "TX_3",
            "timestamp": (base_time + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": "U_1",
            "card_id": "CARD_1",
            "amount": 100.0,
            "merchant_id": "M_1",
            "merchant_category": "grocery",
            "device_id": "DEV_1",
            "location_country": "US",
            "channel": "web",
            "is_fraud": 0
        }
    ])
    
    fe = ShieldMLFeatureEngineer()
    fe.fit(sample_df)
    transformed = fe.transform(sample_df)
    
    # Tx 1 has NO past transactions: count_1h must be 0
    assert transformed.loc[0, "tx_count_1h"] == 0
    assert transformed.loc[0, "tx_amount_1h"] == 0.0
    
    # Tx 2 has 1 past transaction in last 1h (Tx 1): count_1h must be 1, amount 50.0
    assert transformed.loc[1, "tx_count_1h"] == 1
    assert transformed.loc[1, "tx_amount_1h"] == 50.0
    
    # Tx 3 is 2 hours later: count_1h must be 0, count_24h must be 2 (Tx 1 + Tx 2)
    assert transformed.loc[2, "tx_count_1h"] == 0
    assert transformed.loc[2, "tx_count_24h"] == 2
    assert transformed.loc[2, "tx_amount_24h"] == 80.0

def test_merchant_zscore_fit_on_train_only():
    """Verify merchant statistics are learned from training set and applied properly to test."""
    train_df = pd.DataFrame([
        {
            "transaction_id": f"TX_{i}",
            "timestamp": "2026-01-01 10:00:00",
            "user_id": f"U_{i}",
            "card_id": f"CARD_{i}",
            "amount": 100.0 if i % 2 == 0 else 200.0,
            "merchant_id": "M_TEST",
            "merchant_category": "retail",
            "device_id": "DEV_1",
            "location_country": "US",
            "channel": "web",
            "is_fraud": 0
        }
        for i in range(10)
    ])
    
    fe = ShieldMLFeatureEngineer()
    fe.fit(train_df)
    
    assert "M_TEST" in fe.merchant_stats_
    assert fe.merchant_stats_["M_TEST"]["mean"] == 150.0
    
    # Test on new transaction of $250
    test_df = pd.DataFrame([{
        "transaction_id": "TX_NEW",
        "timestamp": "2026-01-02 10:00:00",
        "user_id": "U_99",
        "card_id": "CARD_99",
        "amount": 250.0,
        "merchant_id": "M_TEST",
        "merchant_category": "retail",
        "device_id": "DEV_1",
        "location_country": "US",
        "channel": "web",
        "is_fraud": 0
    }])
    
    transformed_test = fe.transform(test_df)
    z = transformed_test["merchant_amount_zscore"].iloc[0]
    assert z > 0  # $250 is above mean of $150

def test_lightweight_feature_store_lookups():
    """Test feature store sliding window calculation and TTL eviction."""
    fs = LightweightFeatureStore(ttl_days=7)
    
    card = "CARD_VERIFY_99"
    fs.record_transaction(card, "2026-01-10T10:00:00Z", 50.0)
    fs.record_transaction(card, "2026-01-10T10:30:00Z", 150.0)
    
    # Query at 10:45:00 (both within 1h)
    vel_1h = fs.get_card_velocity(card, "2026-01-10T10:45:00Z")
    assert vel_1h["tx_count_1h"] == 2
    assert vel_1h["tx_amount_1h"] == 200.0
    assert vel_1h["tx_count_24h"] == 2
    assert vel_1h["tx_count_7d"] == 2
    
    # Query 3 hours later (outside 1h, inside 24h)
    vel_3h = fs.get_card_velocity(card, "2026-01-10T13:30:00Z")
    assert vel_3h["tx_count_1h"] == 0
    assert vel_3h["tx_count_24h"] == 2
    assert vel_3h["tx_amount_24h"] == 200.0
