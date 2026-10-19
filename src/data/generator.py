"""
Synthetic Financial Transaction Dataset Generator for ShieldML.
Produces realistic time-series transaction records with extreme class imbalance (~1.2% fraud)
and real-world fraud patterns (velocity spikes, merchant amount z-score anomalies, nocturnal bursts,
device fingerprint collisions).
"""
import uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.config import settings

def generate_synthetic_transactions(
    n_transactions: int = 60000,
    n_users: int = 2500,
    n_merchants: int = 150,
    n_devices: int = 4000,
    start_date: str = "2026-01-01 00:00:00",
    random_seed: int = 42
) -> pd.DataFrame:
    """Generate realistic financial transactions with time-series fraud dynamics."""
    np.random.seed(random_seed)
    
    # Define merchant categories and typical average ticket sizes & std
    categories = {
        "grocery": {"mean": 65.0, "std": 25.0, "weight": 0.30, "risk": 0.05},
        "gas_station": {"mean": 45.0, "std": 15.0, "weight": 0.20, "risk": 0.10},
        "dining": {"mean": 38.0, "std": 20.0, "weight": 0.18, "risk": 0.08},
        "online_retail": {"mean": 110.0, "std": 60.0, "weight": 0.15, "risk": 0.20},
        "electronics": {"mean": 380.0, "std": 180.0, "weight": 0.07, "risk": 0.40},
        "travel_hotel": {"mean": 450.0, "std": 220.0, "weight": 0.05, "risk": 0.35},
        "luxury_jewelry": {"mean": 950.0, "std": 400.0, "weight": 0.03, "risk": 0.50},
        "digital_gaming": {"mean": 28.0, "std": 18.0, "weight": 0.02, "risk": 0.45}
    }
    
    cat_names = list(categories.keys())
    cat_weights = [categories[c]["weight"] for c in cat_names]
    
    # Generate merchant pool
    merchants = []
    for i in range(n_merchants):
        cat = np.random.choice(cat_names, p=cat_weights)
        m_id = f"M_{i:04d}"
        # Individual merchant specific mean around category mean
        m_mean = max(10.0, np.random.normal(categories[cat]["mean"], categories[cat]["std"] * 0.3))
        m_std = max(3.0, categories[cat]["std"] * np.random.uniform(0.7, 1.3))
        merchants.append({
            "merchant_id": m_id,
            "merchant_category": cat,
            "m_mean": m_mean,
            "m_std": m_std,
            "risk_weight": categories[cat]["risk"]
        })
    merchant_df = pd.DataFrame(merchants)
    
    # Users, cards, and primary devices
    user_ids = [f"U_{i:05d}" for i in range(n_users)]
    user_card_map = {u: f"CARD_{u[2:]}" for u in user_ids}
    user_device_map = {u: f"DEV_{np.random.randint(0, n_devices):05d}" for u in user_ids}
    
    channels = ["pos_chip", "pos_swipe", "contactless", "web", "mobile_app"]
    channel_weights = [0.35, 0.10, 0.25, 0.20, 0.10]
    countries = ["US", "US", "US", "US", "CA", "GB", "MX", "DE"]
    
    # Fraud ring devices
    fraud_devices = [f"FRAUD_DEV_{i:03d}" for i in range(15)]
    
    base_time = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    # Generate time increments (Poisson process for realistic arrival)
    inter_arrivals = np.random.exponential(scale=35.0, size=n_transactions) # ~35 seconds avg
    cum_seconds = np.cumsum(inter_arrivals)
    timestamps = [base_time + timedelta(seconds=float(s)) for s in cum_seconds]
    
    records = []
    
    # Keep track of recent card events for velocity simulation
    card_recent_tx = {}
    
    for i in range(n_transactions):
        t = timestamps[i]
        tx_id = f"TX_{uuid.uuid4().hex[:12].upper()}"
        
        # Decide if this transaction is candidate for fraud (~1.2% base rate)
        # Higher fraud probability during night hours (01:00 to 05:00 UTC)
        hour = t.hour
        is_night = 1 <= hour <= 5
        base_fraud_prob = 0.008 + (0.012 if is_night else 0.0)
        
        is_fraud = 1 if np.random.rand() < base_fraud_prob else 0
        
        # Select user
        u = np.random.choice(user_ids)
        card = user_card_map[u]
        normal_dev = user_device_map[u]
        
        # Select merchant
        if is_fraud:
            # Fraud favors high-risk categories (electronics, luxury, gaming)
            high_risk_merchants = merchant_df[merchant_df["risk_weight"] >= 0.35]
            m_row = high_risk_merchants.sample(1).iloc[0]
            # Fraud patterns
            fraud_type = np.random.choice(["burst_velocity", "amount_anomaly", "device_hijack", "night_heist"])
            
            if fraud_type == "amount_anomaly":
                # High amount relative to merchant mean: z-score between +3.5 and +8.0
                z = np.random.uniform(3.5, 8.0)
                amount = round(m_row["m_mean"] + z * m_row["m_std"], 2)
                device = normal_dev if np.random.rand() > 0.5 else np.random.choice(fraud_devices)
                ch = np.random.choice(["web", "mobile_app"])
            elif fraud_type == "device_hijack":
                # Fraud device attacking multiple cards
                device = np.random.choice(fraud_devices)
                amount = round(max(5.0, np.random.normal(m_row["m_mean"] * 1.8, m_row["m_std"])), 2)
                ch = "web"
            elif fraud_type == "burst_velocity":
                # Bursts of medium/high amount
                device = np.random.choice(fraud_devices) if np.random.rand() > 0.4 else normal_dev
                amount = round(max(20.0, np.random.normal(m_row["m_mean"] * 1.4, m_row["m_std"])), 2)
                ch = "web"
            else: # night heist
                device = np.random.choice(fraud_devices)
                amount = round(max(50.0, np.random.normal(m_row["m_mean"] * 2.2, m_row["m_std"] * 1.5)), 2)
                ch = "mobile_app"
            
            country = np.random.choice(["US", "RU", "NG", "CN", "BR"]) if np.random.rand() > 0.6 else "US"
        else:
            # Legitimate transaction
            m_row = merchant_df.sample(1, weights=merchant_df["risk_weight"].apply(lambda r: 1.0 - r + 0.1)).iloc[0]
            # Normal amount around merchant mean
            raw_amt = np.random.normal(m_row["m_mean"], m_row["m_std"])
            amount = round(max(2.50, raw_amt), 2)
            device = normal_dev if np.random.rand() > 0.05 else f"DEV_{np.random.randint(0, n_devices):05d}"
            ch = np.random.choice(channels, p=channel_weights)
            country = np.random.choice(countries)
            
        records.append({
            "transaction_id": tx_id,
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": u,
            "card_id": card,
            "amount": amount,
            "merchant_id": m_row["merchant_id"],
            "merchant_category": m_row["merchant_category"],
            "device_id": device,
            "location_country": country,
            "channel": ch,
            "is_fraud": is_fraud
        })
        
    df = pd.DataFrame(records)
    # Sort strictly by timestamp to maintain temporal ordering
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp_dt").reset_index(drop=True)
    df = df.drop(columns=["timestamp_dt"])
    
    return df

def save_raw_dataset(df: pd.DataFrame) -> str:
    """Save raw transactions to disk."""
    output_path = settings.RAW_DATA_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions -> {output_path}")
    print(f"Fraud count: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    return str(output_path)

if __name__ == "__main__":
    df = generate_synthetic_transactions()
    save_raw_dataset(df)
