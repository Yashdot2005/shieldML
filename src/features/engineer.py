"""
Production Feature Engineering Pipeline for ShieldML.
Computes:
1. Velocity features (tx count and amount in 1h, 24h, 7d windows) with zero future leakage.
2. Amount z-score per merchant (fitted strictly on training split).
3. Cyclical hour-of-day and day-of-week encodings (sin/cos).
4. Device fingerprint frequency & distinct cards per device (fitted strictly on training split).
5. Categorical encodings fitted strictly on training split.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ShieldMLFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Learned statistics from training set
        self.merchant_stats_: Dict[str, Dict[str, float]] = {}
        self.global_amount_mean_: float = 0.0
        self.global_amount_std_: float = 1.0
        self.device_frequency_: Dict[str, int] = {}
        self.device_card_counts_: Dict[str, int] = {}
        self.cat_mappings_: Dict[str, Dict[str, int]] = {}
        self.feature_columns_: list = []
        self.is_fitted_: bool = False

    def _compute_cyclical_features(self, dt_series: pd.Series) -> pd.DataFrame:
        """Cyclical encoding for hour (24) and day of week (7)."""
        hours = dt_series.dt.hour
        days = dt_series.dt.dayofweek
        
        return pd.DataFrame({
            "hour_sin": np.sin(2 * np.pi * hours / 24.0),
            "hour_cos": np.cos(2 * np.pi * hours / 24.0),
            "day_sin": np.sin(2 * np.pi * days / 7.0),
            "day_cos": np.cos(2 * np.pi * days / 7.0),
            "is_night": ((hours >= 1) & (hours <= 5)).astype(int)
        }, index=dt_series.index)

    def _compute_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute past-only velocity features (1h, 24h, 7d) per card.
        Guarantees ZERO future data leakage using backward rolling on sorted timestamps.
        """
        df_sorted = df.copy()
        df_sorted["_dt"] = pd.to_datetime(df_sorted["timestamp"])
        df_sorted["_orig_idx"] = np.arange(len(df_sorted))
        
        # Sort by card_id and timestamp
        df_sorted = df_sorted.sort_values(by=["card_id", "_dt"]).reset_index(drop=True)
        
        # We compute rolling counts and sums using closed='left' (strictly past data)
        # To do this efficiently per card:
        grouped = df_sorted.groupby("card_id")
        
        # Rolling 1h
        r_1h = grouped.rolling("1h", on="_dt", closed="left")
        cnt_1h = r_1h["amount"].count().fillna(0).values
        sum_1h = r_1h["amount"].sum().fillna(0.0).values
        
        # Rolling 24h
        r_24h = grouped.rolling("24h", on="_dt", closed="left")
        cnt_24h = r_24h["amount"].count().fillna(0).values
        sum_24h = r_24h["amount"].sum().fillna(0.0).values
        
        # Rolling 7d (168 hours)
        r_7d = grouped.rolling("7D", on="_dt", closed="left")
        cnt_7d = r_7d["amount"].count().fillna(0).values
        sum_7d = r_7d["amount"].sum().fillna(0.0).values
        
        vel_df = pd.DataFrame({
            "tx_count_1h": cnt_1h,
            "tx_amount_1h": sum_1h,
            "tx_count_24h": cnt_24h,
            "tx_amount_24h": sum_24h,
            "tx_count_7d": cnt_7d,
            "tx_amount_7d": sum_7d,
            "_orig_idx": df_sorted["_orig_idx"].values
        })
        
        # Restore original order
        vel_df = vel_df.sort_values(by="_orig_idx").reset_index(drop=True)
        vel_df = vel_df.drop(columns=["_orig_idx"])
        return vel_df

    def fit(self, X: pd.DataFrame, y=None):
        """Fit merchant statistics, device profiles, and category encodings strictly on training set."""
        df = X.copy()
        
        # 1. Global amount statistics
        self.global_amount_mean_ = float(df["amount"].mean())
        self.global_amount_std_ = float(df["amount"].std()) if df["amount"].std() > 0 else 1.0
        
        # 2. Merchant amount mean and std
        m_group = df.groupby("merchant_id")["amount"]
        m_means = m_group.mean().to_dict()
        m_stds = m_group.std().fillna(self.global_amount_std_).to_dict()
        
        self.merchant_stats_ = {
            m: {
                "mean": float(m_means[m]),
                "std": float(m_stds[m]) if m_stds[m] > 1e-4 else self.global_amount_std_
            }
            for m in m_means
        }
        
        # 3. Device frequency and distinct cards per device in training set
        dev_counts = df["device_id"].value_counts().to_dict()
        dev_cards = df.groupby("device_id")["card_id"].nunique().to_dict()
        self.device_frequency_ = {k: int(v) for k, v in dev_counts.items()}
        self.device_card_counts_ = {k: int(v) for k, v in dev_cards.items()}
        
        # 4. Categorical encodings (ordinal mapping)
        cats = ["merchant_category", "channel", "location_country"]
        self.cat_mappings_ = {}
        for c in cats:
            unique_vals = df[c].fillna("UNKNOWN").unique()
            self.cat_mappings_[c] = {val: idx for idx, val in enumerate(unique_vals)}
            
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform raw transactions into model-ready numerical feature matrix."""
        if not self.is_fitted_:
            raise ValueError("ShieldMLFeatureEngineer must be fitted before transforming.")
            
        df = X.copy()
        dt_series = pd.to_datetime(df["timestamp"])
        
        # 1. Velocity features
        vel_features = self._compute_velocity_features(df)
        
        # 2. Cyclical temporal features
        cyclical_features = self._compute_cyclical_features(dt_series)
        
        # 3. Amount z-score per merchant (using fitted statistics)
        merchant_zscores = []
        for _, row in df.iterrows():
            m_id = row["merchant_id"]
            amt = row["amount"]
            if m_id in self.merchant_stats_:
                m_mean = self.merchant_stats_[m_id]["mean"]
                m_std = self.merchant_stats_[m_id]["std"]
            else:
                m_mean = self.global_amount_mean_
                m_std = self.global_amount_std_
            z = (amt - m_mean) / (m_std if m_std > 1e-4 else 1.0)
            merchant_zscores.append(z)
            
        # 4. Device features (frequency and distinct cards seen in training history)
        dev_freqs = [self.device_frequency_.get(dev, 1) for dev in df["device_id"]]
        dev_cards = [self.device_card_counts_.get(dev, 1) for dev in df["device_id"]]
        
        # 5. Encoded categoricals
        cat_cols = {}
        for c in ["merchant_category", "channel", "location_country"]:
            mapping = self.cat_mappings_[c]
            # Use -1 for unseen categories
            cat_cols[f"{c}_code"] = df[c].map(lambda x: mapping.get(x, -1)).values
            
        # 6. High-risk domain flags
        is_high_risk_cat = df["merchant_category"].isin(["electronics", "luxury_jewelry", "digital_gaming"]).astype(int).values
        is_intl = (df["location_country"] != "US").astype(int).values
        amount_log = np.log1p(np.maximum(0, df["amount"].values))
        
        features_dict = {
            "amount": df["amount"].values,
            "amount_log": amount_log,
            "merchant_amount_zscore": np.array(merchant_zscores),
            "device_frequency": np.array(dev_freqs),
            "distinct_cards_per_device": np.array(dev_cards),
            "is_high_risk_category": is_high_risk_cat,
            "is_international": is_intl,
            **{k: cyclical_features[k].values for k in cyclical_features.columns},
            **{k: vel_features[k].values for k in vel_features.columns},
            **cat_cols
        }
        
        feature_df = pd.DataFrame(features_dict, index=df.index)
        self.feature_columns_ = list(feature_df.columns)
        return feature_df
        
    def transform_single(self, tx: dict, past_velocity: dict) -> pd.DataFrame:
        """
        Sub-millisecond single-transaction transformation for live FastAPI serving.
        past_velocity contains pre-computed or feature-store cached 1h, 24h, 7d counts.
        """
        dt = pd.to_datetime(tx["timestamp"])
        hour = dt.hour
        day = dt.dayofweek
        
        # Merchant stats
        m_id = tx.get("merchant_id", "")
        amt = float(tx.get("amount", 0.0))
        if m_id in self.merchant_stats_:
            m_mean = self.merchant_stats_[m_id]["mean"]
            m_std = self.merchant_stats_[m_id]["std"]
        else:
            m_mean = self.global_amount_mean_
            m_std = self.global_amount_std_
        z = (amt - m_mean) / (m_std if m_std > 1e-4 else 1.0)
        
        dev = tx.get("device_id", "")
        dev_freq = self.device_frequency_.get(dev, 1)
        dev_cards = self.device_card_counts_.get(dev, 1)
        
        m_cat = tx.get("merchant_category", "")
        channel = tx.get("channel", "")
        country = tx.get("location_country", "US")
        
        row_dict = {
            "amount": amt,
            "amount_log": np.log1p(max(0.0, amt)),
            "merchant_amount_zscore": z,
            "device_frequency": dev_freq,
            "distinct_cards_per_device": dev_cards,
            "is_high_risk_category": 1 if m_cat in ["electronics", "luxury_jewelry", "digital_gaming"] else 0,
            "is_international": 1 if country != "US" else 0,
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "day_sin": np.sin(2 * np.pi * day / 7.0),
            "day_cos": np.cos(2 * np.pi * day / 7.0),
            "is_night": 1 if 1 <= hour <= 5 else 0,
            "tx_count_1h": past_velocity.get("tx_count_1h", 0),
            "tx_amount_1h": past_velocity.get("tx_amount_1h", 0.0),
            "tx_count_24h": past_velocity.get("tx_count_24h", 0),
            "tx_amount_24h": past_velocity.get("tx_amount_24h", 0.0),
            "tx_count_7d": past_velocity.get("tx_count_7d", 0),
            "tx_amount_7d": past_velocity.get("tx_amount_7d", 0.0),
            "merchant_category_code": self.cat_mappings_.get("merchant_category", {}).get(m_cat, -1),
            "channel_code": self.cat_mappings_.get("channel", {}).get(channel, -1),
            "location_country_code": self.cat_mappings_.get("location_country", {}).get(country, -1),
        }
        
        return pd.DataFrame([row_dict])[self.feature_columns_]
