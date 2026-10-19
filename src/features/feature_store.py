"""
Lightweight in-memory real-time Feature Store for low-latency (<1ms) velocity lookups.
Maintains rolling window state per card/account with automatic TTL eviction (>7 days).
"""
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Any, List

class LightweightFeatureStore:
    def __init__(self, ttl_days: int = 7):
        self.ttl_seconds = ttl_days * 86400
        # Maps card_id -> deque of (timestamp_dt, amount)
        self.card_history: Dict[str, deque] = defaultdict(deque)

    def record_transaction(self, card_id: str, timestamp_str: str, amount: float):
        """Record an incoming transaction and clean up entries older than 7 days."""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00").split("+")[0])
        except Exception:
            dt = datetime.utcnow()
            
        history = self.card_history[card_id]
        history.append((dt, float(amount)))
        
        # Evict transactions older than 7 days relative to this transaction
        cutoff = dt - timedelta(seconds=self.ttl_seconds)
        while history and history[0][0] < cutoff:
            history.popleft()

    def get_card_velocity(self, card_id: str, current_timestamp_str: str) -> Dict[str, float]:
        """
        Query past-only velocity metrics for a card in 1h, 24h, and 7d windows.
        Sub-millisecond latency.
        """
        try:
            now_dt = datetime.fromisoformat(current_timestamp_str.replace("Z", "+00:00").split("+")[0])
        except Exception:
            now_dt = datetime.utcnow()

        history = self.card_history.get(card_id, deque())
        if not history:
            return {
                "tx_count_1h": 0,
                "tx_amount_1h": 0.0,
                "tx_count_24h": 0,
                "tx_amount_24h": 0.0,
                "tx_count_7d": 0,
                "tx_amount_7d": 0.0
            }

        cutoff_1h = now_dt - timedelta(hours=1)
        cutoff_24h = now_dt - timedelta(hours=24)
        cutoff_7d = now_dt - timedelta(days=7)

        cnt_1h, sum_1h = 0, 0.0
        cnt_24h, sum_24h = 0, 0.0
        cnt_7d, sum_7d = 0, 0.0

        for tx_time, amt in reversed(history):
            if tx_time >= now_dt:
                # strictly past events only
                continue
            if tx_time < cutoff_7d:
                # since history is chronologically ordered, we can break early
                break
            
            cnt_7d += 1
            sum_7d += amt

            if tx_time >= cutoff_24h:
                cnt_24h += 1
                sum_24h += amt

            if tx_time >= cutoff_1h:
                cnt_1h += 1
                sum_1h += amt

        return {
            "tx_count_1h": cnt_1h,
            "tx_amount_1h": sum_1h,
            "tx_count_24h": cnt_24h,
            "tx_amount_24h": sum_24h,
            "tx_count_7d": cnt_7d,
            "tx_amount_7d": sum_7d
        }
