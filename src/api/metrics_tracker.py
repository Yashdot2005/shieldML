"""
Thread-safe metrics monitoring tracker for ShieldML API.
Tracks total count, rolling 100-request latency, average latency, p95 latency, and fraud rate.
"""
import time
from collections import deque
import threading
import numpy as np

class MetricsTracker:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        self.total_count = 0
        self.total_fraud = 0
        self.recent_latencies = deque(maxlen=window_size)
        self.recent_fraud_flags = deque(maxlen=window_size)

    def record_prediction(self, latency_ms: float, is_fraud: bool):
        """Record prediction execution metrics thread-safely."""
        with self.lock:
            self.total_count += 1
            if is_fraud:
                self.total_fraud += 1
            self.recent_latencies.append(latency_ms)
            self.recent_fraud_flags.append(1 if is_fraud else 0)

    def get_summary(self) -> dict:
        """Get summary statistics over recent window."""
        with self.lock:
            count = len(self.recent_latencies)
            if count == 0:
                avg_lat = 0.0
                p95_lat = 0.0
                fraud_rate = 0.0
            else:
                avg_lat = float(np.mean(self.recent_latencies))
                p95_lat = float(np.percentile(self.recent_latencies, 95))
                fraud_rate = float(np.mean(self.recent_fraud_flags))
                
            uptime = time.time() - self.start_time
            
            return {
                "total_predictions": self.total_count,
                "window_size": count,
                "average_latency_ms": round(avg_lat, 2),
                "p95_latency_ms": round(p95_lat, 2),
                "fraud_rate_last_100": round(fraud_rate, 4),
                "total_fraud_detected": self.total_fraud,
                "uptime_seconds": round(uptime, 1),
                "status": "HEALTHY"
            }
