
import numpy as np
from collections import deque
from typing import Dict, List, Any
import threading
from datetime import datetime

class MetricsCollector:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.cache_hits = deque(maxlen=window_size)
        self.user_requests = {}
        self.lock = threading.Lock()
        
    def record_request(self, user_id: int, cache_hit: bool, latency: float):
        """Record a recommendation request"""
        with self.lock:
            self.latencies.append(latency)
            self.cache_hits.append(1 if cache_hit else 0)
            
            if user_id not in self.user_requests:
                self.user_requests[user_id] = 0
            self.user_requests[user_id] += 1
    
    def get_p95_latency(self) -> float:
        """Get P95 latency in milliseconds"""
        with self.lock:
            if not self.latencies:
                return 0.0
            return np.percentile(list(self.latencies), 95)
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        with self.lock:
            if not self.cache_hits:
                return 0.0
            return sum(self.cache_hits) / len(self.cache_hits)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        with self.lock:
            latencies_list = list(self.latencies)
            
            if not latencies_list:
                return {
                    'p95_latency_ms': 0.0,
                    'p99_latency_ms': 0.0,
                    'mean_latency_ms': 0.0,
                    'cache_hit_rate': 0.0,
                    'total_requests': 0,
                    'unique_users': 0
                }
            
            return {
                'p95_latency_ms': np.percentile(latencies_list, 95),
                'p99_latency_ms': np.percentile(latencies_list, 99),
                'mean_latency_ms': np.mean(latencies_list),
                'cache_hit_rate': self.get_cache_hit_rate(),
                'total_requests': len(latencies_list),
                'unique_users': len(self.user_requests),
                'timestamp': datetime.utcnow().isoformat()
            }