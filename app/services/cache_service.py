import redis
import json
import numpy as np
from typing import List, Optional, Dict, Any
import pickle
import hashlib
from app.core.config import settings

class RedisCacheService:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=False
        )
        self.ttl = settings.REDIS_CACHE_TTL
        
    def _generate_key(self, user_id: int, video_history: List[int]) -> str:
        """Generate cache key from user_id and video history"""
        history_str = ','.join(map(str, sorted(video_history[-10:])))  # Last 10 videos
        key_str = f"rec:user:{user_id}:history:{history_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_recommendations(self, user_id: int, video_history: List[int]) -> Optional[Dict[str, Any]]:
        """Get cached recommendations"""
        key = self._generate_key(user_id, video_history)
        
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            print(f"Cache retrieval error: {e}")
            
        return None
    
    def set_recommendations(self, user_id: int, video_history: List[int], 
                          recommendations: List[int], scores: List[float]) -> bool:
        """Cache recommendations"""
        key = self._generate_key(user_id, video_history)
        
        data = {
            'user_id': user_id,
            'recommendations': recommendations,
            'scores': scores,
            'timestamp': np.datetime64('now').astype(str)
        }
        
        try:
            self.redis_client.setex(
                key,
                self.ttl,
                pickle.dumps(data)
            )
            return True
        except Exception as e:
            print(f"Cache storage error: {e}")
            return False
    
    def invalidate_user_cache(self, user_id: int):
        """Invalidate all cache entries for a user"""
        pattern = f"rec:user:{user_id}:*"
        for key in self.redis_client.scan_iter(match=pattern):
            self.redis_client.delete(key)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        info = self.redis_client.info()
        return {
            'used_memory': info['used_memory_human'],
            'total_keys': self.redis_client.dbsize(),
            'hit_rate': info.get('keyspace_hits', 0) / max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 1), 1),
            'connected_clients': info['connected_clients']
        }