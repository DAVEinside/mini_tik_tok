"""Redis Client Configuration and Connection Management"""

import redis
from redis import ConnectionPool
from typing import Optional, Any, Dict
import json
import pickle
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class RedisClient:
    """Singleton Redis client with connection pooling"""
    
    _instance: Optional['RedisClient'] = None
    _pool: Optional[ConnectionPool] = None
    
    def __new__(cls) -> 'RedisClient':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._init_pool()
    
    def _init_pool(self):
        """Initialize Redis connection pool"""
        self._pool = redis.ConnectionPool(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            max_connections=50,
            socket_keepalive=True,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        logger.info(f"Redis connection pool initialized: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    
    def get_client(self) -> redis.Redis:
        """Get Redis client from pool"""
        return redis.Redis(connection_pool=self._pool, decode_responses=False)
    
    def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            client = self.get_client()
            return client.ping()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            "created_connections": self._pool.created_connections,
            "available_connections": len(self._pool._available_connections),
            "in_use_connections": len(self._pool._in_use_connections),
            "max_connections": self._pool.max_connections
        }

# Global Redis client instance
redis_client = RedisClient()