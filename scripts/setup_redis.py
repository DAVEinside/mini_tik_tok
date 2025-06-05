"""Setup Redis for video recommendation system"""

import redis
import json
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Dict, List, Any
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedisSetup:
    """Setup and initialize Redis for the recommendation system"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        
    def check_connection(self) -> bool:
        """Check Redis connection"""
        try:
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
            return True
        except redis.ConnectionError:
            logger.error("Failed to connect to Redis. Make sure Redis server is running.")
            return False
    
    def clear_database(self, pattern: str = "*"):
        """Clear Redis database"""
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)
            logger.info(f"Deleted {len(keys)} keys")
    
    def setup_indices(self):
        """Setup Redis indices for efficient retrieval"""
        # Create indices for different data types
        indices = {
            "user:*": "Hash containing user profile data",
            "video:*": "Hash containing video metadata",
            "rec:user:*": "Sorted set of recommendations per user",
            "history:user:*": "List of user watch history",
            "features:video:*": "Video feature vectors",
            "stats:*": "System statistics and metrics"
        }
        
        # Store index information
        for key_pattern, description in indices.items():
            self.redis_client.hset("indices", key_pattern, description)
        
        logger.info("Redis indices configured")
    
    def load_video_metadata(self, metadata_path: str = "data/processed/video_metadata.json"):
        """Load video metadata into Redis"""
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        pipeline = self.redis_client.pipeline()
        
        for video_id, data in metadata.items():
            key = f"video:{video_id}"
            # Store as hash
            pipeline.hset(key, mapping={
                'title': data['title'],
                'duration': data['duration'],
                'categories': json.dumps(data['categories']),
                'creator_id': data['creator_id'],
                'upload_date': data['upload_date'],
                'view_count': data.get('view_count', 0),
                'like_ratio': data.get('like_ratio', 0.0)
            })
            
            # Add to category sets
            for category in data['categories']:
                pipeline.sadd(f"category:{category}", video_id)
        
        pipeline.execute()
        logger.info(f"Loaded {len(metadata)} video metadata entries")
    
    def load_user_profiles(self, profiles_path: str = "data/processed/user_profiles.json"):
        """Load user profiles into Redis"""
        try:
            with open(profiles_path, 'r') as f:
                profiles = json.load(f)
            
            pipeline = self.redis_client.pipeline()
            
            for user_id, data in profiles.items():
                key = f"user:{user_id}"
                pipeline.hset(key, mapping={
                    'age_group': data['age_group'],
                    'join_date': data['join_date'],
                    'preferences': json.dumps(data['preferences']),
                    'activity_level': data['activity_level']
                })
            
            pipeline.execute()
            logger.info(f"Loaded {len(profiles)} user profiles")
        except FileNotFoundError:
            logger.warning(f"User profiles file not found: {profiles_path}")
    
    def load_video_features(self, features_path: str = "data/processed/video_features.npy"):
        """Load video features into Redis"""
        try:
            features = np.load(features_path)
            
            pipeline = self.redis_client.pipeline()
            
            for video_id, feature_vector in enumerate(features):
                key = f"features:video:{video_id}"
                # Store as binary
                pipeline.set(key, feature_vector.tobytes())
            
            pipeline.execute()
            logger.info(f"Loaded features for {len(features)} videos")
        except FileNotFoundError:
            logger.warning(f"Video features file not found: {features_path}")
    
    def create_sample_recommendations(self, num_users: int = 100):
        """Create sample recommendations for testing"""
        pipeline = self.redis_client.pipeline()
        
        for user_id in range(num_users):
            key = f"rec:user:{user_id}"
            
            # Generate random recommendations with scores
            recommendations = {}
            for _ in range(50):
                video_id = np.random.randint(0, 10000)
                score = np.random.random()
                recommendations[video_id] = score
            
            # Add to sorted set
            pipeline.zadd(key, recommendations)
            pipeline.expire(key, 3600)  # 1 hour TTL
        
        pipeline.execute()
        logger.info(f"Created sample recommendations for {num_users} users")
    
    def setup_monitoring(self):
        """Setup monitoring keys"""
        # Initialize counters
        self.redis_client.set("stats:total_requests", 0)
        self.redis_client.set("stats:cache_hits", 0)
        self.redis_client.set("stats:cache_misses", 0)
        
        # Initialize performance metrics
        self.redis_client.delete("stats:latencies")  # List of recent latencies
        
        logger.info("Monitoring keys initialized")
    
    def get_info(self) -> Dict[str, Any]:
        """Get Redis server information"""
        info = self.redis_client.info()
        
        return {
            "version": info.get("redis_version"),
            "memory_used": info.get("used_memory_human"),
            "connected_clients": info.get("connected_clients"),
            "total_keys": self.redis_client.dbsize(),
            "uptime_days": info.get("uptime_in_days")
        }
    
    def run_full_setup(self):
        """Run complete Redis setup"""
        if not self.check_connection():
            sys.exit(1)
        
        logger.info("Starting Redis setup...")
        
        # Clear existing data (optional)
        # self.clear_database()
        
        # Setup indices
        self.setup_indices()
        
        # Load data
        self.load_video_metadata()
        self.load_user_profiles()
        self.load_video_features()
        
        # Create sample data
        self.create_sample_recommendations()
        
        # Setup monitoring
        self.setup_monitoring()
        
        # Print info
        info = self.get_info()
        logger.info(f"Redis setup complete!")
        logger.info(f"Server info: {json.dumps(info, indent=2)}")

def main():
    parser = argparse.ArgumentParser(description="Setup Redis for video recommendation system")
    parser.add_argument("--host", type=str, default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--db", type=int, default=0, help="Redis database")
    parser.add_argument("--clear", action="store_true", help="Clear existing data")
    
    args = parser.parse_args()
    
    setup = RedisSetup(host=args.host, port=args.port, db=args.db)
    
    if args.clear:
        setup.clear_database()
        logger.info("Database cleared")
    
    setup.run_full_setup()

if __name__ == "__main__":
    main()