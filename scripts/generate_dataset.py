"""Generate synthetic dataset for video recommendation system"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import random
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetGenerator:
    """Generate synthetic video recommendation dataset"""
    
    def __init__(
        self,
        num_users: int = 1000,
        num_videos: int = 10000,
        num_creators: int = 500,
        num_categories: int = 20
    ):
        self.num_users = num_users
        self.num_videos = num_videos
        self.num_creators = num_creators
        self.num_categories = num_categories
        
        # Define categories
        self.categories = [
            "Entertainment", "Music", "Gaming", "Education", "Comedy",
            "Sports", "News", "Technology", "Fashion", "Food",
            "Travel", "Fitness", "Art", "Science", "Politics",
            "Business", "Health", "Lifestyle", "Nature", "History"
        ]
        
        # Video length distributions by category
        self.duration_dist = {
            "Music": (180, 60),  # mean, std in seconds
            "Gaming": (600, 300),
            "Education": (480, 180),
            "News": (120, 30),
            "Comedy": (180, 90),
            "default": (300, 150)
        }
    
    def generate_video_metadata(self) -> Dict[int, Dict]:
        """Generate video metadata"""
        logger.info(f"Generating metadata for {self.num_videos} videos...")
        
        metadata = {}
        
        for video_id in tqdm(range(self.num_videos)):
            # Random categories (1-3 per video)
            num_cats = random.randint(1, 3)
            video_categories = random.sample(self.categories, num_cats)
            
            # Duration based on primary category
            primary_cat = video_categories[0]
            if primary_cat in self.duration_dist:
                mean, std = self.duration_dist[primary_cat]
            else:
                mean, std = self.duration_dist["default"]
            
            duration = max(15, np.random.normal(mean, std))  # Min 15 seconds
            
            # Upload date (last 6 months)
            days_ago = random.randint(0, 180)
            upload_date = datetime.now() - timedelta(days=days_ago)
            
            metadata[video_id] = {
                "title": f"Video {video_id} - {primary_cat}",
                "duration": round(duration, 1),
                "categories": video_categories,
                "creator_id": random.randint(0, self.num_creators - 1),
                "upload_date": upload_date.strftime("%Y-%m-%d"),
                "resolution": random.choice([[1920, 1080], [1280, 720], [3840, 2160]]),
                "fps": random.choice([24, 30, 60]),
                "view_count": int(np.random.pareto(2) * 10000),
                "like_ratio": random.uniform(0.85, 0.99)
            }
        
        return metadata
    
    def generate_user_profiles(self) -> Dict[int, Dict]:
        """Generate user profiles with preferences"""
        logger.info(f"Generating {self.num_users} user profiles...")
        
        profiles = {}
        age_groups = ["13-17", "18-24", "25-34", "35-44", "45-54", "55+"]
        
        for user_id in range(self.num_users):
            # User preferences (weighted categories)
            num_interests = random.randint(2, 5)
            interests = random.sample(self.categories, num_interests)
            
            # Create preference weights
            weights = np.random.dirichlet(np.ones(num_interests) * 2)
            preferences = {cat: w for cat, w in zip(interests, weights)}
            
            profiles[user_id] = {
                "age_group": random.choice(age_groups),
                "join_date": (datetime.now() - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
                "preferences": preferences,
                "activity_level": random.choice(["low", "medium", "high"]),
                "average_session_length": random.randint(5, 60)  # minutes
            }
        
        return profiles
    
    def generate_interactions(
        self,
        video_metadata: Dict[int, Dict],
        user_profiles: Dict[int, Dict],
        num_interactions: int = 100000
    ) -> pd.DataFrame:
        """Generate user-video interactions based on preferences"""
        logger.info(f"Generating {num_interactions} user interactions...")
        
        interactions = []
        
        # Create video category index
        video_categories = {}
        for vid, meta in video_metadata.items():
            for cat in meta['categories']:
                if cat not in video_categories:
                    video_categories[cat] = []
                video_categories[cat].append(vid)
        
        for _ in tqdm(range(num_interactions)):
            # Select user
            user_id = random.randint(0, self.num_users - 1)
            profile = user_profiles[user_id]
            
            # Select video based on user preferences
            if random.random() < 0.8:  # 80% based on preferences
                # Choose category based on user preferences
                if profile['preferences']:
                    categories = list(profile['preferences'].keys())
                    weights = list(profile['preferences'].values())
                    category = np.random.choice(categories, p=weights)
                    
                    if category in video_categories:
                        video_id = random.choice(video_categories[category])
                    else:
                        video_id = random.randint(0, self.num_videos - 1)
                else:
                    video_id = random.randint(0, self.num_videos - 1)
            else:  # 20% random exploration
                video_id = random.randint(0, self.num_videos - 1)
            
            # Generate interaction
            video_meta = video_metadata[video_id]
            
            # Watch duration (influenced by video quality and user preference)
            if any(cat in profile['preferences'] for cat in video_meta['categories']):
                # Higher completion rate for preferred content
                completion_rate = random.betavariate(8, 2)
            else:
                # Lower completion rate for non-preferred
                completion_rate = random.betavariate(2, 5)
            
            watch_duration = min(video_meta['duration'], completion_rate * video_meta['duration'])
            
            # Interaction timestamp
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
            
            # Interaction type
            interaction_types = ['view']
            
            # Like probability based on watch duration
            if watch_duration / video_meta['duration'] > 0.8:
                if random.random() < 0.3:
                    interaction_types.append('like')
                if random.random() < 0.1:
                    interaction_types.append('share')
            
            # Add interactions
            for interaction_type in interaction_types:
                interactions.append({
                    'user_id': user_id,
                    'video_id': video_id,
                    'interaction_type': interaction_type,
                    'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    'watch_duration': round(watch_duration, 1) if interaction_type == 'view' else 0
                })
        
        return pd.DataFrame(interactions)
    
    def generate_video_features(self, video_metadata: Dict[int, Dict]) -> np.ndarray:
        """Generate simulated video feature embeddings"""
        logger.info("Generating video feature embeddings...")
        
        # Create category embeddings
        category_embeddings = {}
        for i, cat in enumerate(self.categories):
            # Each category has a distinctive embedding direction
            embedding = np.random.randn(768)
            embedding[i * 38:(i + 1) * 38] += 2.0  # Strengthen specific dimensions
            category_embeddings[cat] = embedding / np.linalg.norm(embedding)
        
        # Generate video embeddings
        features = np.zeros((self.num_videos, 768))
        
        for video_id, meta in tqdm(video_metadata.items()):
            # Combine category embeddings
            video_embedding = np.zeros(768)
            for cat in meta['categories']:
                video_embedding += category_embeddings[cat]
            
            # Add random variation
            video_embedding += np.random.randn(768) * 0.3
            
            # Add creator style
            creator_embedding = np.random.RandomState(meta['creator_id']).randn(768) * 0.2
            video_embedding += creator_embedding
            
            # Normalize
            features[video_id] = video_embedding / np.linalg.norm(video_embedding)
        
        return features
    
    def save_dataset(self, output_dir: str = "data/processed/"):
        """Generate and save complete dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate data
        video_metadata = self.generate_video_metadata()
        user_profiles = self.generate_user_profiles()
        interactions = self.generate_interactions(video_metadata, user_profiles)
        video_features = self.generate_video_features(video_metadata)
        
        # Save video metadata
        with open(output_path / "video_metadata.json", 'w') as f:
            json.dump(video_metadata, f, indent=2)
        logger.info(f"Saved video metadata: {len(video_metadata)} videos")
        
        # Save user profiles
        with open(output_path / "user_profiles.json", 'w') as f:
            json.dump(user_profiles, f, indent=2)
        logger.info(f"Saved user profiles: {len(user_profiles)} users")
        
        # Save interactions
        interactions.to_csv(output_path / "user_interactions.csv", index=False)
        logger.info(f"Saved interactions: {len(interactions)} records")
        
        # Save video features
        np.save(output_path / "video_features.npy", video_features)
        logger.info(f"Saved video features: {video_features.shape}")
        
        # Save category mappings
        category_mapping = {cat: i for i, cat in enumerate(self.categories)}
        with open(output_path / "category_mappings.json", 'w') as f:
            json.dump(category_mapping, f, indent=2)
        
        # Generate summary statistics
        stats = {
            "num_videos": self.num_videos,
            "num_users": self.num_users,
            "num_interactions": len(interactions),
            "num_categories": len(self.categories),
            "avg_interactions_per_user": len(interactions) / self.num_users,
            "unique_user_video_pairs": len(interactions[['user_id', 'video_id']].drop_duplicates())
        }
        
        with open(output_path / "dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Dataset generation complete!")
        logger.info(f"Statistics: {stats}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic video recommendation dataset")
    parser.add_argument("--num-users", type=int, default=1000, help="Number of users")
    parser.add_argument("--num-videos", type=int, default=10000, help="Number of videos")
    parser.add_argument("--num-interactions", type=int, default=100000, help="Number of interactions")
    parser.add_argument("--output-dir", type=str, default="data/processed/", help="Output directory")
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(
        num_users=args.num_users,
        num_videos=args.num_videos
    )
    
    generator.save_dataset(args.output_dir)

if __name__ == "__main__":
    main()