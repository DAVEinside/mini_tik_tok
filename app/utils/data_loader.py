"""Data Loading Utilities for Video Recommendation System"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import json
import os
from pathlib import Path
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class VideoDataLoader:
    """Utility class for loading and preprocessing video data"""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def load_video_metadata(self, filename: str = "video_metadata.json") -> Dict[int, Dict]:
        """Load video metadata from JSON file"""
        filepath = self.processed_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Video metadata file not found: {filepath}")
            return {}
        
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        # Convert string keys to integers
        return {int(k): v for k, v in metadata.items()}
    
    def load_user_interactions(self, filename: str = "user_interactions.csv") -> pd.DataFrame:
        """Load user interaction data"""
        filepath = self.processed_dir / filename
        
        if not filepath.exists():
            logger.warning(f"User interactions file not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        return df
    
    def load_video_features(self, filename: str = "video_features.npy") -> np.ndarray:
        """Load pre-computed video features"""
        filepath = self.processed_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Video features file not found: {filepath}")
            # Return random features for demo
            return np.random.randn(settings.VIDEO_DATASET_SIZE, 768)
        
        return np.load(filepath)
    
    def create_interaction_sequences(
        self,
        interactions_df: pd.DataFrame,
        max_sequence_length: int = 100
    ) -> List[Dict[str, Any]]:
        """Create user interaction sequences for training"""
        
        sequences = []
        
        # Group by user
        for user_id, user_data in interactions_df.groupby('user_id'):
            # Sort by timestamp
            user_data = user_data.sort_values('timestamp')
            
            video_history = user_data['video_id'].tolist()
            
            # Create sequences with sliding window
            for i in range(len(video_history) - 1):
                # Use videos up to position i as history
                history = video_history[max(0, i - max_sequence_length + 1):i + 1]
                target = video_history[i + 1]
                
                sequences.append({
                    'user_id': user_id,
                    'history': history,
                    'target': target,
                    'timestamp': user_data.iloc[i + 1]['timestamp']
                })
        
        return sequences
    
    def save_processed_data(self, data: Any, filename: str):
        """Save processed data to file"""
        filepath = self.processed_dir / filename
        
        if filename.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif filename.endswith('.npy'):
            np.save(filepath, data)
        elif filename.endswith('.csv'):
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        logger.info(f"Saved processed data to {filepath}")

class VideoRecommendationDataset(Dataset):
    """PyTorch Dataset for video recommendation"""
    
    def __init__(
        self,
        sequences: List[Dict[str, Any]],
        video_features: Optional[np.ndarray] = None,
        max_history_length: int = 100,
        negative_samples: int = 5
    ):
        self.sequences = sequences
        self.video_features = video_features
        self.max_history_length = max_history_length
        self.negative_samples = negative_samples
        self.num_videos = settings.VIDEO_DATASET_SIZE
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Get history and target
        history = sequence['history'][-self.max_history_length:]
        target = sequence['target']
        user_id = sequence['user_id']
        
        # Pad history
        if len(history) < self.max_history_length:
            padding_length = self.max_history_length - len(history)
            history = [0] * padding_length + history
            mask = [0] * padding_length + [1] * len(sequence['history'][-self.max_history_length:])
        else:
            mask = [1] * self.max_history_length
        
        # Generate negative samples
        negatives = []
        video_set = set(history + [target])
        
        while len(negatives) < self.negative_samples:
            neg_id = np.random.randint(0, self.num_videos)
            if neg_id not in video_set:
                negatives.append(neg_id)
        
        # Create output dict
        output = {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'history': torch.tensor(history, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'target': torch.tensor(target, dtype=torch.long),
            'negatives': torch.tensor(negatives, dtype=torch.long)
        }
        
        # Add video features if available
        if self.video_features is not None:
            history_features = self.video_features[history]
            target_features = self.video_features[target]
            negative_features = self.video_features[negatives]
            
            output.update({
                'history_features': torch.tensor(history_features, dtype=torch.float32),
                'target_features': torch.tensor(target_features, dtype=torch.float32),
                'negative_features': torch.tensor(negative_features, dtype=torch.float32)
            })
        
        return output

def create_data_loaders(
    data_dir: str = "data/",
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    
    # Load data
    loader = VideoDataLoader(data_dir)
    
    # Load user interactions
    interactions_df = loader.load_user_interactions()
    
    if interactions_df.empty:
        # Generate synthetic data for demo
        logger.warning("No interaction data found, generating synthetic data")
        from training.train import generate_synthetic_data
        train_data, val_data = generate_synthetic_data()
    else:
        # Create sequences
        sequences = loader.create_interaction_sequences(interactions_df)
        
        # Split data
        dataset_size = len(sequences)
        train_size = int(train_split * dataset_size)
        val_size = dataset_size - train_size
        
        train_sequences, val_sequences = random_split(sequences, [train_size, val_size])
        train_data = [sequences[i] for i in train_sequences.indices]
        val_data = [sequences[i] for i in val_sequences.indices]
    
    # Load video features
    video_features = loader.load_video_features()
    
    # Create datasets
    train_dataset = VideoRecommendationDataset(train_data, video_features)
    val_dataset = VideoRecommendationDataset(val_data, video_features)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for handling variable length sequences"""
    
    # Find max lengths in batch
    max_history_len = max(item['history'].shape[0] for item in batch)
    
    # Pad sequences
    padded_batch = {}
    
    for key in batch[0].keys():
        if key == 'history':
            # Pad history sequences
            padded_histories = []
            for item in batch:
                hist = item[key]
                if hist.shape[0] < max_history_len:
                    padding = torch.zeros(max_history_len - hist.shape[0], dtype=hist.dtype)
                    hist = torch.cat([padding, hist])
                padded_histories.append(hist)
            padded_batch[key] = torch.stack(padded_histories)
        elif key == 'mask':
            # Pad masks
            padded_masks = []
            for item in batch:
                mask = item[key]
                if mask.shape[0] < max_history_len:
                    padding = torch.zeros(max_history_len - mask.shape[0], dtype=mask.dtype)
                    mask = torch.cat([padding, mask])
                padded_masks.append(mask)
            padded_batch[key] = torch.stack(padded_masks)
        else:
            # Stack other tensors normally
            padded_batch[key] = torch.stack([item[key] for item in batch])
    
    return padded_batch