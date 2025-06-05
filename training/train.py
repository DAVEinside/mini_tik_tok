import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import wandb
from typing import Tuple, Dict, List
import random
from app.models.transformer_recommender import VideoRecommender
from app.core.config import settings

class VideoInteractionDataset(Dataset):
    def __init__(self, interactions: List[Dict], num_videos: int, max_history: int = 100):
        self.interactions = interactions
        self.num_videos = num_videos
        self.max_history = max_history
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        
        # Get user history
        history = interaction['history'][-self.max_history:]
        target = interaction['target']
        
        # Pad history if needed
        if len(history) < self.max_history:
            history = [0] * (self.max_history - len(history)) + history
            mask = [0] * (self.max_history - len(history)) + [1] * len(interaction['history'][-self.max_history:])
        else:
            mask = [1] * self.max_history
        
        # Create negative samples
        negative_samples = []
        while len(negative_samples) < 5:  # 5 negative samples per positive
            neg_id = random.randint(0, self.num_videos - 1)
            if neg_id not in history and neg_id != target:
                negative_samples.append(neg_id)
        
        return {
            'user_id': interaction['user_id'],
            'history': torch.tensor(history, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'target': torch.tensor(target, dtype=torch.long),
            'negatives': torch.tensor(negative_samples, dtype=torch.long)
        }

def train_model(
    model: VideoRecommender,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: torch.device = torch.device('cuda')
) -> Dict[str, List[float]]:
    """Train the recommendation model"""
    
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Initialize wandb
    wandb.init(project="video-recommender", config={
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "model_type": "transformer"
    })
    
    history = {'train_loss': [], 'val_loss': [], 'val_hit_rate': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            user_ids = batch['user_id'].to(device)
            history_ids = batch['history'].to(device)
            masks = batch['mask'].to(device)
            targets = batch['target'].to(device)
            negatives = batch['negatives'].to(device)
            
            # Get recommendations
            recs, scores = model.get_recommendations(user_ids, history_ids, masks, top_k=100)
            
            # Compute loss (BPR loss)
            positive_scores = scores.gather(1, targets.unsqueeze(1))
            negative_scores = scores.gather(1, negatives)
            
            loss = -torch.log(torch.sigmoid(positive_scores - negative_scores)).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        hit_rates = []
        
        with torch.no_grad():
            for batch in val_loader:
                user_ids = batch['user_id'].to(device)
                history_ids = batch['history'].to(device)
                masks = batch['mask'].to(device)
                targets = batch['target'].to(device)
                
                # Get recommendations
                recs, scores = model.get_recommendations(user_ids, history_ids, masks, top_k=50)
                
                # Calculate hit rate@50
                hits = (recs == targets.unsqueeze(1)).any(dim=1).float()
                hit_rates.extend(hits.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        avg_hit_rate = np.mean(hit_rates)
        
        history['train_loss'].append(avg_train_loss)
        history['val_hit_rate'].append(avg_hit_rate)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Hit Rate@50 = {avg_hit_rate:.4f}")
        
        wandb.log({
            "train_loss": avg_train_loss,
            "hit_rate_50": avg_hit_rate,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        scheduler.step()
        
        # Save checkpoint if hit rate improves
        if avg_hit_rate >= settings.HIT_RATE_THRESHOLD:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'hit_rate': avg_hit_rate
            }, settings.MODEL_PATH)
            print(f"Model saved with hit rate: {avg_hit_rate:.4f}")
    
    wandb.finish()
    return history

def generate_synthetic_data(num_users: int = 1000, num_videos: int = 10000, 
                           num_interactions: int = 50000) -> Tuple[List[Dict], List[Dict]]:
    """Generate synthetic interaction data for training"""
    
    interactions = []
    
    # Generate user preferences (latent factors)
    user_factors = np.random.randn(num_users, 50)
    video_factors = np.random.randn(num_videos, 50)
    
    # Generate interactions based on latent factors
    for _ in range(num_interactions):
        user_id = random.randint(0, num_users - 1)
        
        # Get user's preference scores for all videos
        scores = np.dot(user_factors[user_id], video_factors.T)
        probs = np.exp(scores) / np.sum(np.exp(scores))
        
        # Sample videos based on preferences
        history_length = random.randint(5, 50)
        history = np.random.choice(num_videos, size=history_length, p=probs, replace=False).tolist()
        
        # Target is the next video
        target = np.random.choice(num_videos, p=probs)
        while target in history:
            target = np.random.choice(num_videos, p=probs)
        
        interactions.append({
            'user_id': user_id,
            'history': history[:-1],
            'target': target
        })
    
    # Split into train/val
    random.shuffle(interactions)
    split_idx = int(len(interactions) * settings.TRAIN_TEST_SPLIT)
    
    return interactions[:split_idx], interactions[split_idx:]

if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic data...")
    train_data, val_data = generate_synthetic_data(
        num_users=1000,
        num_videos=settings.VIDEO_DATASET_SIZE,
        num_interactions=50000
    )
    
    # Create datasets
    train_dataset = VideoInteractionDataset(train_data, settings.VIDEO_DATASET_SIZE)
    val_dataset = VideoInteractionDataset(val_data, settings.VIDEO_DATASET_SIZE)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=settings.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    model = VideoRecommender(
        num_videos=settings.VIDEO_DATASET_SIZE,
        num_users=1000,
        embedding_dim=settings.EMBEDDING_DIM,
        num_heads=settings.NUM_HEADS,
        num_layers=settings.NUM_LAYERS
    )
    
    # Train model
    device = torch.device(f"cuda:{settings.GPU_DEVICE_ID}" if settings.USE_GPU and torch.cuda.is_available() else "cpu")
    history = train_model(model, train_loader, val_loader, num_epochs=10, device=device)
    
    print("Training completed!")

### 8. Metrics Collection (app/utils/metrics.py)

```python
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