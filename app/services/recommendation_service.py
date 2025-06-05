import torch
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from app.models.transformer_recommender import VideoRecommender
from app.services.cache_service import RedisCacheService
from app.core.config import settings
from app.utils.metrics import MetricsCollector

class RecommendationService:
    def __init__(self, model_path: str = settings.MODEL_PATH):
        self.device = torch.device(f"cuda:{settings.GPU_DEVICE_ID}" if settings.USE_GPU and torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize services
        self.cache_service = RedisCacheService()
        self.metrics = MetricsCollector()
        
        # Warm up GPU
        if settings.USE_GPU:
            self._warmup_gpu()
    
    def _load_model(self, model_path: str) -> VideoRecommender:
        """Load pre-trained model"""
        # For demo, we'll initialize a new model
        # In production, you'd load from checkpoint
        model = VideoRecommender(
            num_videos=settings.VIDEO_DATASET_SIZE,
            num_users=1000,  # Assuming 1000 demo users
            embedding_dim=settings.EMBEDDING_DIM,
            num_heads=settings.NUM_HEADS,
            num_layers=settings.NUM_LAYERS
        )
        
        # Load checkpoint if exists
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
        except:
            print("No checkpoint found, using random initialization")
            
        return model
    
    def _warmup_gpu(self):
        """Warm up GPU with dummy inference"""
        print("Warming up GPU...")
        with torch.no_grad():
            for _ in range(10):
                dummy_user = torch.randint(0, 1000, (1,)).to(self.device)
                dummy_history = torch.randint(0, settings.VIDEO_DATASET_SIZE, (1, 20)).to(self.device)
                self.model.get_recommendations(dummy_user, dummy_history, top_k=50)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("GPU warmup complete")
    
    def get_recommendations(
        self,
        user_id: int,
        video_history: List[int],
        top_k: int = 50,
        use_cache: bool = True
    ) -> Tuple[List[int], List[float], Dict[str, Any]]:
        """Get video recommendations for a user"""
        start_time = time.time()
        metrics = {'cache_hit': False, 'inference_time': 0, 'total_time': 0}
        
        # Check cache first
        if use_cache:
            cached_result = self.cache_service.get_recommendations(user_id, video_history)
            if cached_result:
                metrics['cache_hit'] = True
                metrics['total_time'] = (time.time() - start_time) * 1000  # ms
                
                self.metrics.record_request(
                    user_id=user_id,
                    cache_hit=True,
                    latency=metrics['total_time']
                )
                
                return (
                    cached_result['recommendations'][:top_k],
                    cached_result['scores'][:top_k],
                    metrics
                )
        
        # GPU inference
        inference_start = time.time()
        
        with torch.no_grad():
            # Prepare inputs
            user_tensor = torch.tensor([user_id], dtype=torch.long).to(self.device)
            history_tensor = torch.tensor([video_history[-settings.MAX_SEQ_LENGTH:]], dtype=torch.long).to(self.device)
            
            # Get recommendations
            video_ids, scores = self.model.get_recommendations(
                user_tensor,
                history_tensor,
                top_k=top_k
            )
            
            # Synchronize GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Convert to lists
            video_ids = video_ids[0].cpu().numpy().tolist()
            scores = scores[0].cpu().numpy().tolist()
        
        metrics['inference_time'] = (time.time() - inference_start) * 1000  # ms
        
        # Cache results
        if use_cache:
            self.cache_service.set_recommendations(user_id, video_history, video_ids, scores)
        
        metrics['total_time'] = (time.time() - start_time) * 1000  # ms
        
        self.metrics.record_request(
            user_id=user_id,
            cache_hit=False,
            latency=metrics['total_time']
        )
        
        return video_ids, scores, metrics
    
    def batch_get_recommendations(
        self,
        user_ids: List[int],
        video_histories: List[List[int]],
        top_k: int = 50
    ) -> List[Tuple[List[int], List[float]]]:
        """Batch inference for multiple users"""
        with torch.no_grad():
            # Prepare batch inputs
            user_tensor = torch.tensor(user_ids, dtype=torch.long).to(self.device)
            
            # Pad histories to same length
            max_len = max(len(h) for h in video_histories)
            padded_histories = []
            masks = []
            
            for history in video_histories:
                padded = history[-max_len:] + [0] * (max_len - len(history))
                mask = [1] * len(history[-max_len:]) + [0] * (max_len - len(history))
                padded_histories.append(padded)
                masks.append(mask)
            
            history_tensor = torch.tensor(padded_histories, dtype=torch.long).to(self.device)
            mask_tensor = torch.tensor(masks, dtype=torch.bool).to(self.device)
            
            # Get batch recommendations
            video_ids, scores = self.model.get_recommendations(
                user_tensor,
                history_tensor,
                mask_tensor,
                top_k=top_k
            )
            
            # Convert to lists
            results = []
            for i in range(len(user_ids)):
                vid_list = video_ids[i].cpu().numpy().tolist()
                score_list = scores[i].cpu().numpy().tolist()
                results.append((vid_list, score_list))
                
        return results