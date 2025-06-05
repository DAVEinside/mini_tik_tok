import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class VideoTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_length: int = 100
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Video feature embedding
        self.video_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, video_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = video_ids.shape
        
        # Generate position indices
        positions = torch.arange(seq_len, device=video_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        video_embeds = self.video_embedding(video_ids)
        position_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        embeddings = self.dropout(video_embeds + position_embeds)
        
        # Transformer encoding
        if mask is not None:
            # Convert padding mask to attention mask
            attention_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            encoded = self.transformer_encoder(embeddings, src_key_padding_mask=attention_mask)
        else:
            encoded = self.transformer_encoder(embeddings)
        
        # Apply layer norm
        encoded = self.layer_norm(encoded)
        
        # Mean pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand(encoded.size()).float()
            sum_embeddings = torch.sum(encoded * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        else:
            return encoded.mean(dim=1)

class UserEncoder(nn.Module):
    def __init__(self, num_users: int, embedding_dim: int = 768):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.user_embedding(user_ids)
        return self.layer_norm(embeddings)

class VideoRecommender(nn.Module):
    def __init__(
        self,
        num_videos: int,
        num_users: int,
        embedding_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_videos = num_videos
        self.video_encoder = VideoTransformerEncoder(
            vocab_size=num_videos,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.user_encoder = UserEncoder(num_users, embedding_dim)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        # Video embeddings for efficient retrieval
        self.video_embeddings = nn.Parameter(torch.randn(num_videos, embedding_dim))
        
    def forward(self, user_ids: torch.Tensor, video_history: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encode user
        user_embeds = self.user_encoder(user_ids)
        
        # Encode video history
        video_embeds = self.video_encoder(video_history, mask)
        
        # Combine embeddings
        combined = torch.cat([user_embeds, video_embeds], dim=-1)
        
        # Generate preference score
        preference_vector = self.fusion(combined).squeeze(-1)
        
        return preference_vector
    
    def get_recommendations(self, user_ids: torch.Tensor, video_history: torch.Tensor, 
                           mask: Optional[torch.Tensor] = None, top_k: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k video recommendations for users"""
        with torch.no_grad():
            # Get user preferences
            user_embeds = self.user_encoder(user_ids)
            video_embeds = self.video_encoder(video_history, mask)
            
            # Compute similarity scores with all videos
            scores = torch.matmul(video_embeds, self.video_embeddings.T)
            
            # Add user preference bias
            user_scores = torch.matmul(user_embeds, self.video_embeddings.T)
            scores = scores + 0.5 * user_scores
            
            # Get top-k videos
            top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
            
            return top_indices, top_scores