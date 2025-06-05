"""User Encoder Module for User Representation Learning"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import numpy as np

class UserProfileEncoder(nn.Module):
    """Enhanced user encoder with profile features"""
    
    def __init__(
        self,
        num_users: int,
        embedding_dim: int = 768,
        num_categories: int = 20,
        num_age_groups: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # User ID embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # Demographic embeddings
        self.category_embedding = nn.Embedding(num_categories, 64)
        self.age_embedding = nn.Embedding(num_age_groups, 32)
        
        # Profile encoder
        self.profile_encoder = nn.Sequential(
            nn.Linear(embedding_dim + 64 + 32, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Temporal encoding for user activity patterns
        self.temporal_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(
        self,
        user_ids: torch.Tensor,
        user_categories: Optional[torch.Tensor] = None,
        user_ages: Optional[torch.Tensor] = None,
        activity_sequence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for user encoding
        
        Args:
            user_ids: User IDs [batch_size]
            user_categories: User interest categories [batch_size, num_interests]
            user_ages: User age groups [batch_size]
            activity_sequence: Temporal activity patterns [batch_size, seq_len, feature_dim]
        
        Returns:
            User embeddings [batch_size, embedding_dim]
        """
        # Base user embedding
        user_embeds = self.user_embedding(user_ids)
        
        # Add demographic information if available
        if user_categories is not None and user_ages is not None:
            # Average category embeddings
            cat_embeds = self.category_embedding(user_categories).mean(dim=1)
            age_embeds = self.age_embedding(user_ages)
            
            # Combine all features
            combined = torch.cat([user_embeds, cat_embeds, age_embeds], dim=-1)
            user_embeds = self.profile_encoder(combined)
        
        # Process temporal patterns if available
        if activity_sequence is not None:
            temporal_out, _ = self.temporal_encoder(activity_sequence)
            temporal_features = temporal_out.mean(dim=1)  # Average pooling
            user_embeds = user_embeds + temporal_features
        
        return self.output_projection(user_embeds)

class UserInteractionEncoder(nn.Module):
    """Encode user interaction patterns"""
    
    def __init__(self, embedding_dim: int = 768, num_interaction_types: int = 5):
        super().__init__()
        
        # Interaction type embeddings (like, share, comment, watch_time, skip)
        self.interaction_embedding = nn.Embedding(num_interaction_types, embedding_dim)
        
        # Multi-head attention for interaction patterns
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, interaction_history: torch.Tensor, interaction_types: torch.Tensor) -> torch.Tensor:
        """
        Encode user interaction patterns
        
        Args:
            interaction_history: Video embeddings of interacted videos [batch, seq_len, embed_dim]
            interaction_types: Types of interactions [batch, seq_len]
        
        Returns:
            Interaction encoding [batch, embed_dim]
        """
        # Get interaction type embeddings
        type_embeds = self.interaction_embedding(interaction_types)
        
        # Combine with video embeddings
        combined = interaction_history + type_embeds
        
        # Self-attention over interactions
        attended, _ = self.attention(combined, combined, combined)
        output = self.norm(attended + combined)