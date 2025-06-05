"""Video Encoder Module for Video Feature Extraction and Encoding"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np

class VideoEncoder(nn.Module):
    """Multimodal video encoder combining visual, audio, and text features"""
    
    def __init__(
        self,
        visual_dim: int = 2048,
        audio_dim: int = 128,
        text_dim: int = 768,
        output_dim: int = 768,
        num_frames: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Visual encoder (for frame features)
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim)
        )
        
        # Temporal attention for video frames
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )
        
        # Text encoder (for titles, descriptions)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # Multimodal fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Learnable modality tokens
        self.visual_token = nn.Parameter(torch.randn(1, 1, output_dim))
        self.audio_token = nn.Parameter(torch.randn(1, 1, output_dim))
        self.text_token = nn.Parameter(torch.randn(1, 1, output_dim))
        
    def forward(
        self,
        visual_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for video encoding
        
        Args:
            visual_features: Frame features [batch, num_frames, visual_dim]
            audio_features: Audio features [batch, audio_dim]
            text_features: Text features [batch, text_dim]
            video_mask: Mask for valid frames [batch, num_frames]
        
        Returns:
            video_embedding: Final video embedding [batch, output_dim]
            modality_embeddings: Dict of individual modality embeddings
        """
        batch_size = (visual_features.shape[0] if visual_features is not None else
                     audio_features.shape[0] if audio_features is not None else
                     text_features.shape[0])
        
        modality_embeddings = {}
        fusion_inputs = []
        
        # Process visual features
        if visual_features is not None:
            # Encode frames
            visual_encoded = self.visual_encoder(visual_features)
            
            # Apply temporal attention
            visual_attended, _ = self.temporal_attention(
                visual_encoded, visual_encoded, visual_encoded,
                key_padding_mask=video_mask if video_mask is not None else None
            )
            
            # Global average pooling
            if video_mask is not None:
                mask_expanded = video_mask.unsqueeze(-1).float()
                visual_pooled = (visual_attended * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                visual_pooled = visual_attended.mean(dim=1)
            
            modality_embeddings['visual'] = visual_pooled
            
            # Add modality token
            visual_with_token = visual_pooled + self.visual_token.squeeze(1)
            fusion_inputs.append(visual_with_token)
        
        # Process audio features
        if audio_features is not None:
            audio_encoded = self.audio_encoder(audio_features)
            modality_embeddings['audio'] = audio_encoded
            
            # Add modality token
            audio_with_token = audio_encoded + self.audio_token.squeeze(1)
            fusion_inputs.append(audio_with_token)
        
        # Process text features
        if text_features is not None:
            text_encoded = self.text_encoder(text_features)
            modality_embeddings['text'] = text_encoded
            
            # Add modality token
            text_with_token = text_encoded + self.text_token.squeeze(1)
            fusion_inputs.append(text_with_token)
        
        # Multimodal fusion
        if len(fusion_inputs) > 1:
            # Stack modalities
            fusion_input = torch.stack(fusion_inputs, dim=1)  # [batch, num_modalities, output_dim]
            
            # Apply cross-modal attention
            fused, _ = self.fusion_attention(fusion_input, fusion_input, fusion_input)
            
            # Concatenate all modalities
            fused_concat = fused.reshape(batch_size, -1)  # [batch, num_modalities * output_dim]
            
            # Final projection
            video_embedding = self.output_projection(fused_concat)
        else:
            # Single modality
            video_embedding = fusion_inputs[0] if fusion_inputs else torch.zeros(batch_size, self.output_dim)
        
        return video_embedding, modality_embeddings

class VideoMetadataEncoder(nn.Module):
    """Encode video metadata (duration, resolution, upload time, etc.)"""
    
    def __init__(self, embedding_dim: int = 768, num_categories: int = 20):
        super().__init__()
        
        # Category embedding
        self.category_embedding = nn.Embedding(num_categories, 128)
        
        # Numerical features encoder
        self.numerical_encoder = nn.Sequential(
            nn.Linear(5, 64),  # duration, fps, width, height, upload_time
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Combine all metadata
        self.metadata_encoder = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(
        self,
        categories: torch.Tensor,
        numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode video metadata
        
        Args:
            categories: Video categories [batch, num_categories]
            numerical_features: Numerical features [batch, 5]
        
        Returns:
            Metadata embedding [batch, embedding_dim]
        """
        # Encode categories (average pooling if multiple)
        cat_embeds = self.category_embedding(categories).mean(dim=1)
        
        # Encode numerical features
        num_embeds = self.numerical_encoder(numerical_features)
        
        # Combine
        combined = torch.cat([cat_embeds, num_embeds], dim=-1)
        
        return self.metadata_encoder(combined)