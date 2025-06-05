from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID")
    video_history: List[int] = Field(..., description="List of video IDs the user has watched")
    top_k: Optional[int] = Field(50, description="Number of recommendations to return")
    use_cache: bool = Field(True, description="Whether to use Redis cache")

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[int] = Field(..., description="List of recommended video IDs")
    scores: List[float] = Field(..., description="Confidence scores for recommendations")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")

class BatchRecommendationRequest(BaseModel):
    user_ids: List[int]
    video_histories: List[List[int]]
    top_k: Optional[int] = Field(50)

class BatchRecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]

class CacheStatsResponse(BaseModel):
    used_memory: str
    total_keys: int
    hit_rate: float
    connected_clients: int