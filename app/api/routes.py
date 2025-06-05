from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from app.api.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    BatchRecommendationRequest,
    BatchRecommendationResponse,
    CacheStatsResponse
)
from app.services.recommendation_service import RecommendationService
from app.core.config import settings

router = APIRouter()
recommendation_service = RecommendationService()

@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get personalized video recommendations for a user"""
    try:
        video_ids, scores, metrics = recommendation_service.get_recommendations(
            user_id=request.user_id,
            video_history=request.video_history,
            top_k=request.top_k or settings.TOP_K_RECOMMENDATIONS,
            use_cache=request.use_cache
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=video_ids,
            scores=scores,
            metrics=metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch_recommend", response_model=BatchRecommendationResponse)
async def batch_get_recommendations(request: BatchRecommendationRequest):
    """Get recommendations for multiple users"""
    try:
        results = recommendation_service.batch_get_recommendations(
            user_ids=request.user_ids,
            video_histories=request.video_histories,
            top_k=request.top_k or settings.TOP_K_RECOMMENDATIONS
        )
        
        recommendations = []
        for i, (video_ids, scores) in enumerate(results):
            recommendations.append({
                "user_id": request.user_ids[i],
                "recommendations": video_ids,
                "scores": scores
            })
        
        return BatchRecommendationResponse(recommendations=recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get Redis cache statistics"""
    try:
        stats = recommendation_service.cache_service.get_cache_stats()
        return CacheStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/invalidate/{user_id}")
async def invalidate_user_cache(user_id: int):
    """Invalidate cache for a specific user"""
    try:
        recommendation_service.cache_service.invalidate_user_cache(user_id)
        return {"message": f"Cache invalidated for user {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        metrics = recommendation_service.metrics.get_summary()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_device": str(recommendation_service.device),
        "cache_connected": recommendation_service.cache_service.redis_client.ping()
    }