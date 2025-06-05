"""Video Recommender Application Package"""

__version__ = "1.0.0"
__author__ = "nimitdave"
__email__ = "nimitdave@gmail.com"

# Import key components for easier access
from app.core.config import settings
from app.services.recommendation_service import RecommendationService

__all__ = ["settings", "RecommendationService"]