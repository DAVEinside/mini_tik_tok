import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    APP_NAME: str = "Video Feed Recommender"
    
    # Model Settings
    MODEL_PATH: str = "models/transformer_recommender.pt"
    EMBEDDING_DIM: int = 768
    NUM_HEADS: int = 12
    NUM_LAYERS: int = 6
    MAX_SEQ_LENGTH: int = 100
    
    # Redis Settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = 0
    REDIS_CACHE_TTL: int = 3600  # 1 hour
    
    # Performance Settings
    BATCH_SIZE: int = 32
    TOP_K_RECOMMENDATIONS: int = 50
    HIT_RATE_THRESHOLD: float = 0.94
    P95_LATENCY_TARGET: int = 80  # ms
    
    # GPU Settings
    USE_GPU: bool = True
    GPU_DEVICE_ID: int = 0
    
    # Dataset Settings
    VIDEO_DATASET_SIZE: int = 10000
    TRAIN_TEST_SPLIT: float = 0.8
    
    class Config:
        case_sensitive = True

settings = Settings()