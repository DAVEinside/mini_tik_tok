from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from app.api.routes import router
from app.core.config import settings
from app.services.recommendation_service import RecommendationService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing recommendation service...")
    app.state.recommendation_service = RecommendationService()
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "message": "Video Feed Recommender API",
        "version": "1.0.0",
        "docs": f"{settings.API_V1_STR}/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1  # Use 1 worker for GPU
    )