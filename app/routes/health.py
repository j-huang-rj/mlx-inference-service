"""Health check endpoint."""

from fastapi import APIRouter

from app.schemas import HealthResponse, ModelStatus
from app.services.embedding_service import get_embedding_service
from app.services.reranker_service import get_reranker_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the health status of the service and loaded models."""

    embedding_service = get_embedding_service()
    reranker_service = get_reranker_service()

    models = [
        ModelStatus(
            name="qwen3-embedding-0.6b",
            loaded=embedding_service.is_loaded,
            error=embedding_service.load_error,
        ),
        ModelStatus(
            name="jina-reranker-v3",
            loaded=reranker_service.is_loaded,
            error=reranker_service.load_error,
        ),
    ]

    # Determine overall status
    all_loaded = all(m.loaded for m in models)
    any_error = any(m.error for m in models)

    if all_loaded and not any_error:
        status = "healthy"
    elif any_error:
        status = "unhealthy"
    else:
        status = "standby"

    return HealthResponse(status=status, models=models)


@router.get("/", include_in_schema=False)
async def root():
    """Root endpoint with service info."""

    return {
        "service": "MLX Inference Service",
        "version": "0.1.0",
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "rerank": "/v1/rerank",
            "health": "/health",
            "docs": "/docs",
        },
    }
