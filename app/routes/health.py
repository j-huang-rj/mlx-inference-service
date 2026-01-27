"""Health check endpoint."""

from fastapi import APIRouter

from app.config import get_settings
from app.schemas import HealthResponse, ModelStatus
from app.services.embedding_service import get_embedding_service
from app.services.reranker_service import get_reranker_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the health status of the service and loaded models."""

    settings = get_settings()
    models = []

    if settings.embedding_enabled:
        embedding_service = get_embedding_service()
        models.append(
            ModelStatus(
                name="Qwen3-Embedding-0.6B",
                loaded=embedding_service.is_loaded,
                error=embedding_service.load_error,
            )
        )

    if settings.reranker_enabled:
        reranker_service = get_reranker_service()
        models.append(
            ModelStatus(
                name="Jina-Reranker-V3",
                loaded=reranker_service.is_loaded,
                error=reranker_service.load_error,
            )
        )

    # Determine overall status
    if not models:
        status = "not_configured"
    elif all(m.loaded for m in models) and not any(m.error for m in models):
        status = "healthy"
    elif any(m.error for m in models):
        status = "unhealthy"
    else:
        status = "standby"

    return HealthResponse(status=status, models=models)


@router.get("/", include_in_schema=False)
async def root():
    """Root endpoint with service info."""

    from app import __version__

    return {
        "service": "MLX Inference Service",
        "version": __version__,
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "rerank": "/v1/rerank",
            "health": "/health",
            "docs": "/docs",
        },
    }
