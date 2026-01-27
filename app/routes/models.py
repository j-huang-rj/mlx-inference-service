"""Route handler for model listing."""

from fastapi import APIRouter

from app.config import get_settings
from app.schemas import ModelCard, ModelList

router = APIRouter()


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    """List available models."""

    settings = get_settings()
    models = []

    if settings.embedding_enabled:
        models.append(ModelCard(id=settings.embedding_model))

    if settings.reranker_enabled:
        models.append(ModelCard(id=settings.reranker_model))

    return ModelList(data=models)
