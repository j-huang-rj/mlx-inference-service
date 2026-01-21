"""Route handler for model listing."""

from fastapi import APIRouter

from app.config import get_settings
from app.schemas import ModelCard, ModelList

router = APIRouter()


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    """List available models."""

    settings = get_settings()

    return ModelList(
        data=[
            ModelCard(id=settings.embedding_model),
            ModelCard(id=settings.reranker_model),
        ]
    )
