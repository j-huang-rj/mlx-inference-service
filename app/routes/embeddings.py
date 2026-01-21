"""OpenAI-compatible embeddings endpoint."""

import logging

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.schemas import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["embeddings"])


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Create embeddings for the given input text(s).

    OpenAI-compatible endpoint.
    """

    service = get_embedding_service()
    settings = get_settings()

    # Normalize input to list
    texts = [request.input] if isinstance(request.input, str) else request.input

    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per request")

    try:
        # Use default dimensions if not specified in request
        dimensions = request.dimensions or settings.embedding_matryoshka_dim

        # Generate embeddings
        embeddings, total_tokens = service.embed(
            texts=texts,
            dimensions=dimensions,
            instruction=request.instruction,
        )

        # Build response
        data = [
            EmbeddingData(index=i, embedding=emb) for i, emb in enumerate(embeddings)
        ]

        return EmbeddingResponse(
            model=request.model,
            data=data,
            usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )

    except RuntimeError as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e

    except Exception as e:
        logger.exception("Unexpected error in embeddings endpoint")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e
