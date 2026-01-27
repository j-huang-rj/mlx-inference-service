"""Rerank endpoint following Cohere/Jina convention."""

import logging

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.schemas import RerankRequest, RerankResponse, RerankResult
from app.services.reranker_service import get_reranker_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rerank"])


@router.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest) -> RerankResponse:
    """
    Rerank documents by relevance to a query.

    Follows Cohere/Jina rerank API convention.
    """

    settings = get_settings()

    if not settings.reranker_enabled:
        raise HTTPException(status_code=404, detail="Reranker model is not enabled")

    service = get_reranker_service()

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(request.documents) > 1000:
        raise HTTPException(
            status_code=400, detail="Maximum 1000 documents per request"
        )

    try:
        # Rerank documents
        results = service.rerank(
            query=request.query,
            documents=request.documents,
            top_n=request.top_n,
        )

        # Build response
        rerank_results = [
            RerankResult(
                index=r["index"],
                relevance_score=r["relevance_score"],
                document=r["document"] if request.return_documents else None,
            )
            for r in results
        ]

        return RerankResponse(model=request.model, results=rerank_results)

    except RuntimeError as e:
        logger.error(f"Rerank error: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e

    except Exception as e:
        logger.exception("Unexpected error in rerank endpoint")
        raise HTTPException(status_code=500, detail="Internal server error") from e
