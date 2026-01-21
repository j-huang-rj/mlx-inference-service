"""Pydantic schemas for OpenAI-compatible API endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


# =============================================================================
# Embedding Schemas
# =============================================================================
class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    model: str = Field(default="qwen3-embedding-8b", description="Model identifier")
    input: str | list[str] = Field(..., description="Text(s) to embed")
    encoding_format: Literal["float", "base64"] = Field(
        default="float", description="Encoding format for embeddings"
    )
    dimensions: int | None = Field(
        default=None,
        ge=32,
        le=4096,
        description="Output dimensions (32-4096 for 8B model)",
    )
    instruction: str | None = Field(
        default=None,
        description="Optional instruction prefix for instruction-aware models",
    )


class EmbeddingData(BaseModel):
    """Single embedding result."""

    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingUsage(BaseModel):
    """Token usage for embedding request."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""

    object: Literal["list"] = "list"
    model: str
    data: list[EmbeddingData]
    usage: EmbeddingUsage


# =============================================================================
# Rerank Schemas
# =============================================================================
class RerankRequest(BaseModel):
    """Rerank request following Cohere/Jina convention."""

    model: str = Field(default="jina-reranker-v3", description="Model identifier")
    query: str = Field(..., description="Search query")
    documents: list[str] = Field(..., min_length=1, description="Documents to rerank")
    top_n: int | None = Field(
        default=None, ge=1, description="Return only top N results"
    )
    return_documents: bool = Field(
        default=True, description="Include document text in response"
    )


class RerankResult(BaseModel):
    """Single rerank result."""

    index: int = Field(..., description="Original document index")
    relevance_score: float = Field(..., description="Relevance score (higher=better)")
    document: str | None = Field(default=None, description="Document text if requested")


class RerankResponse(BaseModel):
    """Rerank response."""

    model: str
    results: list[RerankResult]


# =============================================================================
# Health Check Schemas
# =============================================================================
class ModelStatus(BaseModel):
    """Status of a loaded model."""

    name: str
    loaded: bool
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "standby", "unhealthy"]
    models: list[ModelStatus]


# =============================================================================
# Model Listing Schemas
# =============================================================================
class ModelCard(BaseModel):
    """Information about a single model."""

    id: str = Field(..., description="Model identifier")
    object: Literal["model"] = "model"
    created: int = Field(default=0, description="Creation timestamp")
    owned_by: str = Field(default="system", description="Owner organization")


class ModelList(BaseModel):
    """List of available models."""

    object: Literal["list"] = "list"
    data: list[ModelCard]
