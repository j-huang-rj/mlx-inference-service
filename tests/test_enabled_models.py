"""Tests for the --enabled_models configuration."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.routes import embeddings, health, models, rerank
from app.services.embedding_service import EmbeddingService
from app.services.reranker_service import RerankerService


def test_settings_properties():
    """Test embedding_enabled and reranker_enabled properties."""

    # default="all"
    s = Settings(enabled_models="all")
    assert s.embedding_enabled
    assert s.reranker_enabled

    # "embedding"
    s = Settings(enabled_models="embedding")
    assert s.embedding_enabled
    assert not s.reranker_enabled

    # "reranker"
    s = Settings(enabled_models="reranker")
    assert not s.embedding_enabled
    assert s.reranker_enabled

    # invalid
    s = Settings(enabled_models="none")
    assert not s.embedding_enabled
    assert not s.reranker_enabled


@pytest.fixture
def mock_services(mocker):
    """Mock the services so we don't load real models."""

    mock_emb_svc = mocker.Mock(spec=EmbeddingService)
    mock_emb_svc.is_loaded = True
    mock_emb_svc.load_error = None
    mock_emb_svc.embed.return_value = ([[0.1, 0.2]], 10)

    mock_rerank_svc = mocker.Mock(spec=RerankerService)
    mock_rerank_svc.is_loaded = True
    mock_rerank_svc.load_error = None
    mock_rerank_svc.rerank.return_value = [
        {"index": 0, "relevance_score": 0.9, "document": "doc"}
    ]

    mocker.patch(
        "app.routes.embeddings.get_embedding_service", return_value=mock_emb_svc
    )
    mocker.patch("app.routes.rerank.get_reranker_service", return_value=mock_rerank_svc)
    mocker.patch("app.routes.health.get_embedding_service", return_value=mock_emb_svc)
    mocker.patch("app.routes.health.get_reranker_service", return_value=mock_rerank_svc)

    return mock_emb_svc, mock_rerank_svc


@pytest.fixture
def test_app():
    """Create a fresh app instance for testing."""

    app = FastAPI()
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(embeddings.router)
    app.include_router(rerank.router)

    return app


def test_api_embedding_only(mocker, test_app, mock_services):
    """Test API behavior when only embedding model is enabled."""

    # Mock settings
    settings = Settings(enabled_models="embedding")
    mocker.patch("app.routes.embeddings.get_settings", return_value=settings)
    mocker.patch("app.routes.rerank.get_settings", return_value=settings)
    mocker.patch("app.routes.health.get_settings", return_value=settings)
    mocker.patch("app.routes.models.get_settings", return_value=settings)

    client = TestClient(test_app)

    # Embeddings should work
    resp = client.post("/v1/embeddings", json={"input": "test"})
    assert resp.status_code == 200

    # Rerank should 404
    resp = client.post("/v1/rerank", json={"query": "q", "documents": ["d"]})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Reranker model is not enabled"

    # Health should only show embedding
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) == 1
    assert data["models"][0]["name"] == "qwen3-embedding-0.6b"

    # Models list should only show embedding
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == settings.embedding_model


def test_api_reranker_only(mocker, test_app, mock_services):
    """Test API behavior when only reranker model is enabled."""

    # Mock settings
    settings = Settings(enabled_models="reranker")
    mocker.patch("app.routes.embeddings.get_settings", return_value=settings)
    mocker.patch("app.routes.rerank.get_settings", return_value=settings)
    mocker.patch("app.routes.health.get_settings", return_value=settings)
    mocker.patch("app.routes.models.get_settings", return_value=settings)

    client = TestClient(test_app)

    # Embeddings should 404
    resp = client.post("/v1/embeddings", json={"input": "test"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Embedding model is not enabled"

    # Rerank should work
    resp = client.post("/v1/rerank", json={"query": "q", "documents": ["d"]})
    assert resp.status_code == 200

    # Health should only show reranker
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) == 1
    assert data["models"][0]["name"] == "jina-reranker-v3"

    # Models list should only show reranker
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == settings.reranker_model


def test_api_all_enabled(mocker, test_app, mock_services):
    """Test API behavior when all models are enabled."""

    # Mock settings
    settings = Settings(enabled_models="all")
    mocker.patch("app.routes.embeddings.get_settings", return_value=settings)
    mocker.patch("app.routes.rerank.get_settings", return_value=settings)
    mocker.patch("app.routes.health.get_settings", return_value=settings)
    mocker.patch("app.routes.models.get_settings", return_value=settings)

    client = TestClient(test_app)

    # Embeddings should work
    resp = client.post("/v1/embeddings", json={"input": "test"})
    assert resp.status_code == 200

    # Rerank should work
    resp = client.post("/v1/rerank", json={"query": "q", "documents": ["d"]})
    assert resp.status_code == 200

    # Health should show both
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) == 2

    # Models list should show both
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 2


def test_api_none_enabled(mocker, test_app, mock_services):
    """Test API behavior when no models are enabled."""

    # Mock settings with weird value
    settings = Settings(enabled_models="none")
    mocker.patch("app.routes.embeddings.get_settings", return_value=settings)
    mocker.patch("app.routes.rerank.get_settings", return_value=settings)
    mocker.patch("app.routes.health.get_settings", return_value=settings)
    mocker.patch("app.routes.models.get_settings", return_value=settings)

    client = TestClient(test_app)

    # Embeddings should 404
    resp = client.post("/v1/embeddings", json={"input": "test"})
    assert resp.status_code == 404

    # Rerank should 404
    resp = client.post("/v1/rerank", json={"query": "q", "documents": ["d"]})
    assert resp.status_code == 404

    # Health should show no models
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) == 0
    assert data["status"] == "not_configured"

    # Models list should show nothing
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 0
