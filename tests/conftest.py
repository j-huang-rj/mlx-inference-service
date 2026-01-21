"""Pytest configuration and fixtures."""

import os
from collections.abc import Generator
from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_env():
    """Set test environment variables."""

    # Use shorter timeout for tests
    os.environ["INFERENCE_MODEL_IDLE_TIMEOUT_SECONDS"] = "5"
    os.environ["INFERENCE_MODEL_UNLOAD_CHECK_INTERVAL"] = "1"

    yield

    # Cleanup
    os.environ.pop("INFERENCE_MODEL_IDLE_TIMEOUT_SECONDS", None)
    os.environ.pop("INFERENCE_MODEL_UNLOAD_CHECK_INTERVAL", None)


@pytest.fixture
def mock_embedding_model(mocker):
    """Mock mlx-embeddings load function."""

    import mlx.core as mx
    import numpy as np

    from app.services.embedding_service import EmbeddingService

    # Reset singleton before test
    EmbeddingService._instance = None

    # Create a mock model that returns hidden states
    mock_model = MagicMock()

    def mock_forward(input_ids):
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]

        # Return random hidden states
        hidden_states = np.random.randn(batch_size, seq_length, 4096).astype(np.float32)

        # Create a simple object with last_hidden_state attribute
        output = type("ModelOutput", (), {})()
        output.last_hidden_state = mx.array(hidden_states)

        return output

    mock_model.side_effect = mock_forward

    # Create a mock tokenizer
    mock_tokenizer = MagicMock()

    def mock_tokenize(texts, **_):
        batch_size = len(texts)

        # Fixed for simplicity
        seq_length = 128

        # Return mock tokenizer output with explicit int64 dtype
        return {
            "input_ids": np.random.randint(0, 1000, (batch_size, seq_length)).astype(
                np.int64
            ),
            "attention_mask": np.ones((batch_size, seq_length), dtype=np.int64),
        }

    mock_tokenizer._tokenizer = mock_tokenize

    # Patch the load function to return (model, tokenizer)
    mocker.patch(
        "mlx_embeddings.load",
        return_value=(mock_model, mock_tokenizer),
    )

    yield mock_model

    # Cleanup
    EmbeddingService._instance = None


@pytest.fixture
def mock_reranker(mocker):
    """Mock jina reranker MLXReranker."""

    from app.services.reranker_service import RerankerService

    # Reset singleton before test
    RerankerService._instance = None

    mock_reranker_instance = MagicMock()

    def mock_rerank(query, documents, top_n=None, **_):
        # Return mock rerank results with descending scores
        results = []

        for i, doc in enumerate(documents):
            results.append(
                {
                    "index": i,
                    "relevance_score": 1.0 - (i * 0.1),
                    "document": doc,
                }
            )

        # Sort by score descending
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        if top_n:
            results = results[:top_n]

        return results

    mock_reranker_instance.rerank = mock_rerank

    # Mock snapshot_download to return a fake path
    mocker.patch(
        "app.services.reranker_service.snapshot_download",
        return_value="/fake/model/path",
    )

    # Create a fake module for rerank.MLXReranker
    import sys
    from types import ModuleType

    fake_rerank_module = ModuleType("rerank")
    fake_rerank_module.MLXReranker = Mock(return_value=mock_reranker_instance)  # type: ignore[attr-defined]
    sys.modules["rerank"] = fake_rerank_module

    yield mock_reranker_instance

    # Cleanup
    RerankerService._instance = None
    if "rerank" in sys.modules:
        del sys.modules["rerank"]


@pytest.fixture
def test_client(
    mock_embedding_model, mock_reranker
) -> Generator[TestClient, None, None]:
    """FastAPI test client with mocked models."""

    # Import after mocking to ensure patches apply
    from app.main import create_app

    app = create_app()

    with TestClient(app) as client:
        yield client
