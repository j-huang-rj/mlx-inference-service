"""Tests for embedding service."""

import time

import numpy as np

from app.services.embedding_service import EmbeddingService, get_embedding_service


def test_singleton_pattern():
    """Test that EmbeddingService is a singleton."""

    service1 = get_embedding_service()
    service2 = get_embedding_service()
    assert service1 is service2


def test_lazy_loading(mock_embedding_model):
    """Test that model is not loaded until first use."""

    service = EmbeddingService()
    service._model = None  # Reset
    service._initialized = False
    service.__init__()

    assert not service.is_loaded
    assert service.load_error is None

    # Trigger loading
    service.embed(["test"])
    assert service.is_loaded


def test_basic_embedding_generation(mock_embedding_model):
    """Test basic embedding generation."""

    service = get_embedding_service()
    texts = ["Hello world", "Test sentence"]

    embeddings, token_count = service.embed(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert token_count > 0

    # Check normalization
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01  # Allow small floating point error


def test_batch_processing(mock_embedding_model, mocker):
    """Test that large batches are processed in chunks."""

    service = get_embedding_service()

    # Set small batch size
    mocker.patch("app.config.get_settings").return_value.embedding_batch_size = 2

    texts = ["text1", "text2", "text3", "text4", "text5"]
    embeddings, _ = service.embed(texts)

    assert len(embeddings) == 5


def test_dimension_truncation(mock_embedding_model):
    """Test dimension truncation and re-normalization."""

    service = get_embedding_service()
    texts = ["test"]

    embeddings, _ = service.embed(texts, dimensions=512)

    assert len(embeddings[0]) == 512

    # Check re-normalization after truncation
    norm = np.linalg.norm(embeddings[0])
    assert abs(norm - 1.0) < 0.01


def test_instruction_prefix(mock_embedding_model):
    """Test instruction-aware embedding."""

    service = get_embedding_service()

    # Ensure the service is loaded first
    service.embed(["test"])

    # Now embed with instruction
    embeddings, _ = service.embed(
        ["query text"], instruction="Represent this for retrieval"
    )

    assert len(embeddings) == 1


def test_empty_input_handling(mock_embedding_model):
    """Test handling of empty inputs."""

    service = get_embedding_service()

    # Empty list should work but return empty results
    embeddings, token_count = service.embed([])
    assert embeddings == []
    assert token_count == 0


def test_idle_time_tracking(mock_embedding_model):
    """Test idle time tracking."""

    service = get_embedding_service()

    # Reset state
    service._last_used_time = None

    # Initially no idle time
    assert service.get_idle_seconds() is None

    # After use, idle time should start
    service.embed(["test"])
    time.sleep(0.1)

    idle_seconds = service.get_idle_seconds()
    assert idle_seconds is not None
    assert idle_seconds >= 0.1


def test_unload(mock_embedding_model):
    """Test model unloading."""

    service = get_embedding_service()

    # Load the model
    service.embed(["test"])
    assert service.is_loaded

    # Unload
    service.unload()
    assert not service.is_loaded
    assert service.get_idle_seconds() is None


def test_similarity_scores(mock_embedding_model):
    """Test that similar texts have high cosine similarity."""

    service = get_embedding_service()

    # Use fixed seed for reproducible mock embeddings
    np.random.seed(42)

    texts = [
        "The cat sat on the mat",
        "A feline rested on the rug",
        "Python programming language",
    ]

    embeddings, _ = service.embed(texts)

    # Calculate cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Verify the embeddings are normalized
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01
