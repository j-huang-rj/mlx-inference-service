"""Integration test for idle model unloading."""

import time
from unittest.mock import Mock


def test_idle_unloading_integration(mocker, mock_embedding_model):
    """Test that models are unloaded after idle timeout in real scenario."""

    # Mock settings with short timeout
    settings_mock = Mock()
    settings_mock.model_idle_timeout_seconds = 2
    settings_mock.model_unload_check_interval = 0.5
    settings_mock.embedding_model = "Qwen/Qwen3-Embedding-0.6B"
    settings_mock.embedding_batch_size = 16

    mocker.patch("app.services.model_manager.get_settings", return_value=settings_mock)
    mocker.patch(
        "app.services.embedding_service.get_settings", return_value=settings_mock
    )

    from app.services.embedding_service import get_embedding_service
    from app.services.model_manager import ModelManager

    service = get_embedding_service()
    service._model = None  # Reset

    # Start model manager
    manager = ModelManager()
    manager.start()

    try:
        # Use the service
        embeddings, _ = service.embed(["test text"])
        assert service.is_loaded
        assert len(embeddings) == 1

        # Wait less than timeout - model should stay loaded
        time.sleep(1)
        assert service.is_loaded

        # Wait for timeout to expire
        time.sleep(2)

        # Model should be unloaded now
        assert not service.is_loaded

        # Use service again - should reload
        embeddings2, _ = service.embed(["another test"])
        assert service.is_loaded
        assert len(embeddings2) == 1

    finally:
        manager.stop()


def test_no_unload_during_active_use(mocker, mock_embedding_model):
    """Test that models are not unloaded during active use."""

    # Mock settings with very short timeout
    settings_mock = Mock()
    settings_mock.model_idle_timeout_seconds = 1
    settings_mock.model_unload_check_interval = 0.3
    settings_mock.embedding_model = "Qwen/Qwen3-Embedding-0.6B"
    settings_mock.embedding_batch_size = 16
    mocker.patch("app.services.model_manager.get_settings", return_value=settings_mock)
    mocker.patch(
        "app.services.embedding_service.get_settings", return_value=settings_mock
    )

    from app.services.embedding_service import get_embedding_service
    from app.services.model_manager import ModelManager

    service = get_embedding_service()
    service._model = None  # Reset

    manager = ModelManager()
    manager.start()

    try:
        # Use the service initially
        service.embed(["test"])
        assert service.is_loaded

        # Keep using the service periodically
        for _ in range(5):
            time.sleep(0.5)
            service.embed(["keep alive"])
            assert service.is_loaded

    finally:
        manager.stop()


def test_multiple_services_idle_tracking(mocker, mock_embedding_model, mock_reranker):
    """Test idle tracking works for multiple services independently."""

    # Mock settings
    settings_mock = Mock()
    settings_mock.model_idle_timeout_seconds = 1
    settings_mock.model_unload_check_interval = 0.3
    settings_mock.embedding_model = "Qwen/Qwen3-Embedding-0.6B"
    settings_mock.embedding_batch_size = 16
    settings_mock.reranker_model = "jinaai/jina-reranker-v3-mlx"
    settings_mock.reranker_batch_size = 16
    mocker.patch("app.services.model_manager.get_settings", return_value=settings_mock)
    mocker.patch(
        "app.services.embedding_service.get_settings", return_value=settings_mock
    )
    mocker.patch(
        "app.services.reranker_service.get_settings", return_value=settings_mock
    )

    from app.services.embedding_service import get_embedding_service
    from app.services.model_manager import ModelManager
    from app.services.reranker_service import get_reranker_service

    embedding_service = get_embedding_service()
    reranker_service = get_reranker_service()

    # Reset both services
    embedding_service._model = None
    reranker_service._reranker = None

    manager = ModelManager()
    manager.start()

    try:
        # Use embedding service
        embedding_service.embed(["test"])
        assert embedding_service.is_loaded

        # Wait a bit, then use reranker
        time.sleep(0.5)
        reranker_service.rerank("query", ["doc1", "doc2"])
        assert reranker_service.is_loaded

        # Wait for embedding to time out, but not reranker
        time.sleep(1.5)

        # Embedding should be unloaded, reranker may or may not be depending on timing
        assert not embedding_service.is_loaded

        # Verify reranker eventually unloads
        time.sleep(1.5)
        assert not reranker_service.is_loaded

    finally:
        manager.stop()
