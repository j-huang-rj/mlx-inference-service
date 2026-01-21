"""Tests for model manager."""

import time
from unittest.mock import Mock

from app.services.model_manager import ModelManager


def test_model_manager_start_stop():
    """Test starting and stopping model manager."""

    manager = ModelManager()

    manager.start()
    assert manager._thread is not None
    assert manager._thread.is_alive()

    manager.stop()
    time.sleep(0.5)
    assert not manager._thread.is_alive()


def test_idle_detection(mocker, mock_embedding_model, mock_reranker):
    """Test that idle models are detected and unloaded."""

    # Mock settings with very short timeout
    settings_mock = Mock()
    settings_mock.model_idle_timeout_seconds = 1
    settings_mock.model_unload_check_interval = 0.5
    mocker.patch("app.services.model_manager.get_settings", return_value=settings_mock)

    from app.services.embedding_service import get_embedding_service

    service = get_embedding_service()
    service._model = None  # Reset
    service._last_used_time = None

    # Load the model
    service.embed(["test"])
    assert service.is_loaded

    # Start manager
    manager = ModelManager()
    manager.start()

    try:
        # Wait for model to become idle and get unloaded
        time.sleep(2)

        # Model should be unloaded
        assert not service.is_loaded

    finally:
        manager.stop()


def test_model_reload_after_unload(mocker, mock_embedding_model):
    """Test that model can be reloaded after being unloaded."""

    from app.services.embedding_service import get_embedding_service

    service = get_embedding_service()

    # Load, use, unload
    service.embed(["test1"])
    assert service.is_loaded

    service.unload()
    assert not service.is_loaded

    # Should reload on next use
    service.embed(["test2"])
    assert service.is_loaded


def test_multiple_start_ignored():
    """Test that multiple start calls are ignored."""

    manager = ModelManager()

    manager.start()
    thread1 = manager._thread

    # Second start should be ignored
    manager.start()
    thread2 = manager._thread

    assert thread1 is thread2

    manager.stop()


def test_stop_on_not_running():
    """Test that stopping a non-running manager is safe."""

    manager = ModelManager()

    # Should not raise error
    manager.stop()


def test_graceful_shutdown(mocker, mock_embedding_model):
    """Test graceful shutdown of manager."""

    manager = ModelManager()
    manager.start()

    # Manager should stop cleanly without errors
    manager.stop()

    assert manager._thread is not None
    assert not manager._thread.is_alive()
