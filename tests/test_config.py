"""Tests for configuration and CLI overrides."""

import os

import pytest

from app.config import apply_cli_overrides, clear_settings_cache, get_settings


@pytest.fixture(autouse=True)
def clean_env():
    """Clean up environment variables before and after each test."""

    # Store original values
    original_env = {}
    prefixed_keys = [k for k in os.environ if k.startswith("INFERENCE_")]
    for key in prefixed_keys:
        original_env[key] = os.environ.get(key)

    # Clear the settings cache
    clear_settings_cache()

    yield

    # Restore original values
    for key in prefixed_keys:
        if key in original_env and original_env[key] is not None:
            os.environ[key] = original_env[key]
        elif key in os.environ:
            del os.environ[key]

    # Remove any new prefixed vars
    for key in list(os.environ.keys()):
        if key.startswith("INFERENCE_") and key not in original_env:
            del os.environ[key]

    clear_settings_cache()


def test_apply_cli_overrides_port():
    """Test that CLI port override works."""

    apply_cli_overrides({"port": 51435})
    clear_settings_cache()

    settings = get_settings()
    assert settings.port == 51435


def test_apply_cli_overrides_host():
    """Test that CLI host override works."""

    apply_cli_overrides({"host": "0.0.0.0"})
    clear_settings_cache()

    settings = get_settings()
    assert settings.host == "0.0.0.0"


def test_apply_cli_overrides_batch_sizes():
    """Test that CLI batch size overrides work."""

    apply_cli_overrides(
        {
            "embedding_batch_size": 256,
            "reranker_batch_size": 128,
        }
    )
    clear_settings_cache()

    settings = get_settings()
    assert settings.embedding_batch_size == 256
    assert settings.reranker_batch_size == 128


def test_apply_cli_overrides_boolean():
    """Test that boolean CLI overrides work."""

    apply_cli_overrides({"lazy_load": False})
    clear_settings_cache()

    settings = get_settings()
    assert settings.lazy_load is False


def test_apply_cli_overrides_ignores_none():
    """Test that None values are ignored."""

    original_port = get_settings().port

    apply_cli_overrides({"port": None, "host": "0.0.0.0"})
    clear_settings_cache()

    settings = get_settings()
    assert settings.port == original_port
    assert settings.host == "0.0.0.0"


def test_apply_cli_overrides_timeout():
    """Test that idle timeout override works."""

    apply_cli_overrides({"model_idle_timeout_seconds": 300})
    clear_settings_cache()

    settings = get_settings()
    assert settings.model_idle_timeout_seconds == 300


def test_env_prefix_correct():
    """Test that environment variables use INFERENCE_ prefix."""

    apply_cli_overrides({"port": 51436})

    assert os.environ.get("INFERENCE_PORT") == "51436"


def test_multiple_overrides():
    """Test applying multiple overrides at once."""

    apply_cli_overrides(
        {
            "host": "127.0.0.1",
            "port": 51437,
            "embedding_batch_size": 64,
            "model_idle_timeout_seconds": 600,
        }
    )
    clear_settings_cache()

    settings = get_settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 51437
    assert settings.embedding_batch_size == 64
    assert settings.model_idle_timeout_seconds == 600
