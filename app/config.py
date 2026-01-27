"""Configuration settings for the inference service."""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Server settings
    host: str = "localhost"
    port: int = 11435

    # Model selection: "all", "embedding", or "reranker"
    enabled_models: str = "all"

    # Resource management
    lazy_load: bool = True
    model_idle_timeout_seconds: int = 900
    model_unload_check_interval: int = 60

    # Embedding model settings
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_batch_size: int = 32
    embedding_matryoshka_dim: int = 1024

    # Reranker model settings
    reranker_model: str = "jinaai/jina-reranker-v3-mlx"
    reranker_batch_size: int = 16

    model_config = {
        "env_prefix": "INFERENCE_",
        "env_file": ".env",
        "extra": "ignore",
    }

    @property
    def embedding_enabled(self) -> bool:
        """Check if embedding model is enabled."""

        return self.enabled_models in ("all", "embedding")

    @property
    def reranker_enabled(self) -> bool:
        """Check if reranker model is enabled."""

        return self.enabled_models in ("all", "reranker")


def apply_cli_overrides(args: dict) -> None:
    """Apply CLI argument overrides to environment variables.

    CLI args take precedence over .env values.

    Args:
        args: Dictionary of CLI argument names to values
    """

    env_prefix = "INFERENCE_"

    for arg_name, value in args.items():
        if value is None:
            continue

        # Convert arg name directly to env var name
        env_name = f"{env_prefix}{arg_name.upper()}"

        # Convert value to string for env var
        if isinstance(value, bool):
            os.environ[env_name] = "true" if value else "false"
        else:
            os.environ[env_name] = str(value)


def clear_settings_cache() -> None:
    """Clear the cached settings instance to pick up new values."""
    get_settings.cache_clear()


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""

    return Settings()
