"""Configuration settings for the inference service."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Server settings
    host: str = "localhost"
    port: int = 11435

    # Resource management
    lazy_load: bool = True
    model_idle_timeout_seconds: int = 900
    model_unload_check_interval: int = 60

    # Embedding model settings
    embedding_model: str = "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"
    embedding_batch_size: int = 8
    embedding_matryoshka_dim: int = 4096

    # Reranker model settings
    reranker_model: str = "jinaai/jina-reranker-v3-mlx"
    reranker_batch_size: int = 8

    model_config = {
        "env_prefix": "INFERENCE_",
        "env_file": ".env",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""

    return Settings()
