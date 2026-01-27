"""Embedding service using mlx-embeddings."""

import gc
import logging
import threading
import time
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_embeddings import load

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Thread-safe singleton embedding service."""

    _instance: "EmbeddingService | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False

        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._model: Any = None
        self._tokenizer: Any = None
        self._model_lock = threading.Lock()
        self._load_error: str | None = None
        self._last_used_time: float | None = None
        self._initialized = True

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""

        return self._model is not None

    @property
    def load_error(self) -> str | None:
        """Get load error if any."""

        return self._load_error

    def get_idle_seconds(self) -> float | None:
        """Get seconds since last use, or None if never used or not loaded."""

        if self._last_used_time is None or not self.is_loaded:
            return None

        return time.time() - self._last_used_time

    def _ensure_loaded(self) -> None:
        """Lazy load model on first use."""

        if self._model is not None:
            return

        with self._model_lock:
            if self._model is not None:
                return

            settings = get_settings()
            try:
                logger.info(f"Loading embedding model: {settings.embedding_model}")

                # Load the model using mlx-embeddings
                self._model, self._tokenizer = load(settings.embedding_model)

                # Qwen3 Embedding requires left padding for correct last-token pooling
                self._tokenizer._tokenizer.padding_side = "left"

                logger.info("Embedding model loaded successfully")
                self._load_error = None

            except Exception as e:
                self._load_error = str(e)
                logger.error(f"Failed to load embedding model: {e}")
                raise RuntimeError(f"Failed to load embedding model: {e}") from e

    def _last_token_pool(
        self, hidden_states: mx.array, attention_mask: mx.array
    ) -> mx.array:
        """Extract embeddings using last token pooling.

        Matches official Qwen3 implementation:
        - With left padding: last token is always at position -1
        - With right padding: last token is at (sequence_length - 1)
        """

        # Check if left padding: all sequences have attention=1 at last position
        left_padding = mx.sum(attention_mask[:, -1]) == attention_mask.shape[0]

        if left_padding:
            # With left padding, last real token is always at the last position
            return hidden_states[:, -1, :]
        else:
            # With right padding, find the last non-padded token per sequence
            sequence_lengths = mx.sum(attention_mask, axis=1) - 1
            batch_size = hidden_states.shape[0]
            batch_indices = mx.arange(batch_size)
            return hidden_states[batch_indices, sequence_lengths.astype(mx.int32), :]

    def _normalize(self, embeddings: mx.array) -> mx.array:
        """L2 normalize embeddings."""

        norms = mx.linalg.norm(embeddings, axis=-1, keepdims=True)

        return embeddings / mx.maximum(norms, mx.array(1e-12))

    def embed(
        self,
        texts: list[str],
        dimensions: int | None = None,
        instruction: str | None = None,
    ) -> tuple[list[list[float]], int]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            dimensions: Optional dimension to truncate to.
            instruction: Optional instruction prefix for queries.

        Returns:
            Tuple of (embeddings, token_count)
        """

        self._ensure_loaded()
        settings = get_settings()

        # Update last used time
        self._last_used_time = time.time()

        # Apply instruction prefix if provided
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery:{text}" for text in texts]

        all_embeddings: list[list[float]] = []
        total_tokens = 0

        # Process in batches
        batch_size = settings.embedding_batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize with max_length
            tokens = self._tokenizer._tokenizer(
                batch_texts,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=8192,
            )
            input_ids = mx.array(tokens["input_ids"])
            attention_mask = mx.array(tokens["attention_mask"])
            total_tokens += int(mx.sum(attention_mask).item())

            # Forward pass through model
            outputs = self._model(input_ids)

            # Get hidden states (last layer)
            if hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs

            # Pool and normalize
            embeddings = self._last_token_pool(hidden_states, attention_mask)
            embeddings = self._normalize(embeddings)

            # Handle dimension truncation if requested
            if dimensions and dimensions < embeddings.shape[-1]:
                # Truncate dimensions
                embeddings = embeddings[:, :dimensions]

                # Re-normalize after truncation to maintain unit vectors
                embeddings = self._normalize(embeddings)

            # Convert to Python lists
            batch_embeddings = np.array(embeddings).tolist()
            all_embeddings.extend(batch_embeddings)

        return all_embeddings, total_tokens

    def unload(self) -> None:
        """Unload model to free memory."""

        with self._model_lock:
            if self._model is not None:
                self._model = None
                self._tokenizer = None
                self._last_used_time = None

                # Force garbage collection to release memory
                gc.collect()

                # Clear MLX Metal GPU memory cache
                mx.clear_cache()

                logger.info("Embedding model unloaded and memory cleared")


def get_embedding_service() -> EmbeddingService:
    """Get embedding service singleton."""

    return EmbeddingService()
