"""Reranker service using jina-reranker-v3-mlx."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from huggingface_hub import snapshot_download

from app.config import get_settings

# Type stub for MLXReranker
if TYPE_CHECKING:

    class MLXReranker(Protocol):
        """Type stub for the MLXReranker class from the model's rerank.py."""

        def __init__(
            self, model_path: str, projector_path: str | None = None
        ) -> None: ...

        def rerank(
            self,
            query: str,
            documents: list[str],
            top_n: int | None = None,
            return_embeddings: bool = False,
        ) -> list[dict[str, Any]]: ...


logger = logging.getLogger(__name__)


class RerankerService:
    """Thread-safe singleton reranker service for jina-reranker-v3-mlx."""

    _instance: RerankerService | None = None
    _lock = threading.Lock()

    def __new__(cls) -> RerankerService:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._reranker: MLXReranker | None = None
        self._model_lock = threading.Lock()
        self._load_error: str | None = None
        self._model_path: Path | None = None
        self._last_used_time: float | None = None
        self._initialized = True

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""

        return self._reranker is not None

    @property
    def load_error(self) -> str | None:
        """Get load error if any."""

        return self._load_error

    def get_idle_seconds(self) -> float | None:
        """Get seconds since last use, or None if never used or not loaded."""

        import time

        if self._last_used_time is None or not self.is_loaded:
            return None

        return time.time() - self._last_used_time

    def _ensure_loaded(self) -> None:
        """Lazy load model on first use."""

        if self._reranker is not None:
            return

        with self._model_lock:
            if self._reranker is not None:
                return

            settings = get_settings()
            try:
                logger.info(f"Loading reranker model: {settings.reranker_model}")

                # Download model from HuggingFace
                model_path = snapshot_download(
                    repo_id=settings.reranker_model,
                    allow_patterns=[
                        "*.safetensors",
                        "*.json",
                        "*.txt",
                        "*.model",
                        "*.py",
                    ],
                )
                self._model_path = Path(model_path)

                # Add model path to sys.path to import rerank module
                if str(self._model_path) not in sys.path:
                    sys.path.insert(0, str(self._model_path))

                # Dynamically import the reranker
                from rerank import MLXReranker  # type: ignore[import-not-found]

                projector_path = self._model_path / "projector.safetensors"
                self._reranker = MLXReranker(
                    model_path=str(self._model_path),
                    projector_path=str(projector_path),
                )

                logger.info("Reranker model loaded successfully")
                self._load_error = None

            except Exception as e:
                self._load_error = str(e)
                logger.error(f"Failed to load reranker model: {e}")
                raise RuntimeError(f"Failed to load reranker model: {e}") from e

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of documents to rerank
            top_n: Return only top N results (default: all)

        Returns:
            List of dicts with 'index', 'relevance_score', 'document' keys
        """

        self._ensure_loaded()

        # Update last used time
        import time

        self._last_used_time = time.time()

        settings = get_settings()
        batch_size = settings.reranker_batch_size
        all_results = []

        # Process in batches
        assert self._reranker is not None

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]

            # Rerank batch
            batch_results = self._reranker.rerank(
                query=query, documents=batch_docs, top_n=None
            )

            # Adjust indices with offset and collect
            for result in batch_results:
                result["index"] = result["index"] + i
                all_results.append(result)

        # Sort aggregated results by score
        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Apply top_n globally
        if top_n is not None:
            all_results = all_results[:top_n]

        # Normalize results format
        normalized_results = []
        for result in all_results:
            normalized_results.append(
                {
                    "index": result["index"],
                    "relevance_score": float(result["relevance_score"]),
                    "document": result.get("document"),
                }
            )

        return normalized_results

    def unload(self) -> None:
        """Unload model to free memory."""

        with self._model_lock:
            self._reranker = None

            # Remove model path from sys.path
            if self._model_path and str(self._model_path) in sys.path:
                sys.path.remove(str(self._model_path))

            self._model_path = None
            self._last_used_time = None

            logger.info("Reranker model unloaded")


def get_reranker_service() -> RerankerService:
    """Get reranker service singleton."""

    return RerankerService()
