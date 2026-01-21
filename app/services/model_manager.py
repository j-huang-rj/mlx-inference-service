"""Model manager for automatic idle unloading."""

import logging
import threading

from app.config import get_settings
from app.services.embedding_service import get_embedding_service
from app.services.reranker_service import get_reranker_service

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages automatic model unloading after idle timeout."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background monitoring thread."""

        if self._thread is not None and self._thread.is_alive():
            logger.warning("Model manager already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Model manager started.")

    def stop(self) -> None:
        """Stop the background monitoring thread."""

        if self._thread is None or not self._thread.is_alive():
            return

        logger.info("Stopping model manager...")
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        logger.info("Model manager stopped.")

    def _monitor_loop(self) -> None:
        """Background loop that checks for idle models and unloads them."""

        settings = get_settings()

        while not self._stop_event.is_set():
            try:
                # Check embedding service
                embedding_service = get_embedding_service()
                if embedding_service.is_loaded:
                    idle_seconds = embedding_service.get_idle_seconds()
                    if (
                        idle_seconds is not None
                        and idle_seconds >= settings.model_idle_timeout_seconds
                    ):
                        logger.info(
                            f"Embedding model idle for {idle_seconds:.0f}s, unloading..."
                        )
                        embedding_service.unload()

                # Check reranker service
                reranker_service = get_reranker_service()
                if reranker_service.is_loaded:
                    idle_seconds = reranker_service.get_idle_seconds()
                    if (
                        idle_seconds is not None
                        and idle_seconds >= settings.model_idle_timeout_seconds
                    ):
                        logger.info(
                            f"Reranker model idle for {idle_seconds:.0f}s, unloading..."
                        )
                        reranker_service.unload()

            except Exception as e:
                logger.error(f"Error in model manager loop: {e}", exc_info=True)

            # Sleep for check interval or until stop event
            self._stop_event.wait(timeout=settings.model_unload_check_interval)


# Global instance
_model_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""

    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()

    return _model_manager
