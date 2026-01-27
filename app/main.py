"""FastAPI application entry point."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from app.config import get_settings
from app.routes import embeddings, health, models, rerank
from app.services.embedding_service import get_embedding_service
from app.services.model_manager import get_model_manager
from app.services.reranker_service import get_reranker_service
from app.utils.logging import setup_logging

# Configure logging
root_package = __name__.split(".")[0]
setup_logging(level=logging.INFO, name=root_package, use_level_colors=False)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup / shutdown."""

    settings = get_settings()

    # Start model manager for idle unloading
    model_manager = get_model_manager()
    model_manager.start()

    # Create info block
    console = Console()

    info_text = Text()
    info_text.append("🚀 Service Port        ", style="bold cyan")
    info_text.append(f"{settings.port}\n", style="bright_white")

    info_text.append("🤖 Embedding Model     ", style="bold cyan")
    info_text.append(f"{settings.embedding_model}\n", style="bright_white")

    info_text.append("🎯 Reranker Model      ", style="bold cyan")
    info_text.append(f"{settings.reranker_model}\n", style="bright_white")

    info_text.append("⚡ Loading Strategy    ", style="bold cyan")
    info_text.append("Lazy\n", style="bright_white")

    info_text.append("⏱️ Idle Timeout        ", style="bold cyan")
    info_text.append(f"{settings.model_idle_timeout_seconds}s", style="bright_white")

    panel = Panel(
        info_text,
        title="[bold magenta] ⚙️  MLX Inference Service [/bold magenta]",
        border_style="bright_blue",
        padding=(1, 2),
    )

    console.print(panel)

    yield

    # Cleanup on shutdown
    model_manager.stop()
    logger.info("Unloading models...")
    get_embedding_service().unload()
    get_reranker_service().unload()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="MLX Inference Service",
        description="MLX inference service for embeddings and reranking",
        version="0.1.0",
        lifespan=lifespan,
        default_response_class=ORJSONResponse,
    )

    # CORS middleware for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(_: Request, exc: Exception) -> ORJSONResponse:
        logger.exception("Unhandled exception")
        return ORJSONResponse(
            status_code=500,
            content={"detail": f"Internal Server Error: {exc}"},
        )

    # Include routers
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(embeddings.router)
    app.include_router(rerank.router)

    return app


# Create app instance
app = create_app()


def run() -> None:
    """Run the server."""

    settings = get_settings()

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=logging.INFO,
    )


def serve() -> None:
    """
    Run the server with optional CLI argument overrides.

    CLI arguments override .env values. Use --help for available options.
    """

    import argparse
    import os
    import subprocess
    from pathlib import Path

    from app.config import apply_cli_overrides, clear_settings_cache

    # Load .env file first
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file)

    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="MLX Inference Service - Embeddings & Reranking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server settings
    parser.add_argument(
        "--host",
        "-H",
        type=str,
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["dev", "prod"],
        help="Run mode (dev enables auto-reload)",
    )

    # Resource management
    parser.add_argument(
        "--lazy_load",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        metavar="BOOL",
        help="Enable lazy loading of models",
    )
    parser.add_argument(
        "--model_idle_timeout_seconds",
        type=int,
        metavar="SECONDS",
        help="Seconds before unloading idle models",
    )
    parser.add_argument(
        "--model_unload_check_interval",
        type=int,
        metavar="SECONDS",
        help="Interval to check for idle models",
    )

    # Embedding settings
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--embedding_matryoshka_dim",
        type=int,
        help="Matryoshka dimension for embeddings",
    )

    # Reranker settings
    parser.add_argument(
        "--reranker_batch_size",
        type=int,
        help="Batch size for reranking",
    )

    args = parser.parse_args()

    # Apply CLI overrides to environment variables
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    if cli_args:
        apply_cli_overrides(cli_args)
        clear_settings_cache()

    # Get settings with CLI overrides applied
    settings = get_settings()

    # Determine run mode
    mode = args.mode or os.getenv("INFERENCE_MODE", "prod")
    fastapi_cmd = "dev" if mode == "dev" else "run"

    cmd = [
        "fastapi",
        fastapi_cmd,
        "--host",
        settings.host,
        "--port",
        str(settings.port),
        "app/main.py",
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
