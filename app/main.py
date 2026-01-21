"""FastAPI application entry point."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    async def global_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception")
        return JSONResponse(
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
    Run the server.
    """

    import os
    import subprocess
    from pathlib import Path

    # Load .env file if it exists
    env_file = Path(".env")

    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file)

    settings = get_settings()

    mode = os.getenv("INFERENCE_MODE", "prod")

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
