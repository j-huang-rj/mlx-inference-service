# MLX Inference Service

A high-performance inference service for **Qwen3 Embedding 0.6B** and **Jina Reranker V3**, optimized for Apple Silicon using the MLX framework.

## ✨ Features

- 🚀 **Apple Silicon Optimized** — Built on MLX for maximum Metal GPU performance
- 🧠 **Qwen3 Embedding 0.6B** — Embeddings with Matryoshka support
- 🎯 **Jina Reranker V3** — Listwise document reranker for multilingual retrieval
- 💤 **Smart Memory Management** — Lazy loading with automatic unload after idle timeout
- 🔌 **OpenAI-Compatible API** — Standardized interface for requests

## 📦 Models

| Type          | Model                         | Parameters |
| ------------- | ----------------------------- | ---------- |
| **Embedding** | `Qwen/Qwen3-Embedding-0.6B`   | 0.6B       |
| **Reranker**  | `jinaai/jina-reranker-v3-mlx` | 0.6B       |

## 🛠 Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/j-huang-rj/mlx-inference-service.git
cd mlx-inference-service
uv sync

# Configure
cp .env.example .env

# Run
uv run serve
```

Service starts at `http://localhost:11435`

## ⚙️ CLI Options

Override `.env` settings via command-line arguments:

```bash
# Custom port
uv run serve --port 11434

# Multiple overrides
uv run serve -p 11434 -H 0.0.0.0 --embedding_batch_size 512

# Dev mode (auto-reload)
uv run serve --mode dev

# View all options
uv run serve --help
```

| Argument                        | Short | Description                                |
| ------------------------------- | ----- | ------------------------------------------ |
| `--host`                        | `-H`  | Host to bind the server                    |
| `--port`                        | `-p`  | Port to bind the server                    |
| `--mode`                        | `-m`  | `dev` or `prod` (dev enables auto-reload)  |
| `--lazy_load`                   |       | Enable lazy model loading (`true`/`false`) |
| `--model_idle_timeout_seconds`  |       | Seconds before unloading idle models       |
| `--model_unload_check_interval` |       | Interval to check for idle models          |
| `--embedding_batch_size`        |       | Batch size for embedding generation        |
| `--embedding_matryoshka_dim`    |       | Matryoshka dimension for embeddings        |
| `--reranker_batch_size`         |       | Batch size for reranking                   |

## 📡 API Endpoints

### Embeddings (OpenAI-compatible)

```bash
curl -X POST http://localhost:11435/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is machine learning?",
    "model": "Qwen/Qwen3-Embedding-0.6B"
  }'
```

### Reranking (Jina/Cohere-style)

```bash
curl -X POST http://localhost:11435/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a subset of artificial intelligence",
      "The weather is sunny today",
      "Deep learning uses neural networks"
    ],
    "top_n": 2
  }'
```

### Other Endpoints

| Endpoint     | Method | Description                    |
| ------------ | ------ | ------------------------------ |
| `/v1/models` | GET    | List available models          |
| `/health`    | GET    | Health check with model status |
| `/docs`      | GET    | Interactive API documentation  |

## 🧪 Development

```bash
# Run tests
uv run pytest

# Linting
uv run ruff check .
```

## 📄 License

[MIT License](LICENSE)

---

**Maintainer**: Rui-Jie Huang ([@j-huang-rj](https://github.com/j-huang-rj))
