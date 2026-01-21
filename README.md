# MLX Inference Service

A high-performance, local inference service for embeddings and reranking, built on Apple's MLX framework. Designed for Apple Silicon.

## Features

- **High Performance**: Optimized for Apple Silicon using MLX.
- **Lazy Loading**: Models are loaded only when needed and unloaded after idle time to save memory.
- **Dual Functionality**: Supports both text embeddings and reranking.
- **Configurable**: Adjustable batch sizes, timeouts, and model selection via environment variables.

## Models

- **Embedding**: `mlx-community/Qwen3-Embedding-8B-4bit-DWQ`
- **Reranker**: `jinaai/jina-reranker-v3-mlx`

## Prerequisites

- macOS with Apple Silicon
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)

## Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd inference
   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

3. **Configure environment:**

   Copy the example configuration:

   ```bash
   cp .env.example .env
   ```

   Modify `.env` if needed to adjust models or batch sizes.

## Usage

### Start the Server

```bash
uv run serve
```

The server will start at `http://localhost:11435`.

### API Documentation

Interactive API documentation is available at `http://localhost:11435/docs`.

### API Endpoints

#### 1. List Models

**Endpoint**: `GET /v1/models`

**Compatible**: OpenAI API

```bash
curl http://localhost:11435/v1/models
```

#### 2. Embeddings

**Endpoint**: `POST /v1/embeddings`

**Compatible**: OpenAI API

```bash
curl -X POST http://localhost:11435/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is machine learning?",
    "model": "qwen3-embedding-8b",
    "instruction": "Represent this query for retrieval"
  }'
```

#### 3. Reranking

**Endpoint**: `POST /v1/rerank`

**Compatible**: Jina/Cohere style

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

#### 4. Health Check

**Endpoint**: `GET /health`

```bash
curl http://localhost:11435/health
```

## Development

### Running Tests

```bash
uv run pytest
```

### Linting

```bash
uv run ruff check .
```

## License

This project is licensed under the terms of the [MIT License](LICENSE).

---

## Maintainer

Rui-Jie Huang ([@j-huang-rj](https://github.com/j-huang-rj))
