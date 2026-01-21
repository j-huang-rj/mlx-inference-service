"""Integration tests for API endpoints."""


def test_health_endpoint(test_client):
    """Test health check endpoint."""

    response = test_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "models" in data
    assert len(data["models"]) == 2


def test_root_endpoint(test_client):
    """Test root endpoint."""

    response = test_client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "service" in data
    assert "endpoints" in data


def test_list_models(test_client):
    """Test model listing endpoint."""

    response = test_client.get("/v1/models")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2

    assert "Qwen3-Embedding-8B" in data["data"][0]["id"]
    assert "jina-reranker-v3" in data["data"][1]["id"]


def test_embeddings_single_text(test_client):
    """Test embeddings endpoint with single text."""

    response = test_client.post(
        "/v1/embeddings",
        json={
            "input": "Hello world",
            "model": "qwen3-embedding-8b",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "list"
    assert data["model"] == "qwen3-embedding-8b"
    assert len(data["data"]) == 1
    assert len(data["data"][0]["embedding"]) > 0
    assert "usage" in data
    assert data["usage"]["total_tokens"] > 0


def test_embeddings_multiple_texts(test_client):
    """Test embeddings endpoint with multiple texts."""

    response = test_client.post(
        "/v1/embeddings",
        json={
            "input": ["Text 1", "Text 2", "Text 3"],
            "model": "qwen3-embedding-8b",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data["data"]) == 3
    for i, item in enumerate(data["data"]):
        assert item["index"] == i


def test_embeddings_with_dimensions(test_client):
    """Test embeddings endpoint with dimension truncation."""

    response = test_client.post(
        "/v1/embeddings",
        json={
            "input": "Test",
            "model": "qwen3-embedding-8b",
            "dimensions": 512,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data["data"][0]["embedding"]) == 512


def test_embeddings_with_instruction(test_client):
    """Test embeddings endpoint with instruction parameter."""

    response = test_client.post(
        "/v1/embeddings",
        json={
            "input": "How to learn Python?",
            "model": "qwen3-embedding-8b",
            "instruction": "Represent this query for retrieval",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data["data"]) == 1
    assert len(data["data"][0]["embedding"]) > 0


def test_embeddings_empty_input_error(test_client):
    """Test that empty input returns 400 error."""

    response = test_client.post(
        "/v1/embeddings",
        json={
            "input": [],
            "model": "qwen3-embedding-8b",
        },
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_embeddings_too_many_texts_error(test_client):
    """Test that too many texts returns 400 error."""

    response = test_client.post(
        "/v1/embeddings",
        json={
            "input": ["text"] * 101,  # More than 100
            "model": "qwen3-embedding-8b",
        },
    )

    assert response.status_code == 400
    assert "maximum" in response.json()["detail"].lower()


def test_rerank_basic(test_client):
    """Test rerank endpoint with basic request."""

    response = test_client.post(
        "/v1/rerank",
        json={
            "query": "What is machine learning?",
            "documents": [
                "ML is a subset of AI",
                "The weather is nice",
                "Neural networks are used in ML",
            ],
            "model": "jina-reranker-v3",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["model"] == "jina-reranker-v3"
    assert len(data["results"]) == 3
    assert all("index" in r for r in data["results"])
    assert all("relevance_score" in r for r in data["results"])
    assert all("document" in r for r in data["results"])


def test_rerank_with_top_n(test_client):
    """Test rerank endpoint with top_n parameter."""

    response = test_client.post(
        "/v1/rerank",
        json={
            "query": "test query",
            "documents": ["doc1", "doc2", "doc3", "doc4"],
            "top_n": 2,
            "model": "jina-reranker-v3",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data["results"]) == 2


def test_rerank_without_documents(test_client):
    """Test rerank endpoint without return_documents."""

    response = test_client.post(
        "/v1/rerank",
        json={
            "query": "test",
            "documents": ["doc1", "doc2"],
            "return_documents": False,
            "model": "jina-reranker-v3",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Documents should be None when return_documents=False
    assert all(r["document"] is None for r in data["results"])


def test_rerank_empty_query_error(test_client):
    """Test that empty query returns 400 error."""

    response = test_client.post(
        "/v1/rerank",
        json={
            "query": "   ",  # Whitespace only
            "documents": ["doc1"],
            "model": "jina-reranker-v3",
        },
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_rerank_too_many_documents_error(test_client):
    """Test that too many documents returns 400 error."""

    response = test_client.post(
        "/v1/rerank",
        json={
            "query": "test",
            "documents": ["doc"] * 1001,  # More than 1000
            "model": "jina-reranker-v3",
        },
    )

    assert response.status_code == 400
    assert "maximum" in response.json()["detail"].lower()


def test_rerank_score_ordering(test_client):
    """Test that rerank results are ordered by score descending."""

    response = test_client.post(
        "/v1/rerank",
        json={
            "query": "test",
            "documents": ["doc1", "doc2", "doc3"],
            "model": "jina-reranker-v3",
        },
    )

    assert response.status_code == 200
    data = response.json()

    scores = [r["relevance_score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True)
