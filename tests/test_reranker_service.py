"""Tests for reranker service."""

import time

from app.services.reranker_service import RerankerService, get_reranker_service


def test_singleton_pattern():
    """Test that RerankerService is a singleton."""

    service1 = get_reranker_service()
    service2 = get_reranker_service()
    assert service1 is service2


def test_lazy_loading(mock_reranker):
    """Test that model is not loaded until first use."""

    service = RerankerService()
    service._reranker = None  # Reset
    service._initialized = False
    service.__init__()

    assert not service.is_loaded
    assert service.load_error is None

    # Trigger loading
    service.rerank("query", ["doc1", "doc2"])
    assert service.is_loaded


def test_basic_reranking(mock_reranker):
    """Test basic reranking functionality."""

    service = get_reranker_service()

    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of AI",
        "The weather is sunny today",
        "Neural networks are used in ML",
    ]

    results = service.rerank(query, documents)

    assert len(results) == 3
    assert all("index" in r for r in results)
    assert all("relevance_score" in r for r in results)
    assert all("document" in r for r in results)


def test_top_n_filtering(mock_reranker):
    """Test top_n parameter."""

    service = get_reranker_service()

    documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    results = service.rerank("query", documents, top_n=2)

    assert len(results) == 2


def test_score_ordering(mock_reranker):
    """Test that results are ordered by relevance score descending."""

    service = get_reranker_service()

    documents = ["doc1", "doc2", "doc3"]
    results = service.rerank("query", documents)

    # Check descending order
    for i in range(len(results) - 1):
        assert results[i]["relevance_score"] >= results[i + 1]["relevance_score"]


def test_relevance_score_type(mock_reranker):
    """Test that relevance scores are floats."""

    service = get_reranker_service()

    results = service.rerank("query", ["doc1", "doc2"])

    for result in results:
        assert isinstance(result["relevance_score"], float)


def test_idle_time_tracking(mock_reranker):
    """Test idle time tracking."""

    service = get_reranker_service()

    # Reset state
    service._last_used_time = None

    # Initially no idle time
    assert service.get_idle_seconds() is None

    # After use, idle time should start
    service.rerank("query", ["doc1"])
    time.sleep(0.1)

    idle_seconds = service.get_idle_seconds()
    assert idle_seconds is not None
    assert idle_seconds >= 0.1


def test_unload(mock_reranker):
    """Test model unloading."""

    service = get_reranker_service()

    # Load the model
    service.rerank("query", ["doc1"])
    assert service.is_loaded

    # Unload
    service.unload()
    assert not service.is_loaded
    assert service.get_idle_seconds() is None


def test_empty_documents(mock_reranker):
    """Test handling of empty document list."""

    service = get_reranker_service()

    # Empty list should return empty results
    results = service.rerank("query", [])
    assert results == []


def test_single_document(mock_reranker):
    """Test reranking with single document."""

    service = get_reranker_service()

    results = service.rerank("query", ["single doc"])

    assert len(results) == 1
    assert results[0]["document"] == "single doc"
