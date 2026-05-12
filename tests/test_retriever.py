"""Tests for the retrieval pipeline."""
import pytest


def test_retrieve_returns_chunks_for_known_topic():
    from src.retriever import retrieve
    results = retrieve(
        query_text="ما هي مدة الإجازة السنوية",
        law_domain="Labor",
        top_k=5,
        threshold=0.0,
    )
    assert len(results) > 0, "Retrieval returned no chunks for known topic"


def test_retrieved_chunks_have_required_fields():
    from src.retriever import retrieve
    results = retrieve(
        query_text="حقوق العامل",
        law_domain="Labor",
        top_k=3,
        threshold=0.0,
    )
    if not results:
        pytest.skip("No results to inspect")
    chunk = results[0]
    for field in ["chunk_id", "chunk_text", "score", "rank"]:
        assert field in chunk, f"Retrieved chunk missing field: {field}"


def test_scores_are_sorted_descending():
    from src.retriever import retrieve
    results = retrieve(
        query_text="عقد العمل",
        law_domain="Labor",
        top_k=10,
        threshold=0.0,
    )
    if len(results) < 2:
        pytest.skip("Not enough results to check sorting")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Scores are not sorted descending"


def test_threshold_filters_low_score_results():
    from src.retriever import retrieve
    very_high = retrieve("xyz nonsense query", top_k=5, threshold=0.99)
    assert len(very_high) == 0, "Threshold did not filter low-similarity results"
