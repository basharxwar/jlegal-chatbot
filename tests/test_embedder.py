"""Tests for the AraBERT embedding model."""
import pytest


def test_embedder_loads():
    from src.embedder import get_model
    model = get_model()
    assert model is not None


def test_embedding_dimension():
    from src.embedder import embed_texts
    vectors = embed_texts(["نص قانوني للاختبار"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 768, "AraBERT should produce 768-dim vectors"


def test_embedding_is_normalized_or_finite():
    import math
    from src.embedder import embed_texts
    vec = embed_texts(["مادة قانونية"])[0]
    assert all(math.isfinite(v) for v in vec), "Embedding has NaN or Inf"
