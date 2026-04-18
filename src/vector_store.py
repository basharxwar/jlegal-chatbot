"""
vector_store.py — Pure-Python/numpy vector store replacing ChromaDB.

Stores chunk embeddings as JSON files (one per law domain) under vector_store/.
Cosine similarity search is done entirely in numpy — zero C++ dependencies.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent / "vector_store"

# In-memory cache: domain -> (chunks_list, embeddings_matrix)
_cache: dict[str, tuple[list[dict], np.ndarray]] = {}


def _domain_path(law_domain: str) -> Path:
    return VECTOR_STORE_DIR / f"{law_domain}.json"


def save_chunks(chunks: list[dict], law_domain: str) -> None:
    """Persist chunk list (each dict must contain an 'embedding' key) to JSON."""
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    path = _domain_path(law_domain)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    _cache.pop(law_domain, None)


def _load_domain(law_domain: str) -> tuple[list[dict], np.ndarray]:
    """Return (chunks, embeddings_matrix) for a domain, using cache."""
    if law_domain in _cache:
        return _cache[law_domain]

    path = _domain_path(law_domain)
    if not path.exists():
        return [], np.empty((0, 0), dtype=np.float32)

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        return [], np.empty((0, 0), dtype=np.float32)

    matrix = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    _cache[law_domain] = (chunks, matrix)
    return chunks, matrix


def search(
    query_embedding: list[float],
    law_domain: str,
    top_k: int,
) -> list[dict]:
    """Return top_k chunks sorted by cosine similarity (highest first).

    Returns chunk dicts with 'score' added and 'embedding' removed.
    """
    chunks, matrix = _load_domain(law_domain)
    if not chunks:
        return []

    q = np.array(query_embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return []
    q = q / q_norm

    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1e-10
    normalized = matrix / row_norms
    scores = normalized @ q

    top_k_actual = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:top_k_actual]

    results = []
    for idx in top_indices:
        chunk = {k: v for k, v in chunks[idx].items() if k != "embedding"}
        chunk["score"] = float(scores[idx])
        results.append(chunk)

    return results


def collection_exists(law_domain: str) -> bool:
    return _domain_path(law_domain).exists()


def collection_count(law_domain: str) -> int:
    chunks, _ = _load_domain(law_domain)
    return len(chunks)
