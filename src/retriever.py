"""
retriever.py — Vector similarity search for JLegal-ChatBot.

Uses numpy cosine similarity via src.vector_store — zero C++ dependencies.
"""

import traceback
from typing import Optional

from src.ingest import get_vectorizer
from src.vector_store import search as vs_search, collection_exists

ALL_DOMAINS: list[str] = [
    "Labor",
    "Commercial",
    "PersonalStatus",
    "Cybercrime",
    "CivilService",
]

# When searching all domains, fetch 3 per domain then merge to top_k.
# This gives better cross-law coverage than fetching top_k from each.
_PER_DOMAIN_K = 3


def embed_query(query_text: str) -> list[float]:
    """Embed a single query string using the saved TF-IDF vectorizer."""
    vec = get_vectorizer()
    sparse = vec.transform([query_text])
    return sparse.toarray()[0].tolist()


def retrieve(
    query_text: str,
    law_domain: Optional[str] = None,
    top_k: int = 5,
    threshold: float = 0.05,
) -> list[dict]:
    """Retrieve the most relevant chunks for a given query.

    Parameters
    ----------
    query_text:
        The user's legal question.
    law_domain:
        If provided, restricts search to that domain. None = all domains.
    top_k:
        Maximum number of results to return after threshold filtering.
    threshold:
        Minimum cosine similarity score (0-1) for inclusion.

    Returns
    -------
    list[dict] with keys: chunk_id, chunk_text, law_domain, law_name_ar,
        article_number, page_number, score, rank
    """
    try:
        query_embedding = embed_query(query_text)
        raw: list[dict] = []

        if law_domain:
            if collection_exists(law_domain):
                raw = vs_search(query_embedding, law_domain, top_k)
        else:
            # Fetch _PER_DOMAIN_K from each domain then merge — better coverage
            for domain in ALL_DOMAINS:
                if collection_exists(domain):
                    results = vs_search(query_embedding, domain, _PER_DOMAIN_K)
                    raw.extend(results)

        filtered = [r for r in raw if r["score"] >= threshold]
        filtered.sort(key=lambda x: x["score"], reverse=True)
        top_results = filtered[:top_k]

        for rank, item in enumerate(top_results, start=1):
            item["rank"] = rank

        return top_results

    except Exception as exc:
        print(f"[ERROR] retriever exception: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return []
