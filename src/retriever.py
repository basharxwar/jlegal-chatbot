"""
retriever.py — Vector similarity search for JLegal-ChatBot.

Uses numpy cosine similarity via src.vector_store — zero C++ dependencies.
Supports query expansion: colloquial dialect queries are rewritten to formal
MSA legal Arabic via Claude Haiku before embedding, then results from both
versions are merged for better recall.
"""

import traceback
from typing import Optional

from src.embedder import embed_text
from src.vector_store import search as vs_search, collection_exists

ALL_DOMAINS: list[str] = [
    "Labor",
    "Commercial",
    "PersonalStatus",
    "Cybercrime",
    "CivilService",
    "CivilStatus",
    "SocialSecurity",
    "PersonalStatus2019",
    "TrafficLaw",
    "HRManagement",
]

# Per-domain fetch count when searching all domains, then merged to top_k
_PER_DOMAIN_K = 4


def embed_query(query_text: str) -> list[float]:
    """Embed a single query string using AraBERT."""
    return embed_text(query_text)


def _retrieve_single(
    query_text: str,
    law_domain: Optional[str],
    top_k: int,
    threshold: float,
) -> list[dict]:
    """Core retrieval for one query string — no expansion, no ranking."""
    query_embedding = embed_query(query_text)
    raw: list[dict] = []

    if law_domain:
        if collection_exists(law_domain):
            raw = vs_search(query_embedding, law_domain, top_k)
    else:
        for domain in ALL_DOMAINS:
            if collection_exists(domain):
                results = vs_search(query_embedding, domain, _PER_DOMAIN_K)
                raw.extend(results)

    return [r for r in raw if r["score"] >= threshold]


def retrieve(
    query_text: str,
    law_domain: Optional[str] = None,
    top_k: int = 8,
    threshold: float = 0.50,
    expand: bool = True,
) -> list[dict]:
    """Retrieve the most relevant chunks for a given query.

    Parameters
    ----------
    query_text:
        The user's legal question (dialect or MSA).
    law_domain:
        Restrict to one domain. None = search all.
    top_k:
        Maximum results after threshold filtering.
    threshold:
        Minimum cosine similarity score for inclusion.
    expand:
        If True, rewrite the query to formal MSA via Claude Haiku and
        search both the original and expanded versions, then merge.

    Returns
    -------
    list[dict] with keys: chunk_id, chunk_text, law_domain, law_name_ar,
        article_number, page_number, score, rank
    """
    try:
        queries = [query_text]

        if expand:
            from src.generator import expand_query
            expanded = expand_query(query_text)
            if expanded and expanded != query_text and len(expanded) > 5:
                queries.append(expanded)

        # Search each query version, merge keeping best score per chunk
        seen: dict[str, dict] = {}
        for q in queries:
            for chunk in _retrieve_single(q, law_domain, top_k, threshold):
                cid = chunk.get("chunk_id")
                if cid not in seen or chunk["score"] > seen[cid]["score"]:
                    seen[cid] = chunk

        final = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        top_results = final[:top_k]

        for rank, item in enumerate(top_results, start=1):
            item["rank"] = rank

        return top_results

    except Exception as exc:
        print(f"[ERROR] retriever exception: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return []
