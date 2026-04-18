"""
pipeline.py — Orchestration layer for JLegal-ChatBot.

Coordinates the full RAG flow:
    user query -> input validation -> retrieve chunks -> generate answer
    -> log everything to SQLite -> return structured result to the UI.

All exceptions are caught and converted to user-friendly Arabic error messages
so the Streamlit UI never crashes due to a backend failure.
"""

import traceback
import uuid
from typing import Optional

from src.retriever import retrieve
from src.generator import generate_answer
from src.database import (
    create_session,
    touch_session,
    insert_query,
    insert_response,
    insert_query_chunks,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.05
DEFAULT_TOP_K = 5

ERROR_MESSAGE_AR = (
    "عذراً، حدث خطأ أثناء معالجة سؤالك. "
    "يُرجى المحاولة مرة أخرى أو التواصل مع الدعم الفني."
)


# ---------------------------------------------------------------------------
# Session management helper
# ---------------------------------------------------------------------------

def ensure_session(session_id: Optional[str] = None) -> str:
    """Return a valid session_id, creating a new session row if needed."""
    if session_id is None:
        return create_session()
    # INSERT OR IGNORE: creates the row if missing, no-op if it already exists.
    # touch_session's UPDATE silently does nothing when the row is absent,
    # so we must guarantee the row exists before relying on it.
    create_session(session_id=session_id)
    touch_session(session_id)
    return session_id


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_query(
    query_text: str,
    session_id: str,
    law_domain: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    similarity_threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Execute the full RAG pipeline for a single user query.

    Steps
    -----
    1. Validate input (non-empty query, valid session).
    2. Ensure the session row exists in SQLite.
    3. Retrieve relevant chunks from the vector store.
    4. Generate an Arabic answer via the Claude API.
    5. Log QUERY, RESPONSE, and QUERY_CHUNK rows to SQLite.
    6. Return a result dict for the UI layer.

    Returns
    -------
    dict with keys:
        success        (bool)
        response_text  (str)
        chunks         (list[dict])
        is_no_result   (bool)
        tokens_used    (int)
        model_used     (str)
        query_id       (str)
        error          (str | None)
    """
    try:
        return _run_query_inner(
            query_text=query_text,
            session_id=session_id,
            law_domain=law_domain,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
    except BaseException as exc:
        traceback.print_exc()
        return _error_result(ERROR_MESSAGE_AR, exc)


def _run_query_inner(
    query_text: str,
    session_id: str,
    law_domain: Optional[str],
    top_k: int,
    similarity_threshold: float,
) -> dict:
    """Inner implementation of run_query — may raise; caller handles it."""

    # ---- 1. Input validation -----------------------------------------------
    if not query_text or not query_text.strip():
        return _error_result("الرجاء إدخال سؤال قانوني قبل الإرسال.")

    # ---- 2. Session guard --------------------------------------------------
    try:
        session_id = ensure_session(session_id)
    except Exception as exc:
        traceback.print_exc()
        return _error_result(ERROR_MESSAGE_AR, exc)

    query_id = str(uuid.uuid4())

    try:
        # ---- 3. Retrieval --------------------------------------------------
        chunks = retrieve(
            query_text=query_text.strip(),
            law_domain=law_domain,
            top_k=top_k,
            threshold=similarity_threshold,
        )

        # ---- 4. Generation -------------------------------------------------
        gen_result = generate_answer(
            query_text=query_text.strip(),
            chunks=chunks,
        )

        # ---- 5. Logging ----------------------------------------------------
        _log_to_db(
            query_id=query_id,
            session_id=session_id,
            query_text=query_text.strip(),
            law_domain=law_domain,
            similarity_threshold=similarity_threshold,
            chunks=chunks,
            gen_result=gen_result,
        )

        # ---- 6. Return result ----------------------------------------------
        return {
            "success": True,
            "response_text": gen_result["response_text"],
            "chunks": gen_result["chunks_used"],
            "is_no_result": gen_result["is_no_result"],
            "tokens_used": gen_result["tokens_used"],
            "model_used": gen_result["model_used"],
            "query_id": query_id,
            "error": None,
        }

    except Exception as exc:
        print(f"[ERROR] pipeline exception: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        try:
            _log_failed_query(
                query_id=query_id,
                session_id=session_id,
                query_text=query_text.strip(),
                law_domain=law_domain,
                similarity_threshold=similarity_threshold,
                error_message=str(exc),
            )
        except Exception:
            pass  # Logging failure must never propagate to the UI

        return _error_result(ERROR_MESSAGE_AR, exc)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_to_db(
    query_id: str,
    session_id: str,
    query_text: str,
    law_domain: Optional[str],
    similarity_threshold: float,
    chunks: list[dict],
    gen_result: dict,
) -> None:
    """Write QUERY, RESPONSE, and QUERY_CHUNK rows to SQLite."""
    insert_query(
        session_id=session_id,
        query_text=query_text,
        law_domain=law_domain,
        similarity_threshold=similarity_threshold,
        query_id=query_id,
    )

    insert_response(
        query_id=query_id,
        response_text=gen_result["response_text"],
        llm_model=gen_result["model_used"],
        tokens_used=gen_result["tokens_used"],
        is_no_result=gen_result["is_no_result"],
    )

    if chunks:
        insert_query_chunks(
            query_id=query_id,
            chunks=[
                {
                    "chunk_id": c["chunk_id"],
                    "score": c["score"],
                    "rank": c["rank"],
                }
                for c in chunks
            ],
        )


def _log_failed_query(
    query_id: str,
    session_id: str,
    query_text: str,
    law_domain: Optional[str],
    similarity_threshold: float,
    error_message: str,
) -> None:
    """Persist a failed query so there is at least a partial audit record."""
    insert_query(
        session_id=session_id,
        query_text=query_text,
        law_domain=law_domain,
        similarity_threshold=similarity_threshold,
        query_id=query_id,
    )
    insert_response(
        query_id=query_id,
        response_text=f"[ERROR] {error_message}",
        llm_model="N/A",
        tokens_used=0,
        is_no_result=True,
    )


def _error_result(message: str, exc: Optional[Exception] = None) -> dict:
    """Build a standardised error result dict."""
    return {
        "success": False,
        "response_text": message,
        "chunks": [],
        "is_no_result": True,
        "tokens_used": 0,
        "model_used": "N/A",
        "query_id": None,
        "error": str(exc) if exc else message,
    }
