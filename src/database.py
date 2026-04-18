"""
database.py — SQLite schema creation and all CRUD operations for JLegal-ChatBot.

All table definitions match the exact schema specified in the project requirements.
WAL mode and foreign-key enforcement are enabled on every connection.
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "jlegal.db"

# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection.

    Enables WAL journal mode and foreign-key enforcement on every new
    connection so the settings are guaranteed regardless of call site.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS SESSION (
    session_id      TEXT PRIMARY KEY,
    user_identifier TEXT DEFAULT NULL,
    language        TEXT NOT NULL DEFAULT 'ar' CHECK (language IN ('ar','en')),
    created_at      TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    last_active_at  TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

CREATE TABLE IF NOT EXISTS DOCUMENT (
    document_id    TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    law_domain     TEXT NOT NULL CHECK (law_domain IN ('Labor','Commercial','PersonalStatus','Cybercrime','CivilService')),
    language       TEXT NOT NULL DEFAULT 'ar',
    source_url     TEXT DEFAULT NULL,
    effective_date DATE DEFAULT NULL,
    ingested_at    TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

CREATE TABLE IF NOT EXISTS CHUNK (
    chunk_id         TEXT PRIMARY KEY,
    document_id      TEXT NOT NULL,
    chunk_index      INTEGER NOT NULL,
    chunk_text       TEXT NOT NULL,
    article_number   VARCHAR(50) DEFAULT NULL,
    page_number      INTEGER DEFAULT NULL,
    embedding_model  TEXT NOT NULL,
    token_count      INTEGER NOT NULL DEFAULT 0,
    created_at       TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    FOREIGN KEY (document_id) REFERENCES DOCUMENT(document_id) ON DELETE CASCADE,
    UNIQUE (document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS QUERY (
    query_id             TEXT PRIMARY KEY,
    session_id           TEXT NOT NULL,
    query_text           TEXT NOT NULL,
    law_domain           TEXT DEFAULT NULL,
    similarity_threshold REAL NOT NULL DEFAULT 0.75,
    created_at           TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    FOREIGN KEY (session_id) REFERENCES SESSION(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS RESPONSE (
    response_id      TEXT PRIMARY KEY,
    query_id         TEXT NOT NULL UNIQUE,
    response_text    TEXT NOT NULL,
    llm_model        TEXT NOT NULL,
    tokens_used      INTEGER NOT NULL DEFAULT 0,
    is_no_result     INTEGER NOT NULL DEFAULT 0 CHECK (is_no_result IN (0,1)),
    created_at       TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    FOREIGN KEY (query_id) REFERENCES QUERY(query_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS QUERY_CHUNK (
    query_id         TEXT NOT NULL,
    chunk_id         TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    rank             INTEGER NOT NULL,
    PRIMARY KEY (query_id, chunk_id),
    FOREIGN KEY (query_id)  REFERENCES QUERY(query_id)  ON DELETE CASCADE,
    FOREIGN KEY (chunk_id)  REFERENCES CHUNK(chunk_id)  ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_query_session  ON QUERY(session_id);
CREATE INDEX IF NOT EXISTS idx_chunk_document ON CHUNK(document_id);
CREATE INDEX IF NOT EXISTS idx_qc_query       ON QUERY_CHUNK(query_id);
"""


def init_db() -> None:
    """Create all tables and indexes if they do not already exist."""
    with get_connection() as conn:
        conn.executescript(_DDL)


# ---------------------------------------------------------------------------
# SESSION CRUD
# ---------------------------------------------------------------------------

def create_session(
    session_id: Optional[str] = None,
    user_identifier: Optional[str] = None,
    language: str = "ar",
) -> str:
    """Insert a new session row and return the session_id."""
    session_id = session_id or str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO SESSION (session_id, user_identifier, language)
            VALUES (?, ?, ?)
            """,
            (session_id, user_identifier, language),
        )
    return session_id


def touch_session(session_id: str) -> None:
    """Update last_active_at for an existing session."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE SESSION SET last_active_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (session_id,),
        )


# ---------------------------------------------------------------------------
# DOCUMENT CRUD
# ---------------------------------------------------------------------------

def upsert_document(
    document_id: str,
    title: str,
    law_domain: str,
    language: str = "ar",
    source_url: Optional[str] = None,
    effective_date: Optional[str] = None,
) -> str:
    """Insert a document row; silently ignore if already present (idempotent)."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO DOCUMENT
                (document_id, title, law_domain, language, source_url, effective_date)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (document_id, title, law_domain, language, source_url, effective_date),
        )
    return document_id


def get_document_by_title(title: str) -> Optional[sqlite3.Row]:
    """Return the DOCUMENT row whose title matches, or None."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM DOCUMENT WHERE title = ?", (title,)
        ).fetchone()
    return row


# ---------------------------------------------------------------------------
# CHUNK CRUD
# ---------------------------------------------------------------------------

def insert_chunk(
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    chunk_text: str,
    embedding_model: str,
    article_number: Optional[str] = None,
    page_number: Optional[int] = None,
    token_count: int = 0,
) -> None:
    """Insert a single chunk row; ignore duplicates (idempotent re-runs)."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO CHUNK
                (chunk_id, document_id, chunk_index, chunk_text,
                 article_number, page_number, embedding_model, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                document_id,
                chunk_index,
                chunk_text,
                article_number,
                page_number,
                embedding_model,
                token_count,
            ),
        )


def count_chunks_for_document(document_id: str) -> int:
    """Return the number of chunks already stored for a document."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM CHUNK WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# QUERY / RESPONSE / QUERY_CHUNK CRUD
# ---------------------------------------------------------------------------

def insert_query(
    session_id: str,
    query_text: str,
    law_domain: Optional[str] = None,
    similarity_threshold: float = 0.75,
    query_id: Optional[str] = None,
) -> str:
    """Insert a QUERY row and return the query_id."""
    query_id = query_id or str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO QUERY
                (query_id, session_id, query_text, law_domain, similarity_threshold)
            VALUES (?, ?, ?, ?, ?)
            """,
            (query_id, session_id, query_text, law_domain, similarity_threshold),
        )
    return query_id


def insert_response(
    query_id: str,
    response_text: str,
    llm_model: str,
    tokens_used: int = 0,
    is_no_result: bool = False,
    response_id: Optional[str] = None,
) -> str:
    """Insert a RESPONSE row and return the response_id."""
    response_id = response_id or str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO RESPONSE
                (response_id, query_id, response_text, llm_model,
                 tokens_used, is_no_result)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                response_id,
                query_id,
                response_text,
                llm_model,
                tokens_used,
                1 if is_no_result else 0,
            ),
        )
    return response_id


def insert_query_chunks(
    query_id: str,
    chunks: list[dict],
) -> None:
    """Bulk-insert QUERY_CHUNK rows linking a query to its retrieved chunks.

    Each dict in *chunks* must contain: chunk_id, score (float), rank (int).
    """
    rows = [(query_id, c["chunk_id"], c["score"], c["rank"]) for c in chunks]
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO QUERY_CHUNK
                (query_id, chunk_id, similarity_score, rank)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )


# ---------------------------------------------------------------------------
# Initialise schema on import
# ---------------------------------------------------------------------------

init_db()
