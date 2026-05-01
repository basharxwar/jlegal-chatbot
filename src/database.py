"""
database.py — SQLite schema creation and all CRUD operations for JLegal-ChatBot.

All table definitions match the exact schema specified in the project requirements.
WAL mode and foreign-key enforcement are enabled on every connection.
All foreign keys use ON DELETE CASCADE so a reset is safe and clean.
"""

import sqlite3
import uuid
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
# Schema DDL
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
    law_domain     TEXT NOT NULL,
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
    similarity_threshold REAL NOT NULL DEFAULT 0.05,
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
    FOREIGN KEY (query_id) REFERENCES QUERY(query_id)  ON DELETE CASCADE,
    FOREIGN KEY (chunk_id) REFERENCES CHUNK(chunk_id)  ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_query_session  ON QUERY(session_id);
CREATE INDEX IF NOT EXISTS idx_chunk_document ON CHUNK(document_id);
CREATE INDEX IF NOT EXISTS idx_qc_query       ON QUERY_CHUNK(query_id);
"""


def init_db() -> None:
    """Create all tables and indexes if they do not already exist."""
    with get_connection() as conn:
        conn.executescript(_DDL)
        # Migration: add display_name to SESSION if not present (safe for existing DBs)
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(SESSION)").fetchall()]
        if "display_name" not in cols:
            conn.execute("ALTER TABLE SESSION ADD COLUMN display_name TEXT DEFAULT NULL")


def reset_db() -> None:
    """Drop all tables and recreate from scratch.

    Used after a --force re-ingestion with a new embedding model so that
    chunk UUIDs in SQLite match those in the JSON vector store exactly.
    Disables FK enforcement during the drop so order does not matter.
    """
    drop_script = """
    PRAGMA foreign_keys = OFF;
    DROP TABLE IF EXISTS QUERY_CHUNK;
    DROP TABLE IF EXISTS RESPONSE;
    DROP TABLE IF EXISTS QUERY;
    DROP TABLE IF EXISTS CHUNK;
    DROP TABLE IF EXISTS DOCUMENT;
    DROP TABLE IF EXISTS SESSION;
    PRAGMA foreign_keys = ON;
    """
    with get_connection() as conn:
        conn.executescript(drop_script)
    init_db()


# ---------------------------------------------------------------------------
# SESSION CRUD
# ---------------------------------------------------------------------------

def create_session(
    session_id: Optional[str] = None,
    user_identifier: Optional[str] = None,
    language: str = "ar",
) -> str:
    """Insert a new session row and return the session_id. Safe to call twice."""
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
    """Insert a document row; silently ignore if already present."""
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
    """Upsert a chunk row.

    Uses INSERT OR REPLACE so that a --force re-ingestion with new chunk UUIDs
    correctly replaces old rows (via the UNIQUE document_id/chunk_index constraint)
    instead of silently keeping stale UUIDs that would break FK references.
    The ON DELETE CASCADE on QUERY_CHUNK ensures any old logging rows are cleaned up.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO CHUNK
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
    similarity_threshold: float = 0.05,
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

    Each dict must contain: chunk_id, score (float), rank (int).

    Filters out any chunk_ids that are not present in the CHUNK table before
    inserting so we never hit a FK violation from stale references.
    """
    if not chunks:
        return

    # Resolve only chunk_ids that actually exist in the CHUNK table.
    # This guards against any UUID mismatch between the vector store and SQLite.
    chunk_ids = [c["chunk_id"] for c in chunks]
    placeholders = ",".join("?" * len(chunk_ids))

    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT chunk_id FROM CHUNK WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        valid_ids = {r["chunk_id"] for r in rows}

        valid_rows = [
            (query_id, c["chunk_id"], c["score"], c["rank"])
            for c in chunks
            if c["chunk_id"] in valid_ids
        ]

        if valid_rows:
            conn.executemany(
                """
                INSERT OR IGNORE INTO QUERY_CHUNK
                    (query_id, chunk_id, similarity_score, rank)
                VALUES (?, ?, ?, ?)
                """,
                valid_rows,
            )


def get_chunk_count() -> int:
    """Return total number of chunks in the database (used for status checks)."""
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM CHUNK").fetchone()
    return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# Session history CRUD (for chat history sidebar)
# ---------------------------------------------------------------------------

def list_sessions(limit: int = 50) -> list[dict]:
    """Return recent sessions that have at least one query, newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                s.session_id,
                s.display_name,
                s.created_at,
                s.last_active_at,
                (SELECT q.query_text FROM QUERY q
                 WHERE q.session_id = s.session_id
                 ORDER BY q.created_at ASC LIMIT 1) AS first_query
            FROM SESSION s
            WHERE EXISTS (SELECT 1 FROM QUERY q WHERE q.session_id = s.session_id)
            ORDER BY s.last_active_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def rename_session(session_id: str, new_name: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE SESSION SET display_name = ? WHERE session_id = ?",
            (new_name.strip()[:100], session_id),
        )


def delete_session(session_id: str) -> None:
    """Delete a session and all its queries/responses via CASCADE."""
    with get_connection() as conn:
        conn.execute("DELETE FROM SESSION WHERE session_id = ?", (session_id,))


def load_session_messages(session_id: str) -> list[dict]:
    """Reconstruct the messages list for a session from QUERY + RESPONSE tables."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                q.query_id, q.query_text,
                r.response_text, r.is_no_result
            FROM QUERY q
            LEFT JOIN RESPONSE r ON r.query_id = q.query_id
            WHERE q.session_id = ?
            ORDER BY q.created_at ASC
            """,
            (session_id,),
        ).fetchall()

    messages = []
    for row in rows:
        messages.append({
            "role": "user",
            "content": row["query_text"],
            "attachments": [],
        })
        if row["response_text"]:
            messages.append({
                "role": "assistant",
                "content": row["response_text"],
                "chunks": [],        # not reloaded — too expensive
                "question": row["query_text"],
                "query_id": row["query_id"],
                "attachments": [],
                "confidence": {},    # historical sessions don't store confidence
            })
    return messages


# ---------------------------------------------------------------------------
# Initialise schema on import
# ---------------------------------------------------------------------------

init_db()
