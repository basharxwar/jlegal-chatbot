"""Tests for the SQLite database schema and content."""
import sqlite3
import pytest


@pytest.fixture
def conn(db_path):
    if not db_path.exists():
        pytest.skip("Database not initialized — run run_ingestion.py first")
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    yield c
    c.close()


def test_all_tables_exist(conn):
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    tables = {row["name"] for row in cursor.fetchall()}
    required = {"SESSION", "DOCUMENT", "CHUNK", "QUERY", "RESPONSE", "QUERY_CHUNK"}
    assert required.issubset(tables), f"Missing tables: {required - tables}"


def test_document_count(conn):
    cursor = conn.execute("SELECT COUNT(*) AS cnt FROM DOCUMENT")
    count = cursor.fetchone()["cnt"]
    assert count >= 10, f"Expected at least 10 documents, got {count}"


def test_chunk_count(conn):
    cursor = conn.execute("SELECT COUNT(*) AS cnt FROM CHUNK")
    count = cursor.fetchone()["cnt"]
    assert count >= 2000, f"Expected at least 2000 chunks, got {count}"


def test_law_domains_present(conn):
    cursor = conn.execute("SELECT DISTINCT law_domain FROM DOCUMENT")
    domains = {row["law_domain"] for row in cursor.fetchall()}
    required = {"Labor", "Commercial", "PersonalStatus", "Cybercrime",
                "CivilService", "PenalCode",
                "SocialSecurity", "IncomeTax", "Companies", "Constitution",
                "ConsumerProtection", "Customs"}
    assert required.issubset(domains), f"Missing domains: {required - domains}"


def test_chunks_have_article_numbers(conn):
    cursor = conn.execute(
        "SELECT COUNT(*) AS cnt FROM CHUNK WHERE article_number IS NOT NULL"
    )
    count = cursor.fetchone()["cnt"]
    assert count > 0, "No chunks have article numbers — article detection failed"
