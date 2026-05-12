"""Tests for the JSON-based vector store."""
import pytest


EXPECTED_DOMAINS = [
    "Labor", "Commercial", "PersonalStatus", "Cybercrime",
    "CivilService", "PenalCode",
]


def test_vector_store_directory_exists(vector_store_dir):
    if not vector_store_dir.exists():
        pytest.skip("Vector store not built — run run_ingestion.py first")
    assert vector_store_dir.is_dir()


def test_all_domains_have_json_files(vector_store_dir):
    if not vector_store_dir.exists():
        pytest.skip("Vector store not built")
    for domain in EXPECTED_DOMAINS:
        path = vector_store_dir / f"{domain}.json"
        assert path.exists(), f"Missing vector store for {domain}"


def test_collection_loading_works():
    from src.vector_store import collection_exists, collection_count
    if not collection_exists("Labor"):
        pytest.skip("Labor collection not built")
    count = collection_count("Labor")
    assert count > 0, "Labor collection is empty"
