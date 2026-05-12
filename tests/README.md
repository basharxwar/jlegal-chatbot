# JLegal-ChatBot Tests

Smoke test suite verifying project structure, database integrity, AraBERT embedding pipeline, vector store, and retrieval.

## Run all tests
```bash
pytest tests/ -v
```

## Run specific test file
```bash
pytest tests/test_retriever.py -v
```

## Test coverage
- Project structure and required files (test_structure.py)
- SQLite schema and chunk counts (test_database.py)
- AraBERTv02 embedding model and dimensions (test_embedder.py)
- JSON vector store and domain files (test_vector_store.py)
- Retrieval pipeline with ranking and threshold filtering (test_retriever.py)

## Prerequisites
The vector store and database must be built before running tests that depend on them:
```bash
python run_ingestion.py
```

Tests that depend on built artifacts will skip with a clear message if those artifacts are not present.

## Notes
Generation tests (calling the Anthropic API) are intentionally excluded from this smoke suite to keep tests fast, deterministic, and free.
