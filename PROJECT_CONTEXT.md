# JLegal-ChatBot — Project Context for AI Assistants

> Read this file first before helping with any task.

## Project Overview
- Arabic Legal RAG Chatbot for Jordanian laws
- Yarmouk University, Faculty of IT, Department of Data Science and AI
- Team: Bashar Wardat, Abdulrahman Khasawneh, Furat Al-Ahmad
- Supervisor: Dr. Ali Malkawi
- GitHub: github.com/basharxwar/jlegal-chatbot

## Current System State (v10)
- Embedding: AraBERTv02 with mean pooling
- Retrieval: numpy cosine similarity on JSON vector store
- Generation: Claude API (claude-sonnet-4-20250514)
- Query expansion: Claude Haiku rewrites colloquial to MSA
- Database: SQLite with 6 tables in 3NF
- UI: Streamlit with Arabic RTL support, voice input, dialect toggle

## Corpus
9 Jordanian laws, 1567 chunks total:
- Labor Law (179)
- Commercial Law (404)
- Personal Status (190)
- Cybercrime (64)
- Civil Service (71)
- Civil Status (69)
- Personal Status 2019 (209)
- Traffic Law (108) - article numbers blank due to ligature PDF
- HR Management System (273)
- Social Security NOT included - scanned PDF, no text layer

## Key Files
- src/ingest.py - PDF reading, chunking, embedding, storage
- src/retriever.py - query embedding, cosine search, top-k retrieval
- src/embedder.py - AraBERTv02 model wrapper
- src/vector_store.py - JSON-based vector storage
- src/generator.py - Claude API call, system prompts (formal + Jordanian)
- src/pipeline.py - orchestration: validate → retrieve → generate → log
- src/database.py - SQLite schema and CRUD
- app.py - Streamlit UI with voice input, dialect toggle, examples
- run_ingestion.py - processes all 9 PDFs

## Evolution History
- v1: TF-IDF + pure Python (Windows DLL workaround)
- v2: camelbert-mix (failed, flat scores)
- v3: AraBERTv02 + mean pooling (works for MSA)
- v4: MARBERT experiment (failed, embeddings collapse)
- v5: Article regex fix (40% → 99.8% coverage) + chunk filter (38 removed)
- v6: Voice + dialect toggle + UI polish
- v7: Speed optimization (model caching) + query expansion
- v8: Corpus expansion (5 → 9 laws)
- v9: Performance caching
- v10: UI overhaul with navy+gold theme, English filenames

## Critical Engineering Decisions
1. ChromaDB replaced with numpy/JSON: Windows + Python 3.12 DLL incompatibility
2. TF-IDF upgraded to AraBERTv02: dialectal queries needed semantic search
3. MARBERT rejected: collapses to 0.98+ for all queries on legal text
4. Query expansion via Claude Haiku: cheap fix for colloquial→formal translation
5. Mean pooling on AraBERTv02: required because base BERT does not output sentence embeddings

## Configuration Defaults
- DEFAULT_THRESHOLD = 0.50
- DEFAULT_TOP_K = 8
- _PER_DOMAIN_K = 4
- AraBERT model: aubmindlab/bert-base-arabertv02

## Things NOT to Modify Without Discussion
- src/embedder.py (model selection — three swaps already tested)
- src/vector_store.py (cosine similarity logic)
- vector_store/*.json (would require re-ingestion)
- The article number regex (took 3 iterations to get right)

## Active Defense Story for Committee
- Anti-hallucination: minimum wage question correctly refused
- Three-model comparison: camelbert-mix (failed), AraBERTv02 (works), MARBERT (failed)
- Engineering solution: replaced ChromaDB with custom numpy implementation
- Hybrid LLM-search: query expansion uses LLM to help search, then search results help LLM
