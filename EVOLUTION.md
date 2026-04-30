# JLegal-ChatBot — System Evolution Log

## v1 — TF-IDF + Pure Python
Reason: Windows + Python 3.12 + ChromaDB DLL incompatibility
Result: Works for formal MSA, fails on colloquial
Embedding: sklearn TfidfVectorizer char n-grams (2-4), 4096 features
Storage: JSON vector store with numpy cosine similarity

## v2 — camelbert-mix (first neural attempt)
Model: CAMeL-Lab/bert-base-arabic-camelbert-mix
Result: Failed — flat 0.70-0.78 scores with no discrimination between queries
Finding: Model not suited for legal domain retrieval

## v3 — AraBERTv02 + mean pooling
Model: aubmindlab/bert-base-arabertv02
Result: Works for formal MSA queries with scores 0.62-0.81
Key fix: Explicit mean pooling via sentence-transformers pipeline (base BERT has no pooling head)

## v4 — MARBERT experiment
Model: UBC-NLP/MARBERT
Result: Failed — collapses to 0.98-0.99 for ALL queries regardless of content
Finding: Social media training does not transfer to legal document text
Decision: Rolled back to AraBERTv02 in <30 seconds using backup vector store

## v5 — Article regex fix + chunk filter
Article number coverage: 40% → 99.8% (379/946 → 944/946 chunks labelled)
Root cause: PDF used (المادة (45): format, regex only handled المادة 45
Fix: 4-pattern regex cascade + propagate_article_numbers() forward pass
Chunk filter: Removed 38 short/empty chunks (pure article headers, punctuation)
Chunk minimum: 80 chars, 12 words, 50 Arabic characters

## v6 — Voice + Dialect + UI Polish
Whisper STT via streamlit-mic-recorder
Jordanian dialect toggle (SYSTEM_PROMPT_JORDANIAN)
Example questions panel
👍/👎 feedback buttons with query_id keying
PDF download button for each answer

## v7 — Speed + Query Expansion
Cached AraBERT model at startup via @st.cache_resource
Cached all vector store domains into RAM at startup
Query expansion: Claude Haiku rewrites dialect query to formal MSA
Dialectal query accuracy: 0/8 → 4/8 Labor Law hits (scores 0.73-0.75 vs 0.46-0.50)

## v8 — Corpus Expansion (5 → 9 laws)
Added: Civil Status, Personal Status 2019, Traffic Law, HR Management
Excluded: Social Security (scanned image PDF, no text layer)
Total chunks: 908 → 1567
Unicode fix: NFC normalization for Arabic filenames on Windows NTFS
Filter fix: Extended Arabic range to include Presentation Forms (FB50-FEFF) for Traffic Law

## v9 — Performance Caching
Sub-second query response after first load
preload_all_collections() warms all 9 domain JSON files at startup
Model loading moved entirely to startup phase

## v10 — Final Polish
PDF files renamed to English slugs (labor.pdf, commercial.pdf, etc.)
Language detection for no-result messages (Arabic/English/Jordanian)
UI overhaul: navy #1B3A57 + gold #C9A961 + teal #2D7D8E palette
Gradient sidebar, header block with stats badges
3-column example buttons with emoji icons
Source cards redesigned with prominent article numbers
run_ingestion.py restructured with LAWS list format
