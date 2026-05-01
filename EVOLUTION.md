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

## v11.1 — Surgical fix pack (defense prep)

1. **Voice input restored** (`app.py`): replaced bare `except` with structured error logging; surfaced real exceptions instead of generic "غير متاح"; Whisper loader now returns None on failure instead of raising.
2. **System prompt rewrite** (`src/generator.py`): both FORMAL and JORDANIAN prompts rewritten to encourage confident synthesis from retrieved articles; added explicit structure examples and forbidden-behavior list.
3. **Sidebar reorganization** (`app.py`): compact logo, horizontal style radio, stats/status collapsed into expander. Fits 1080p without scrolling.
4. **Cleanup** (`app.py`): Arabic-only domain dropdown labels; replaced broken Traffic Law example with dismissal-without-notice example.
5. **Source display threshold** (`src/pipeline.py`): `DISPLAY_THRESHOLD = 0.65` — retrieval still passes all 0.50+ chunks to Claude, but only ≥0.65 shown as user-visible sources. Prevents fake citations on casual greetings.

Not changed: embeddings, vector store, retrieval logic, database schema, ingestion pipeline.

## v11 — Pre-Defense Fixes
Bug #1: Spinner nested in st.chat_message("assistant") cleared container on exit (Streamlit 1.5x).
Fix: moved spinner to run between messages — assistant bubble now renders on first script run.
Bug #2: Full navy #1B3A57 theme applied to main chat area, expanders, and chat input.
Bug #3: Multimodal attachments — st.chat_input accepts PNG/JPG/WebP (5 MB) and PDF (10 MB).
Images sent as base64 content blocks to Claude vision API.
PDFs extracted via PyMuPDF and appended to retrieval query (original kept for audit log).
pipeline.run_query: added images and pdf_texts parameters.
generator.generate_answer: added images parameter with magic-byte MIME detection.

## v10 — Final Polish
PDF files renamed to English slugs (labor.pdf, commercial.pdf, etc.)
Language detection for no-result messages (Arabic/English/Jordanian)
UI overhaul: navy #1B3A57 + gold #C9A961 + teal #2D7D8E palette
Gradient sidebar, header block with stats badges
3-column example buttons with emoji icons
Source cards redesigned with prominent article numbers
run_ingestion.py restructured with LAWS list format
