# JLegal-ChatBot — System Evolution Log

## v12.5 — PDF path fix, button outside try, voice spinner width

**Goal:** Three surgical fixes to `app.py` only — no logic changes.

1. **PDF path resolution** (`_generate_pdf`) — replaced `Path(__file__).resolve().parent`
   with `os.path.dirname(os.path.abspath(__file__))`. Streamlit may change the working
   directory at startup, making `resolve()` return a wrong path so `font_path.exists()`
   returned False and the Amiri font was never loaded. `os.path.abspath` anchors to the
   file's real location regardless of cwd.

2. **`LetterheadPDF` constructor injection** — eliminated closure-captured `_mf` variable.
   `LetterheadPDF` now accepts `(use_ar, fp, lp, ar_fn)` as constructor arguments stored
   as instance attributes. `header()` and `footer()` read `self._use_ar` / `self._f()`
   instead of a mutable outer variable. The font is registered inside `__init__` on the
   same instance that will call `add_page()`, so no race between font registration and
   `header()` execution.

3. **PDF button outside try block** — in both `_render_assistant_message` (history replay)
   and the new-message block, the pattern is now: generate bytes first into `_pdf_bytes`,
   then render `st.download_button` only if bytes are not None, else show caption. This
   prevents Streamlit from swallowing the button when `_generate_pdf` raises mid-way.

4. **Voice transcription spinner at full width** — the `st.spinner("تحويل الصوت...")`
   block and all transcription logic moved outside the `with voice_cols[1]:` context.
   Only the `mic_recorder` widget stays inside the narrow [20,1] column; processing
   renders at full page width so the spinner is visible instead of stacking vertically.

**Not changed:** AraBERT embedder, vector store, retrieval logic, ingestion pipeline,
database schema, corpus, color palette, system prompts.

**Status:** Code-complete. v12.5-stable is the defense version.

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

## v12.4 — PDF button final fix

**Goal:** Fix the recurring PDF button bug that survived v12, v12.1, v12.2, v12.3.

**Root cause:** `_mf` was initialized to `"Helvetica"` before `LetterheadPDF` was
defined. The class captured `_mf` by closure. Then `add_font` was called on the
real `pdf` instance AFTER `LetterheadPDF()` was instantiated, and `add_page()`
(which triggers `header()`) was called immediately after — before `_mf` was
updated to `"Arabic"`. So `header()` always saw `_mf = "Helvetica"`, and every
Arabic string in the header threw `FPDFUnicodeEncodingException`.

**Fix:** Two-step pattern:
1. Probe font loading on a throwaway FPDF instance. Set `_mf = "Arabic"` ONLY
   if `add_font()` succeeds without exception.
2. Build `LetterheadPDF` after `_mf` is finalized. On the real `pdf` instance,
   re-register the font (each FPDF instance needs its own registration), then
   call `add_page()` — `header()` now reads the correct `_mf`.

**Status:** Code-complete. v12.4-stable is the defense version. No more code
changes before defense.

## v12.3 — Cleanup + targeted fixes

**Goal:** Fix recurring PDF bug, remove confidence bar, optimize chunking, clean junk files.

1. **PDF button works** — root cause: `_mf := "Arabic"` walrus operator set `_mf` before `add_font` succeeded; if `add_font` raised, `header()` called `set_font("Arabic")` with unregistered font. Fixed by assigning `_mf = "Arabic"` only after `add_font` returns successfully. Errors now surface as visible `st.caption` instead of silent `except: pass`.
2. **Confidence card removed** — user preferred cleaner UI. Pipeline still computes and stores the score for potential future use; only rendering is removed.
3. **Article-boundary chunking** — `SEPARATORS` now prioritizes `\nالمادة` and `المادة` before `\n\n`. Chunks prefer to break at legal article boundaries. Chunk counts increased (e.g. Labor 179→212, Penal Code 529→625) confirming better segmentation.
4. **Project cleanup** — deleted: `traffic2.pdf`, `tfidf_model.joblib`, `diagnose_retrieval.py`, 3 diagnosis text files, `Amiri.zip`, 44MB Social Security PDF, `Arabic_Legal_RAG/`, `Amiri/` folders, 5 unrelated files from `assets/`. Only `yarmouk_logo.png` remains in `assets/`.

**Corpus after re-ingestion (10 laws, 2421 chunks):**
Labor 212, Commercial 460, PersonalStatus 215, PersonalStatus2019 209,
Cybercrime 73, CivilService 81, CivilStatus 75, HRManagement 317,
TrafficLaw 154, PenalCode 625

**Not changed:** AraBERT embedder, vector store schema, retrieval logic, generator, database, color palette, sidebar layout.

**Status:** Code-complete. v12.3-stable is the defense version.

## v12.2 — Corpus expansion + PDF restoration

**Goal:** Add Penal Code, fix PDF download button, improve letterhead.

1. **PDF download button restored** — root cause found: `fontTools` not installed, causing every `_generate_pdf` call to throw silently. Installed `fontTools`, rewrote function cleanly.
2. **Emoji stripping** — emojis removed from question/answer text before PDF rendering. Eliminates Amiri font missing-glyph warnings.
3. **PDF letterhead** — Yarmouk University logo (assets/yarmouk_logo.png) added. Formal header: project name, university, date. Per-page disclaimer footer. Duplicate `add_font()` bug fixed.
4. **Penal Code added** — قانون العقوبات الأردني (1367KB, 129 pages, clean Unicode). 529 chunks, 100% article number coverage. Theft, assault, fraud, criminal questions now answered.
5. **Corpus: 10 laws, 2096 chunks total** (was 1567).
6. **Traffic Law replacement failed** — new traffic.pdf still garbled (different encoding, Tifinagh-range codepoints). Traffic limitation updated in KNOWN_LIMITATIONS.md.

**Not changed:** AraBERT embedder, vector store format, retrieval logic, ingestion pipeline, UI color palette, confidence score, chat history.

## v12.1 — Final polish patch

**Goal:** Last UI refinements before defense. No new features, only fixes and arrangement.

1. **Restored PDF download button** below every assistant answer (lost in v12 rewrite).
2. **Moved confidence bar** from above answer to below answer for better reading flow.
3. **Reorganized sidebar order** — response style and domain filter moved above chat history (more frequently used controls promoted).
4. **Replaced radio toggle** with segmented control for cleaner appearance (no red circle dot).
5. **Compact voice button** right-aligned above chat input as small circular icon. Removed sidebar duplicate.
6. **About expander** in sidebar with project info, tech stack, corpus list, and limitations.
7. **Footer credit** at sidebar bottom.

**Not changed:** AraBERT model, vector store, retrieval logic, ingestion, color palette, system prompts.

**Status:** Code-complete. v12.1-stable is the defense version.

## v12 — Polish pack (final code change before defense)

**Goal:** Professional-grade polish for defense demo and committee review.

1. **Confidence Score** — every answer now shows a top-3-average similarity percentage with color-coded label (مرتفعة/متوسطة/منخفضة) and visual bar. Computed in pipeline, rendered above answer.
2. **PDF letterhead** — consultations now have a formal header (project name, university, date), per-page footer with legal disclaimer, page numbers. No team credits. Used FPDF2 subclass with header()/footer() methods.
3. **Voice button inline** — moved from sidebar to small column directly above chat input (reliable fallback approach).
4. **Avatars** — ⚖️ for assistant, 👤 for user via st.chat_message(avatar=...).
5. **Sidebar collapsed by default** — initial_sidebar_state="collapsed"; toggle button opens it.
6. **Chat history with rename** — past sessions listed in sidebar, click to reload messages from SQLite QUERY+RESPONSE tables. Edit (✏) to rename, delete (🗑) to remove. display_name column added to SESSION via migration.
7-9. **Visual redesign** — dark navy/blue palette (#1A2332 base, #3B82F6 accent), gray assistant bubbles, blue user bubbles, refined chat-app aesthetic.

**Not changed:** AraBERT embedder, vector store, retrieval logic, ingestion pipeline, corpus.

**Defense-relevant features:**
- Confidence Score answers "how does the system know when it's right?"
- Anti-hallucination behavior (v11) preserved
- Cross-law reasoning (v8) preserved
- Voice input (v6) preserved
- Jordanian dialect toggle (v6) preserved

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
