"""
ingest.py — PDF ingestion pipeline for JLegal-ChatBot.

Responsibilities:
  1. Extract text from a PDF file page-by-page using PyMuPDF.
  2. Detect article numbers via Arabic regex.
  3. Split text into overlapping chunks with LangChain's RecursiveCharacterTextSplitter.
  4. Embed chunks with a TF-IDF vectorizer (sklearn) — zero torch/onnxruntime dependency.
  5. Persist chunks to a named ChromaDB collection (one collection per law domain).
  6. Mirror every chunk row into the SQLite CHUNK table via database.py.

Embedding strategy
------------------
We use sklearn's TfidfVectorizer with character n-grams (2-4), which handles Arabic
morphology well without any Arabic-specific tokenizer.  The fitted vectorizer is saved
to TFIDF_MODEL_PATH so retriever.py can load it and embed queries with the same
vocabulary.

IMPORTANT: fit_and_save_vectorizer() must be called with ALL document texts before
any ingest_pdf() call.  run_ingestion.py enforces this order.
"""

import re
import uuid
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import joblib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

from src.database import (
    upsert_document,
    insert_chunk,
    count_chunks_for_document,
    get_document_by_title,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "tfidf-char-ngram-4096"
TFIDF_MODEL_PATH = Path(__file__).resolve().parent.parent / "tfidf_model.joblib"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SEPARATORS = ["\n\n", "\n", "المادة", ".", " ", ""]

ARTICLE_PATTERN = re.compile(r"المادة\s*-?\s*(\d+)")

LAW_NAMES_AR: dict[str, str] = {
    "Labor": "قانون العمل الأردني",
    "Commercial": "قانون التجارة الأردني",
    "PersonalStatus": "قانون الأحوال الشخصية",
    "Cybercrime": "قانون الجرائم الإلكترونية الأردني",
    "CivilService": "نظام الخدمة المدنية",
}

# ---------------------------------------------------------------------------
# TF-IDF vectorizer singleton
# ---------------------------------------------------------------------------

_vectorizer: Optional[TfidfVectorizer] = None


def get_vectorizer() -> TfidfVectorizer:
    """Return the fitted TF-IDF vectorizer, loading from disk if needed."""
    global _vectorizer
    if _vectorizer is not None:
        return _vectorizer
    if TFIDF_MODEL_PATH.exists():
        _vectorizer = joblib.load(TFIDF_MODEL_PATH)
        return _vectorizer
    raise RuntimeError(
        f"TF-IDF model not found at {TFIDF_MODEL_PATH}. "
        "Run python run_ingestion.py first to fit and save the vectorizer."
    )


def fit_and_save_vectorizer(texts: list[str]) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on all provided texts and save to disk.

    Must be called with the combined text of ALL documents before ingestion
    so that query embeddings and stored embeddings share the same vocabulary.
    """
    global _vectorizer
    print(f"  Fitting TF-IDF on {len(texts)} chunks ...")
    _vectorizer = TfidfVectorizer(
        analyzer="char_wb",   # character n-grams within word boundaries
        ngram_range=(2, 4),   # captures Arabic root patterns
        max_features=4096,
        sublinear_tf=True,    # log(tf) + 1 — reduces impact of very frequent terms
        strip_accents=None,   # preserve Arabic diacritics
    )
    _vectorizer.fit(texts)
    joblib.dump(_vectorizer, TFIDF_MODEL_PATH)
    print(f"  Vectorizer saved → {TFIDF_MODEL_PATH}")
    return _vectorizer


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings using the fitted TF-IDF vectorizer."""
    vec = get_vectorizer()
    sparse = vec.transform(texts)
    return sparse.toarray().tolist()


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text from every page of a PDF. Returns [{page_number, text}]."""
    pages: list[dict] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append({"page_number": page_index, "text": text})
    return pages


def detect_article_number(text: str) -> Optional[str]:
    match = ARTICLE_PATTERN.search(text)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_into_chunks(pages: list[dict]) -> list[dict]:
    """Split page texts into overlapping chunks. Returns [{text, page_number, article_number, chunk_index}]."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    raw_chunks: list[dict] = []
    for page in pages:
        sub_texts = splitter.split_text(page["text"])
        for sub_text in sub_texts:
            raw_chunks.append(
                {
                    "text": sub_text,
                    "page_number": page["page_number"],
                    "article_number": detect_article_number(sub_text),
                }
            )

    for idx, chunk in enumerate(raw_chunks):
        chunk["chunk_index"] = idx

    return raw_chunks


# ---------------------------------------------------------------------------
# Vector store storage
# ---------------------------------------------------------------------------

def store_in_vector_store(
    chunks: list[dict],
    document_id: str,
    law_domain: str,
    law_name_ar: str,
) -> None:
    """Embed chunks with TF-IDF and save to the JSON vector store."""
    from src.vector_store import save_chunks

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    records = [
        {
            "chunk_id": c["chunk_id"],
            "document_id": document_id,
            "law_domain": law_domain,
            "law_name_ar": law_name_ar,
            "chunk_text": c["text"],
            "article_number": c["article_number"] or "",
            "page_number": c["page_number"],
            "chunk_index": c["chunk_index"],
            "embedding": emb,
        }
        for c, emb in zip(chunks, embeddings)
    ]
    save_chunks(records, law_domain)


# ---------------------------------------------------------------------------
# SQLite storage
# ---------------------------------------------------------------------------

def store_in_sqlite(chunks: list[dict], document_id: str) -> None:
    for chunk in chunks:
        insert_chunk(
            chunk_id=chunk["chunk_id"],
            document_id=document_id,
            chunk_index=chunk["chunk_index"],
            chunk_text=chunk["text"],
            embedding_model=EMBEDDING_MODEL_NAME,
            article_number=chunk.get("article_number"),
            page_number=chunk.get("page_number"),
            token_count=len(chunk["text"].split()),
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: Path, law_domain: str, force: bool = False) -> int:
    """Ingest a single PDF into ChromaDB and SQLite.

    Requires fit_and_save_vectorizer() to have been called first.

    Returns number of chunks created (0 if skipped).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    title = pdf_path.stem
    law_name_ar = LAW_NAMES_AR.get(law_domain, title)

    existing = get_document_by_title(title)
    if existing and not force:
        existing_chunks = count_chunks_for_document(existing["document_id"])
        if existing_chunks > 0:
            print(
                f"  [skip] '{title}' already ingested "
                f"({existing_chunks} chunks). Use --force to re-ingest."
            )
            return 0

    document_id = existing["document_id"] if existing else str(uuid.uuid4())
    upsert_document(
        document_id=document_id,
        title=title,
        law_domain=law_domain,
        language="ar",
    )

    print(f"  Extracting text from '{pdf_path.name}' ...")
    pages = extract_pages(pdf_path)
    print(f"  Pages extracted: {len(pages)}")

    chunks = split_into_chunks(pages)
    print(f"  Chunks created:  {len(chunks)}")

    for chunk in chunks:
        chunk["chunk_id"] = str(uuid.uuid4())

    print(f"  Embedding and storing in vector store (domain='{law_domain}') ...")
    store_in_vector_store(
        chunks=chunks,
        document_id=document_id,
        law_domain=law_domain,
        law_name_ar=law_name_ar,
    )

    print(f"  Storing {len(chunks)} chunks in SQLite ...")
    store_in_sqlite(chunks=chunks, document_id=document_id)

    print(f"  Done. {len(chunks)} chunks ingested for '{title}'.")
    return len(chunks)
