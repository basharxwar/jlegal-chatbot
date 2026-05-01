"""
ingest.py — PDF ingestion pipeline for JLegal-ChatBot.

Responsibilities:
  1. Extract text from a PDF file page-by-page using PyMuPDF.
  2. Detect article numbers via Arabic regex.
  3. Split text into overlapping chunks with LangChain's RecursiveCharacterTextSplitter.
  4. Embed chunks with CamelBERT (AraBERT) for semantic Arabic understanding.
  5. Save chunks + embeddings to a JSON vector store (one file per law domain).
  6. Mirror every chunk row into the SQLite CHUNK table via database.py.
"""

import re
import uuid
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.embedder import embed_texts
from src.database import (
    upsert_document,
    insert_chunk,
    count_chunks_for_document,
    get_document_by_title,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "arabertv02-768"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SEPARATORS = ["\n\n", "\n", "المادة", ".", " ", ""]

# Multiple patterns to handle all formats found in the 5 PDFs:
#   المادة (45):   — Commercial law uses parentheses
#   المادة 45      — plain space
#   المادة-45      — dash separator
#   المادة رقم 45  — explicit "رقم" keyword
_ARTICLE_PATTERNS = [
    re.compile(r"المادة\s*\(?\s*(\d+)\s*\)?\s*[:.\-]"),
    re.compile(r"المادة\s*-?\s*(\d+)"),
    re.compile(r"^\s*\(?\s*(\d+)\s*\)?\s*[-–:]", re.MULTILINE),
    re.compile(r"المادة\s*رقم\s*\(?\s*(\d+)\s*\)?"),
]

LAW_NAMES_AR: dict[str, str] = {
    "Labor": "قانون العمل الأردني",
    "Commercial": "قانون التجارة الأردني",
    "PersonalStatus": "قانون الأحوال الشخصية",
    "Cybercrime": "قانون الجرائم الإلكترونية الأردني",
    "CivilService": "نظام الخدمة المدنية",
}

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
    """Try each pattern in order; return first digit capture found."""
    for pattern in _ARTICLE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def propagate_article_numbers(chunks: list[dict]) -> list[dict]:
    """Carry the last-seen article number forward to continuation chunks.

    Chunks that are mid-article have no 'المادة X' header, so they get
    a blank article_number.  This pass fills them in from the preceding
    header chunk so every chunk in the same article is labelled correctly.
    """
    last_article = None
    for chunk in chunks:
        if chunk.get("article_number"):
            last_article = chunk["article_number"]
        elif last_article:
            chunk["article_number"] = last_article
    return chunks


def filter_meaningful_chunks(chunks: list[dict]) -> list[dict]:
    """Remove chunks that are too short or are just article headers.

    Chunks like 'المادة (328) :' get generic embeddings and pollute
    retrieval results. We keep only chunks with real legal content.
    """
    MIN_LENGTH = 80
    filtered = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if len(text) < MIN_LENGTH:
            continue
        if len(text.split()) < 12:
            continue
        arabic_chars = sum(
            1 for c in text
            if '؀' <= c <= '߿'          # standard Arabic block
            or 'ﭐ' <= c <= '﷿'           # Arabic Presentation Forms-A
            or 'ﹰ' <= c <= '﻿'           # Arabic Presentation Forms-B
        )
        if arabic_chars < 50:
            continue
        filtered.append(chunk)
    return filtered


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_into_chunks(pages: list[dict]) -> list[dict]:
    """Split page texts into overlapping chunks."""
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
    """Embed chunks with AraBERT and save to the JSON vector store."""
    from src.vector_store import save_chunks

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)  # numpy array shape (n_chunks, 768)

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
            "embedding": embeddings[i].tolist(),
        }
        for i, c in enumerate(chunks)
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
    """Ingest a single PDF into the vector store and SQLite.

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

    chunks = propagate_article_numbers(chunks)
    with_art = sum(1 for c in chunks if c.get("article_number"))
    print(f"  Article numbers: {with_art}/{len(chunks)} chunks labelled")

    before_filter = len(chunks)
    chunks = filter_meaningful_chunks(chunks)
    print(f"  After filtering: {len(chunks)}/{before_filter} chunks kept "
          f"({before_filter - len(chunks)} short/empty removed)")

    # Re-index chunk_index after filtering so it stays contiguous
    for idx, chunk in enumerate(chunks):
        chunk["chunk_index"] = idx

    for chunk in chunks:
        chunk["chunk_id"] = str(uuid.uuid4())

    print(f"  Embedding with AraBERT (domain='{law_domain}') ...")
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
