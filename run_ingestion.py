"""
run_ingestion.py — Standalone script to ingest all 5 Jordanian law PDFs.

Run this once before starting the Streamlit app:
    python run_ingestion.py

Re-running is safe: already-ingested documents are skipped unless you
pass --force on the command line.

Two-phase flow
--------------
Phase 1 — Fit vectorizer:
    Extract and chunk ALL PDFs, combine every chunk text, fit the TF-IDF
    vectorizer on the full corpus, and save it to tfidf_model.joblib.
    This ensures query embeddings and stored embeddings share one vocabulary.

Phase 2 — Ingest:
    For each PDF, embed its chunks using the fitted vectorizer and store
    them in ChromaDB + SQLite.  Already-ingested PDFs are skipped.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingest import (  # noqa: E402
    extract_pages,
    split_into_chunks,
    fit_and_save_vectorizer,
    ingest_pdf,
)

# ---------------------------------------------------------------------------
# PDF → law domain mapping
# ---------------------------------------------------------------------------

PDF_DIR = Path(__file__).resolve().parent

PDF_MAP: list[tuple[str, str]] = [
    ("Labor_Law_Jordan.pdf",          "Labor"),
    ("Jordanian trade law.pdf",       "Commercial"),
    ("Personal status law.pdf",       "PersonalStatus"),
    ("Jordanian Cybercrimes Law.pdf", "Cybercrime"),
    ("Civil Service System Law.pdf",  "CivilService"),
]

DOMAIN_LABELS: dict[str, str] = {
    "Labor":         "Labor",
    "Commercial":    "Commercial",
    "PersonalStatus":"Personal Status",
    "Cybercrime":    "Cybercrime",
    "CivilService":  "Civil Service",
}


def main(force: bool = False) -> None:
    print("=" * 60)
    print("JLegal-ChatBot — PDF Ingestion Pipeline")
    print("=" * 60)

    # Identify which PDFs are present
    available: list[tuple[Path, str]] = []
    missing: list[str] = []
    for filename, law_domain in PDF_MAP:
        pdf_path = PDF_DIR / filename
        if pdf_path.exists():
            available.append((pdf_path, law_domain))
        else:
            print(f"\n[WARNING] File not found: {filename}")
            missing.append(filename)

    if not available:
        print("\n[ERROR] No PDF files found. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 1: Collect all chunks from every available PDF and fit TF-IDF
    # ------------------------------------------------------------------
    print("\n--- Phase 1: Fitting TF-IDF vectorizer on full corpus ---")
    all_texts: list[str] = []
    for pdf_path, _ in available:
        print(f"  Reading '{pdf_path.name}' ...")
        pages = extract_pages(pdf_path)
        chunks = split_into_chunks(pages)
        all_texts.extend(c["text"] for c in chunks)

    print(f"  Total chunks across all PDFs: {len(all_texts)}")
    fit_and_save_vectorizer(all_texts)

    # ------------------------------------------------------------------
    # Phase 2: Ingest each PDF (embed + store in ChromaDB + SQLite)
    # ------------------------------------------------------------------
    print("\n--- Phase 2: Ingesting PDFs ---")
    summary: dict[str, int] = {}

    for pdf_path, law_domain in available:
        domain_label = DOMAIN_LABELS[law_domain]
        print(f"\nIngesting: {pdf_path.name} → {domain_label} domain ...")
        try:
            n_chunks = ingest_pdf(pdf_path=pdf_path, law_domain=law_domain, force=force)
            summary[domain_label] = n_chunks
        except Exception as exc:
            print(f"  [ERROR] Failed to ingest '{pdf_path.name}': {exc}")
            summary[domain_label] = -1

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Ingestion Summary")
    print("=" * 60)
    for domain, count in summary.items():
        if count > 0:
            status = f"{count:>5} chunks created"
        elif count == 0:
            status = "       skipped (already ingested)"
        else:
            status = "       FAILED"
        print(f"  {domain:<20} {status}")

    if missing:
        print("\nMissing files:")
        for f in missing:
            print(f"  - {f}")

    print("\nDone. Run 'streamlit run app.py' to start the chatbot.")


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    if force_flag:
        print("[INFO] --force flag detected: re-ingesting all documents.")
    main(force=force_flag)
