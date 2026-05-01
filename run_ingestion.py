"""
run_ingestion.py — Ingest all Jordanian law PDFs into the vector store.

Run once before starting the app:
    python run_ingestion.py

Re-running is safe: already-ingested documents are skipped unless you
pass --force on the command line.
"""

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.ingest import ingest_pdf  # noqa: E402

# ---------------------------------------------------------------------------
# Law definitions — English filenames only
# ---------------------------------------------------------------------------

PDF_DIR = Path(__file__).resolve().parent

LAWS: list[dict] = [
    {"file": "labor.pdf",               "domain": "Labor",              "label": "Labor"},
    {"file": "commercial.pdf",          "domain": "Commercial",         "label": "Commercial"},
    {"file": "personal_status.pdf",     "domain": "PersonalStatus",     "label": "Personal Status"},
    {"file": "cybercrime.pdf",          "domain": "Cybercrime",         "label": "Cybercrime"},
    {"file": "civil_service.pdf",       "domain": "CivilService",       "label": "Civil Service"},
    {"file": "civil_status.pdf",        "domain": "CivilStatus",        "label": "Civil Status"},
    {"file": "personal_status_2019.pdf","domain": "PersonalStatus2019", "label": "Personal Status 2019"},
    {"file": "traffic.pdf",             "domain": "TrafficLaw",         "label": "Traffic Law"},
    {"file": "hr_system.pdf",           "domain": "HRManagement",       "label": "HR Management"},
]


def main(force: bool = False) -> None:
    print("=" * 60)
    print("JLegal-ChatBot — PDF Ingestion Pipeline (AraBERTv02)")
    print("=" * 60)

    # Resolve each PDF path (glob fallback handles Unicode normalization edge cases)
    all_pdfs = {unicodedata.normalize("NFC", f.name): f for f in PDF_DIR.glob("*.pdf")}

    available: list[tuple[Path, str, str]] = []
    missing: list[str] = []

    for law in LAWS:
        filename = law["file"]
        pdf_path = PDF_DIR / filename
        if pdf_path.exists():
            available.append((pdf_path, law["domain"], law["label"]))
        elif unicodedata.normalize("NFC", filename) in all_pdfs:
            available.append((all_pdfs[unicodedata.normalize("NFC", filename)], law["domain"], law["label"]))
        else:
            print(f"\n[WARNING] File not found: {filename}")
            missing.append(filename)

    if not available:
        print("\n[ERROR] No PDF files found. Aborting.")
        sys.exit(1)

    print(f"\nFound {len(available)} PDF(s). Loading AraBERT model...")
    from src.embedder import get_model
    get_model()
    print("Model ready.\n")

    print("--- Ingesting PDFs ---")
    summary: dict[str, int] = {}

    for pdf_path, domain, label in available:
        print(f"\nIngesting: {pdf_path.name} -> {label} ...")
        try:
            n_chunks = ingest_pdf(pdf_path=pdf_path, law_domain=domain, force=force)
            summary[label] = n_chunks
        except Exception as exc:
            print(f"  [ERROR] Failed to ingest '{pdf_path.name}': {exc}")
            summary[label] = -1

    print("\n" + "=" * 60)
    print("Ingestion Summary")
    print("=" * 60)
    for label, count in summary.items():
        if count > 0:
            status = f"{count:>5} chunks created"
        elif count == 0:
            status = "       skipped (already ingested)"
        else:
            status = "       FAILED"
        print(f"  {label:<25} {status}")

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
