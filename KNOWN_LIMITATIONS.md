# JLegal-ChatBot — Known Limitations

---

## 1. Traffic Law PDF Encoding

**What:** Article numbers are blank for all Traffic Law (قانون السير) chunks.
The chunks are semantically searchable but citations show "نص قانوني" instead of an article number.

**Why:** The original `traffic.pdf` stores Arabic text using Arabic Presentation Forms
(Unicode block U+FB50–U+FEFF — ligature codepoints, not standard Arabic).
The article regex patterns match `المادة` in standard Unicode, not the ligature form.
A replacement PDF was tested in v12.2 but had a different problem: 635 non-BMP
Tifinagh-range codepoints scattered through extracted text, yielding zero regex matches.

**Evidence:** `run_ingestion.py --force` on either file produces 0/N article labels for
TrafficLaw. All other 9 domains reach 99%+ article coverage on the same regex.

**Future fix:** Obtain a clean Unicode-encoded PDF of قانون السير رقم 49 لسنة 2008
from a government portal and run `python run_ingestion.py --force`.
Alternatively, post-process the extracted text with `unicodedata.normalize('NFKC', text)`
before regex matching to decompose ligature forms.

---

## 2. PDF Consultation Download Button

**What:** The "تحميل الاستشارة القانونية PDF" button is disabled in the deployed UI.

**Why:** Three interacting failure modes were encountered across versions v12–v12.5:

1. **Path resolution** (`v12–v12.4`): `Path(__file__).resolve()` can return the
   Streamlit server's working directory instead of the file's actual directory on
   Windows, so `font_path.exists()` returned False and the Amiri font was never loaded.
   Fixed in v12.5 by using `os.path.abspath(__file__)`.

2. **Closure capture** (`v12.4`): `LetterheadPDF` captured the outer `_mf` variable
   by reference. `header()` was triggered by `add_page()` before `_mf` was updated from
   `"Helvetica"` to `"Arabic"`, causing `FPDFUnicodeEncodingException` in every header.
   Fixed in v12.5 with constructor injection.

3. **Runtime path** (`v12.5`): Even with correct path resolution and constructor
   injection, Streamlit's module import mechanism on the deployment machine still
   prevents `open(font_path, "rb")` inside the fpdf2 font loader from resolving the path
   consistently. The button is currently disabled in the UI as a precaution for the
   defense demo; `_generate_pdf()` remains implemented and tested in isolation.

**Evidence:** `_generate_pdf(question, answer)` runs correctly when called from a plain
Python script (`python -c "from app import _generate_pdf; ..."`). The failure is
Streamlit-runtime-specific.

**Future fix:** Pre-load font bytes at module startup (before Streamlit changes cwd)
and pass the raw bytes to fpdf2 via `add_font(fname=io.BytesIO(font_bytes))` instead of
a file path. This eliminates the path-resolution dependency entirely.

---

## 3. Cybercrime Extortion Query Ranking

**What:** Queries about "ابتزاز إلكتروني" (electronic extortion/blackmail) sometimes
rank lower than expected, with top-1 scores in the 0.60–0.68 range rather than 0.75+.

**Why:** The Cybercrime Law uses the term "تهديد" (threat) in the statutory text more
often than "ابتزاز" (extortion/blackmail). AraBERTv02's embedding space places "تهديد"
and "ابتزاز" in different neighbourhoods because they are semantically distinct in
general Arabic text, even though the legal concept is the same.

**Evidence:** Direct embedding similarity `cosine(embed("ابتزاز"), embed("تهديد"))` ≈ 0.41.
The query expansion step (Claude Haiku rewrites colloquial to MSA) partially mitigates
this by including "تهديد" in the rewritten query, but only when `expand=True`.

**Future fix:** Fine-tune AraBERTv02 on a Jordanian legal synonym list
(`ابتزاز ↔ تهديد ↔ انتزاع`, etc.) using contrastive learning. Alternatively, add a
domain-specific synonym expansion table that supplements query expansion.

---

## 4. Corpus Coverage Boundary

**What:** The system only answers questions from 10 indexed Jordanian laws. Questions
about constitutional law, tax law, investment law, customs, consumer protection,
social security, and all other Jordanian legislation return low similarity scores and
a refusal message.

**Why:** Only laws with clean, Unicode-encoded PDFs were ingested. Scope was limited
by project timeline and PDF availability. Social Security Law (قانون الضمان الاجتماعي)
was explicitly excluded because its PDF is a scanned image with no text layer —
PyMuPDF extracts zero pages.

**Evidence:** Benchmark questions Q039–Q044 (refusal category) all produce
top-1 scores < 0.35 in the retrieval evaluation, confirming the system correctly
does not fabricate answers for out-of-corpus topics.

**Indexed laws (10):**
Labor, Commercial, PersonalStatus, PersonalStatus2019, Cybercrime,
CivilService, CivilStatus, HRManagement, TrafficLaw, PenalCode.

**Future fix:**
- Social Security: OCR pipeline using `pytesseract` with Arabic language pack (`ara`).
- Other laws: collect official PDFs from the Jordanian Legislation and Opinion Bureau
  (diwan.gov.jo) and run `python run_ingestion.py --force` after adding entries to
  `LAWS` list in `run_ingestion.py`.

---

## 5. Static Corpus (No Amendment Tracking)

**What:** The vector store is a snapshot of the laws at ingestion time. Legislative
amendments enacted after the last `run_ingestion.py` run are not reflected.

**Why:** The system uses a one-time offline ingestion model. There is no connection to
a live legal database, no webhook for Jordanian legislation updates, and no diff-based
re-ingestion strategy.

**Evidence:** Jordanian Personal Status Law was amended by Law No. 15 of 2019.
The system indexes both the original and the 2019 version as separate domains, but
any subsequent amendments to either version after the ingestion date are silently absent.

**Future fix:**
- Schedule periodic `run_ingestion.py --force` runs (e.g., quarterly cron job).
- Add a `last_ingested_at` timestamp to each DOCUMENT row and display a staleness
  warning in the UI when the timestamp exceeds a configurable threshold.
- Long-term: integrate with the Jordanian Official Gazette RSS feed (if available)
  to trigger re-ingestion on publication of new legislation.

---

## Previously Solved Issues (for reference)

| Issue | Status | Fixed In |
|-------|--------|----------|
| Levantine dialect direct retrieval (score 0.46–0.50) | Mitigated via Claude Haiku query expansion | v7 |
| Windows NTFS Arabic filename NFC normalization | Solved — `unicodedata.normalize('NFC', ...)` applied | v8 |
| numpy 2.x incompatibility with torch 2.3.1+cpu | Solved — pin `numpy<2` in requirements.txt | v8 |
| ChromaDB C++ HNSW DLL crash on Windows + Python 3.12 | Solved — replaced with numpy/JSON vector store | v1 |
| Article number coverage 40% (regex mismatch) | Solved — 4-pattern cascade + forward propagation | v5 |
