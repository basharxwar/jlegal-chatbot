# JLegal-ChatBot — Known Limitations

## 1. Traffic Law Article Numbers Not Detected
**Status:** Known, documented, low priority

The Traffic Law PDF (traffic.pdf) uses Arabic Presentation Forms
(Unicode range FB50-FEFF) instead of standard Arabic codepoints (0600-06FF).
PyMuPDF extracts the text correctly but the article number regex
`r"المادة\s*-?\s*(\d+)"` operates on standard codepoints only.

Result: All 108 Traffic Law chunks are searchable and retrievable,
but the article_number field is empty — citations show "نص قانوني" instead
of article numbers.

Fix: Apply `unicodedata.normalize('NFKC', text)` at PDF extraction time
to convert presentation forms to standard codepoints before regex matching.

Priority: Low — law is still useful for semantic search

---

## 2. Social Security Law Not Ingested
**Status:** Known, documented, future work

The Social Security PDF is a scanned image file (compressed PDF with no
text layer). PyMuPDF extracts 0 pages of text.

Fix: OCR pipeline using pytesseract with Arabic language pack (`ara`).
Steps needed:
1. Install tesseract-ocr with Arabic training data
2. Convert PDF pages to images (PyMuPDF can do this)
3. Run pytesseract.image_to_string(image, lang='ara')
4. Feed extracted text through existing chunking pipeline

Priority: Medium — documented as Future Work in project report

---

## 3. Levantine Dialect Direct Retrieval
**Status:** Partially mitigated via query expansion

AraBERTv02 was trained primarily on MSA (Modern Standard Arabic).
Direct dialectal queries like "فصلوني من الشغل فجأة" score 0.46-0.50,
which is below useful retrieval threshold.

Mitigation: Query expansion via Claude Haiku rewrites the query to formal
MSA before embedding. This improves scores to 0.73-0.75 and retrieves
relevant Labor Law articles.

Remaining gap: Article 25 (arbitrary dismissal) specifically still requires
lowering threshold to 0.45 or increasing top_k to 12.

Future work: Fine-tune AraBERTv02 on a Jordanian legal QA corpus.

Priority: Medium — mitigated, good demo story for committee

---

## 4. Windows NTFS Unicode Normalization
**Status:** Solved, documented for reference

Arabic filenames with Eastern Arabic numerals (١، ٢، ١٦) use different
Unicode normalization forms on Windows NTFS vs Python string literals.
`Path.exists()` returns False even when the file is present.

Fix applied: `unicodedata.normalize('NFC', filename)` on both the stored
filename and the glob results before comparison.

---

## 5. numpy Version Pinning Required
**Status:** Solved, documented for reference

torch 2.3.1+cpu was compiled against numpy 1.x.
numpy 2.x causes `RuntimeError: Numpy is not available` in torch.

Fix: Always install with `pip install "numpy<2"`.
Any dependency (numba, scipy, etc.) that pulls in numpy 2.x will break
torch tensor-to-numpy conversion.
