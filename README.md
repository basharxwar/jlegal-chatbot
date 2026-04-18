# JLegal-ChatBot | المساعد القانوني الأردني

An Arabic Legal RAG (Retrieval-Augmented Generation) system for Jordanian law,
built as a graduation project at Yarmouk University.

It reads five Jordanian law PDFs, stores TF-IDF embeddings as JSON files,
and answers legal questions in Arabic using the Claude API with strict citation
of article numbers and law names.

> **Note:** This project uses numpy-based vector search instead of ChromaDB
> for full Windows compatibility — no C++ build tools required.

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Create a `.env` file in the project root**
```
ANTHROPIC_API_KEY=your-key-here
```

**3. Run ingestion (once — builds the vector store from the PDFs)**
```bash
python run_ingestion.py
```

**4. Start the app**
```bash
streamlit run app.py
```

To re-ingest all documents from scratch:
```bash
python run_ingestion.py --force
```

---

## Laws Covered

| Domain | Law |
|--------|-----|
| Labor | قانون العمل الأردني |
| Commercial | قانون التجارة الأردني |
| PersonalStatus | قانون الأحوال الشخصية |
| Cybercrime | قانون الجرائم الإلكترونية الأردني |
| CivilService | نظام الخدمة المدنية |

---

## PDF Download (Arabic font)

The PDF export button uses `fpdf2` with `arabic-reshaper` and `python-bidi`.
For correct Arabic rendering in the PDF, download the Amiri font and place it at:

```
fonts/Amiri-Regular.ttf
```

Download: https://fonts.google.com/specimen/Amiri

Without the font file the PDF is still generated but Arabic text may not display correctly.

---

## Project Structure

```
jlegal_chatbot/
├── src/
│   ├── ingest.py        # PDF extraction, chunking, TF-IDF embedding
│   ├── vector_store.py  # Pure numpy cosine similarity search
│   ├── retriever.py     # Query embedding and retrieval logic
│   ├── generator.py     # Claude API integration
│   ├── database.py      # SQLite schema and CRUD
│   └── pipeline.py      # RAG orchestration
├── app.py               # Streamlit UI
├── run_ingestion.py     # One-time ingestion script
├── vector_store/        # JSON embedding files (created after ingestion)
├── fonts/               # Place Amiri-Regular.ttf here for PDF Arabic support
├── requirements.txt
└── .env
```

---

## Legal Notice

هذا النظام أداة بحثية مساعدة ولا يُعدّ بديلاً عن الاستشارة القانونية المتخصصة.
