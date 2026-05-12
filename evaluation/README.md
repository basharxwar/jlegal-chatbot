# JLegal-ChatBot — Evaluation Suite

Offline evaluation framework for retrieval quality, system metrics, and RAG vs LLM-only comparison.

---

## Quick Start

```bash
# Step 1 — retrieval-only evaluation (free, ~30 seconds)
python evaluation/run_retrieval_eval.py

# Step 2 — compute metrics from the results
python evaluation/compute_metrics.py

# Step 3 — generate 4 PNG charts for the report
python evaluation/generate_charts.py

# Step 4 (optional, costs ~$0.50 USD) — RAG vs LLM-only comparison
python evaluation/run_comparison.py
```

---

## Files

| File | Purpose |
|------|---------|
| `benchmark_questions.json` | 50-question dataset with ground truth |
| `run_retrieval_eval.py` | Runs retrieval on all 50 questions, no LLM calls |
| `compute_metrics.py` | Reads results, computes Hit Rate, Refusal Rate, timing |
| `generate_charts.py` | Produces 4 PNG charts at 300 dpi |
| `run_comparison.py` | RAG vs LLM-only on 10 questions (costs API credits) |
| `results/` | Output JSON files (gitignored except `.gitkeep`) |
| `charts/` | Output PNG files |

---

## Benchmark Dataset

**50 questions** across all 10 law domains.

| Category | Count | Description |
|----------|-------|-------------|
| `factual` | 30 | 3 per law domain, known articles, in-corpus |
| `cross_law` | 8 | Require articles from 2+ laws |
| `refusal` | 6 | Out-of-corpus topics (tax, investment, constitutional…) |
| `dialect` | 4 | Jordanian dialect phrasing of factual questions |
| `edge_case` | 2 | Very short (1 word) and very long (multi-clause) |

Each question record:
```json
{
  "id": "Q001",
  "category": "factual",
  "law_domain": "Labor",
  "question": "...",
  "expected_article": "61",
  "expected_keywords": ["إجازة", "سنوية"],
  "is_in_corpus": true,
  "notes": "..."
}
```

`expected_article` is `null` for questions where the target article number was not verified — Hit Rate metrics only cover questions where it is set.

---

## Metrics Explained

### Hit Rate @K
Percentage of in-corpus questions where the `expected_article` appears in the top-K retrieved chunks.

- **@1**: Does the most relevant chunk come from the right article?
- **@5**: Is the right article anywhere in the top 5?

Computed only on questions where `is_in_corpus=true` and `expected_article` is not null.

### Refusal Rate
Percentage of out-of-corpus questions where the system correctly produces a low similarity score (< 0.45).
A high refusal rate indicates the system does not hallucinate answers for topics outside the indexed corpus.

### Average Retrieval Time
Mean time in milliseconds to embed a query and search all vector store domains (AraBERT inference + numpy cosine similarity).

### Per-Domain Hit Rate @5
Hit Rate @5 broken down by law domain — shows which law domains are better or worse indexed.

---

## Charts

| File | Contents |
|------|---------|
| `chart_hit_rate_by_domain.png` | Bar chart: Hit Rate @1 and @5 per domain |
| `chart_response_time.png` | Histogram of retrieval times across 50 questions |
| `chart_score_distribution.png` | Similarity score distribution, in-corpus vs out-of-corpus |
| `chart_rag_vs_llm.png` | RAG Pipeline vs LLM-Only comparison (illustrative) |

Charts are generated at 300 dpi, white background, suitable for direct inclusion in the project report.

If results files are not present, charts use labelled placeholder data with a visible note.

---

## Notes

- Generation tests (Anthropic API calls) are excluded from the free evaluation path — only `run_comparison.py` uses API credits.
- `run_retrieval_eval.py` passes `expand=False` to the retriever, disabling the Claude Haiku query expansion step to keep it free.
- Re-running `run_retrieval_eval.py` always creates a new timestamped file — old results are not overwritten.
- `compute_metrics.py` and `generate_charts.py` always read the **latest** results file automatically.
