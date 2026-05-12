"""
run_retrieval_eval.py — Retrieval-only evaluation for JLegal-ChatBot.

Runs all 50 benchmark questions through the retriever.
No LLM calls are made — this script is completely free to run.
Saves results to evaluation/results/retrieval_results_TIMESTAMP.json.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK = Path(__file__).parent / "benchmark_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    from src.retriever import retrieve

    with open(BENCHMARK, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Running retrieval evaluation on {len(questions)} questions...")
    print("(No LLM calls — query expansion disabled)\n")

    results = []

    for q in questions:
        start = time.perf_counter()
        chunks = retrieve(
            query_text=q["question"],
            law_domain=q.get("law_domain"),
            top_k=10,
            threshold=0.0,
            expand=False,   # no API calls
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        top5 = [
            {
                "chunk_id": c.get("chunk_id", ""),
                "article_number": c.get("article_number", ""),
                "law_domain": c.get("law_domain", ""),
                "law_name_ar": c.get("law_name_ar", ""),
                "score": round(c["score"], 4),
                "rank": c.get("rank", i + 1),
            }
            for i, c in enumerate(chunks[:5])
        ]

        top_score = round(chunks[0]["score"], 4) if chunks else 0.0

        results.append({
            "question_id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "law_domain": q.get("law_domain"),
            "expected_article": q.get("expected_article"),
            "is_in_corpus": q["is_in_corpus"],
            "top_5_chunks": top5,
            "top_score": top_score,
            "result_count": len(chunks),
            "retrieval_time_ms": round(elapsed_ms, 2),
        })

        status = "IN " if q["is_in_corpus"] else "OUT"
        print(f"  [{q['id']}] {status} top={top_score:.3f}  time={elapsed_ms:5.0f}ms")

    # Summary statistics
    avg_time = sum(r["retrieval_time_ms"] for r in results) / len(results)
    avg_score = sum(r["top_score"] for r in results) / len(results)
    in_corpus = [r for r in results if r["is_in_corpus"]]
    out_corpus = [r for r in results if not r["is_in_corpus"]]
    avg_in = sum(r["top_score"] for r in in_corpus) / len(in_corpus) if in_corpus else 0
    avg_out = sum(r["top_score"] for r in out_corpus) / len(out_corpus) if out_corpus else 0

    print(f"\n{'='*55}")
    print(f"  Total questions    : {len(results)}")
    print(f"  In-corpus          : {len(in_corpus)}")
    print(f"  Out-of-corpus      : {len(out_corpus)}")
    print(f"  Avg retrieval time : {avg_time:.1f} ms")
    print(f"  Avg top-1 score    : {avg_score:.3f}")
    print(f"  Avg score in-corpus: {avg_in:.3f}")
    print(f"  Avg score out-corp : {avg_out:.3f}")
    print(f"{'='*55}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"retrieval_results_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {out_path}")
    print("Next step: python evaluation/compute_metrics.py")


if __name__ == "__main__":
    main()
