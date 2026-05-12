"""
compute_metrics.py — Compute evaluation metrics from retrieval results.

Reads the latest evaluation/results/retrieval_results_*.json and prints:
  - Hit Rate @1, @5 (for in-corpus questions with expected_article)
  - Refusal Rate (out-of-corpus questions where top_score < 0.45)
  - Average retrieval time
  - Per-domain Hit Rate @5

Also saves a metrics_TIMESTAMP.json to evaluation/results/.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"


def _latest_results() -> Path:
    files = sorted(RESULTS_DIR.glob("retrieval_results_*.json"))
    if not files:
        raise FileNotFoundError(
            "No results files found in evaluation/results/. "
            "Run python evaluation/run_retrieval_eval.py first."
        )
    return files[-1]


def _hit_at_k(evaluable: list[dict], k: int) -> float:
    """Fraction of evaluable questions where expected_article appears in top-K retrieved."""
    if not evaluable:
        return 0.0
    hits = 0
    for r in evaluable:
        expected = str(r["expected_article"])
        retrieved = [c["article_number"] for c in r["top_5_chunks"][:k]]
        if expected in retrieved:
            hits += 1
    return hits / len(evaluable)


def main() -> None:
    path = _latest_results()
    print(f"Loading: {path.name}\n")

    with open(path, encoding="utf-8") as f:
        results = json.load(f)

    # Partition question sets
    evaluable = [r for r in results if r["is_in_corpus"] and r.get("expected_article")]
    in_corpus = [r for r in results if r["is_in_corpus"]]
    out_corpus = [r for r in results if not r["is_in_corpus"]]

    hr1 = _hit_at_k(evaluable, 1)
    hr5 = _hit_at_k(evaluable, 5)

    # Refusal: out-of-corpus question where system correctly returns low score
    refusal_threshold = 0.45
    correct_refusals = sum(1 for r in out_corpus if r["top_score"] < refusal_threshold)
    refusal_rate = correct_refusals / len(out_corpus) if out_corpus else 0.0

    avg_time = sum(r["retrieval_time_ms"] for r in results) / len(results)
    avg_score_all = sum(r["top_score"] for r in results) / len(results)
    avg_score_in = sum(r["top_score"] for r in in_corpus) / len(in_corpus) if in_corpus else 0.0
    avg_score_out = sum(r["top_score"] for r in out_corpus) / len(out_corpus) if out_corpus else 0.0

    # Per-domain Hit Rate @5 (only questions with expected_article)
    domain_stats: dict[str, dict] = {}
    for r in evaluable:
        d = r.get("law_domain") or "Mixed"
        if d not in domain_stats:
            domain_stats[d] = {"total": 0, "hits": 0}
        domain_stats[d]["total"] += 1
        expected = str(r["expected_article"])
        articles = [c["article_number"] for c in r["top_5_chunks"][:5]]
        if expected in articles:
            domain_stats[d]["hits"] += 1

    # ── Print results ────────────────────────────────────────────────────
    w = 55
    print("=" * w)
    print("  JLegal-ChatBot - Retrieval Evaluation Metrics")
    print("=" * w)
    print(f"  Source file             : {path.name}")
    print(f"  Total questions         : {len(results)}")
    print(f"  In-corpus               : {len(in_corpus)}")
    print(f"  Out-of-corpus (refusal) : {len(out_corpus)}")
    print(f"  Evaluable (w/ article)  : {len(evaluable)}")
    print("-" * w)
    print(f"  Avg retrieval time      : {avg_time:.1f} ms")
    print(f"  Avg top-1 score (all)   : {avg_score_all:.3f}")
    print(f"  Avg top-1 score in-corp : {avg_score_in:.3f}")
    print(f"  Avg top-1 score out-corp: {avg_score_out:.3f}")
    print("-" * w)
    print(f"  Hit Rate @1             : {hr1:.1%}  ({round(hr1 * len(evaluable))}/{len(evaluable)})")
    print(f"  Hit Rate @5             : {hr5:.1%}  ({round(hr5 * len(evaluable))}/{len(evaluable)})")
    print(f"  Refusal Rate (<{refusal_threshold:.2f})    : {refusal_rate:.1%}  ({correct_refusals}/{len(out_corpus)})")
    print("-" * w)
    print("  Per-Domain Hit Rate @5 (evaluable questions only):")
    if domain_stats:
        for domain, stats in sorted(domain_stats.items()):
            hr = stats["hits"] / stats["total"] if stats["total"] else 0.0
            bar = "#" * int(hr * 10) + "-" * (10 - int(hr * 10))
            print(f"    {domain:<22} [{bar}]  {hr:.0%}  ({stats['hits']}/{stats['total']})")
    else:
        print("    (no evaluable questions with expected_article in this run)")
    print("=" * w)

    # Save metrics JSON
    metrics = {
        "source_file": path.name,
        "computed_at": datetime.now().isoformat(),
        "total_questions": len(results),
        "in_corpus_questions": len(in_corpus),
        "out_of_corpus_questions": len(out_corpus),
        "evaluable_questions": len(evaluable),
        "avg_retrieval_time_ms": round(avg_time, 2),
        "avg_top1_score_all": round(avg_score_all, 4),
        "avg_top1_score_in_corpus": round(avg_score_in, 4),
        "avg_top1_score_out_of_corpus": round(avg_score_out, 4),
        "hit_rate_at_1": round(hr1, 4),
        "hit_rate_at_5": round(hr5, 4),
        "refusal_rate": round(refusal_rate, 4),
        "refusal_threshold": refusal_threshold,
        "per_domain_hit_rate_at_5": {
            d: {
                "rate": round(s["hits"] / s["total"], 4) if s["total"] else 0.0,
                "hits": s["hits"],
                "total": s["total"],
            }
            for d, s in domain_stats.items()
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"metrics_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nMetrics saved to: {out}")
    print("Next step: python evaluation/generate_charts.py")


if __name__ == "__main__":
    main()
