"""
compute_domain_precision.py — Domain routing precision for JLegal-ChatBot.

Measures whether the retriever returns chunks from the *correct* law domain.
Reads the latest retrieval_results_*.json and benchmark_questions.json.

Evaluable questions: is_in_corpus=True AND law_domain is not None
  (Cross-law questions with law_domain=null are excluded — they have no
   single expected domain and are reported separately.)

Metrics computed
----------------
- Domain Hit @1  : top-1 chunk comes from the expected law domain
- Domain Hit @5  : any of the top-5 chunks comes from the expected domain
- Per-domain breakdown: evaluable count, hits @1, hits @5, rates

Note: because run_retrieval_eval.py calls retrieve() with law_domain=<expected>
for single-domain questions, the retriever restricts results to that domain —
so domain precision is expected to be ~100% when the vector store is healthy.
Any domain below 100% indicates a missing or empty vector store collection.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK   = Path(__file__).parent / "benchmark_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"


def _latest_results() -> Path:
    files = sorted(RESULTS_DIR.glob("retrieval_results_*.json"))
    if not files:
        raise FileNotFoundError(
            "No retrieval_results_*.json found. "
            "Run python evaluation/run_retrieval_eval.py first."
        )
    return files[-1]


def main() -> None:
    results_path = _latest_results()
    print(f"Loading results : {results_path.name}")
    print(f"Loading benchmark: {BENCHMARK.name}\n")

    with open(results_path, encoding="utf-8") as f:
        results: list[dict] = json.load(f)

    with open(BENCHMARK, encoding="utf-8") as f:
        benchmark: list[dict] = json.load(f)

    # Build expected-domain lookup from benchmark (ground truth)
    bq_map = {q["id"]: q for q in benchmark}

    # Partition questions
    evaluable   = []   # is_in_corpus=True, law_domain != None
    cross_law   = []   # is_in_corpus=True, law_domain == None
    out_corpus  = []   # is_in_corpus=False

    for r in results:
        if not r["is_in_corpus"]:
            out_corpus.append(r)
        elif r.get("law_domain") is None:
            cross_law.append(r)
        else:
            evaluable.append(r)

    print(f"Total questions : {len(results)}")
    print(f"  Evaluable (in-corpus, single domain): {len(evaluable)}")
    print(f"  Cross-law  (in-corpus, null domain) : {len(cross_law)}  -- excluded")
    print(f"  Out-of-corpus (refusal)             : {len(out_corpus)} -- excluded\n")

    # ── Compute domain hits ──────────────────────────────────────────────────
    hits_at_1: list[bool] = []
    hits_at_5: list[bool] = []

    # Per-domain accumulators
    domain_stats: dict[str, dict] = {}

    for r in evaluable:
        expected = r["law_domain"]  # expected domain from benchmark / results record

        chunks = r.get("top_5_chunks", [])
        top1_domain = chunks[0]["law_domain"] if chunks else None
        top5_domains = {c["law_domain"] for c in chunks[:5]}

        h1 = (top1_domain == expected)
        h5 = (expected in top5_domains)

        hits_at_1.append(h1)
        hits_at_5.append(h5)

        if expected not in domain_stats:
            domain_stats[expected] = {"evaluable": 0, "hit1": 0, "hit5": 0,
                                       "question_ids": []}
        domain_stats[expected]["evaluable"] += 1
        domain_stats[expected]["hit1"] += int(h1)
        domain_stats[expected]["hit5"] += int(h5)
        domain_stats[expected]["question_ids"].append(r["question_id"])

    total = len(evaluable)
    overall_h1 = sum(hits_at_1) / total if total else 0.0
    overall_h5 = sum(hits_at_5) / total if total else 0.0

    # ── Print results ────────────────────────────────────────────────────────
    w = 60
    print("=" * w)
    print("  JLegal-ChatBot - Domain Routing Precision")
    print("=" * w)
    print(f"  Evaluable questions : {total}")
    print(f"  Domain Hit Rate @1  : {overall_h1:.1%}  ({sum(hits_at_1)}/{total})")
    print(f"  Domain Hit Rate @5  : {overall_h5:.1%}  ({sum(hits_at_5)}/{total})")
    print("-" * w)
    print(f"  {'Domain':<22} {'N':>3}  {'@1':>6}  {'@5':>6}  {'@1%':>7}  {'@5%':>7}")
    print(f"  {'-'*22} {'-'*3}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}")

    # Sort domains by expected order for readability
    DOMAIN_ORDER = [
        "Labor", "Commercial", "PersonalStatus", "PersonalStatus2019",
        "Cybercrime", "CivilService", "CivilStatus",
        "HRManagement", "TrafficLaw", "PenalCode",
    ]
    sorted_domains = [d for d in DOMAIN_ORDER if d in domain_stats]
    # Append any unexpected domains
    sorted_domains += [d for d in domain_stats if d not in sorted_domains]

    for d in sorted_domains:
        s = domain_stats[d]
        n   = s["evaluable"]
        h1  = s["hit1"]
        h5  = s["hit5"]
        r1  = h1 / n if n else 0.0
        r5  = h5 / n if n else 0.0
        bar = "#" * int(r1 * 8) + "-" * (8 - int(r1 * 8))
        print(f"  {d:<22} {n:>3}  {h1:>6}  {h5:>6}  [{bar}] {r1:>5.0%}  {r5:>5.0%}")

    print("=" * w)

    # ── Cross-law summary ─────────────────────────────────────────────────────
    if cross_law:
        print(f"\n  Cross-law questions (excluded from domain precision):")
        for r in cross_law:
            top_domains = ", ".join(
                f"{c['law_domain']}({c['score']:.2f})" for c in r["top_5_chunks"][:3]
            )
            print(f"    {r['question_id']}  top-3: {top_domains}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    per_domain_out = {}
    for d in sorted_domains:
        s = domain_stats[d]
        n = s["evaluable"]
        per_domain_out[d] = {
            "evaluable":    n,
            "hit_at_1":     s["hit1"],
            "hit_at_5":     s["hit5"],
            "rate_at_1":    round(s["hit1"] / n, 4) if n else 0.0,
            "rate_at_5":    round(s["hit5"] / n, 4) if n else 0.0,
            "question_ids": s["question_ids"],
        }

    output = {
        "source_file":          results_path.name,
        "computed_at":          datetime.now().isoformat(),
        "total_questions":      len(results),
        "evaluable_questions":  total,
        "cross_law_excluded":   len(cross_law),
        "out_of_corpus":        len(out_corpus),
        "overall_hit_rate_at_1": round(overall_h1, 4),
        "overall_hit_rate_at_5": round(overall_h5, 4),
        "per_domain":           per_domain_out,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"domain_precision_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {out_path}")
    print("Next step: python evaluation/generate_charts.py")


if __name__ == "__main__":
    main()
