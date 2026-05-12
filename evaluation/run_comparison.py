"""
run_comparison.py — RAG vs LLM-Only comparison for JLegal-ChatBot.

WARNING
-------
This script makes approximately 20 Anthropic API calls.
Estimated cost: ~$0.50 USD (10 questions × 2 calls each at Haiku pricing).
Do NOT run this automatically in CI or evaluation loops.
The script will abort immediately if ANTHROPIC_API_KEY is not set.

Usage
-----
    python evaluation/run_comparison.py

Output
------
    evaluation/results/comparison_TIMESTAMP.json
    Console: side-by-side RAG vs LLM-only answers for each question
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK = Path(__file__).parent / "benchmark_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 10 representative questions: factual, cross-law, refusal, dialect, edge
COMPARISON_IDS = [
    "Q001",  # factual / Labor — annual leave
    "Q002",  # factual / Labor — arbitrary dismissal
    "Q010",  # factual / Cybercrime — extortion
    "Q019",  # factual / PersonalStatus2019 — custody
    "Q028",  # factual / PenalCode — theft
    "Q031",  # cross-law — Labor vs CivilService
    "Q032",  # cross-law — Commerce + Cybercrime
    "Q039",  # refusal — tax law (not in corpus)
    "Q045",  # dialect — dismissal
    "Q050",  # edge case — long multi-aspect question
]


def _llm_only_answer(client, question: str) -> str:
    """Call Claude with no retrieved context (LLM-only baseline)."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system="أجب على السؤال القانوني التالي بشكل مختصر دون الرجوع إلى أي نصوص قانونية محددة.",
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text if response.content else ""


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set. Skipping comparison.")
        print("Set the key and re-run: python evaluation/run_comparison.py")
        return

    import anthropic
    from src.pipeline import run_query, ensure_session

    with open(BENCHMARK, encoding="utf-8") as f:
        all_questions = {q["id"]: q for q in json.load(f)}

    questions = [all_questions[qid] for qid in COMPARISON_IDS if qid in all_questions]

    print(f"RAG vs LLM-Only comparison — {len(questions)} questions")
    print(f"Estimated cost: ~$0.50 USD ({len(questions) * 2} API calls)\n")

    client = anthropic.Anthropic(api_key=api_key)
    session_id = ensure_session("eval_comparison_2026")

    comparisons = []

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']} — {q['question'][:60]}...")

        # RAG path: full pipeline with retrieval
        rag_result = run_query(
            query_text=q["question"],
            session_id=session_id,
            law_domain=q.get("law_domain"),
            style="formal",
        )
        rag_answer = rag_result.get("response_text", "")
        rag_chunks = len(rag_result.get("chunks", []))
        rag_success = rag_result.get("success", False)

        # LLM-only path: same question, no retrieval context
        llm_answer = _llm_only_answer(client, q["question"])

        comparisons.append({
            "question_id": q["id"],
            "category": q["category"],
            "law_domain": q.get("law_domain"),
            "is_in_corpus": q["is_in_corpus"],
            "question": q["question"],
            "rag_answer": rag_answer,
            "rag_chunks_used": rag_chunks,
            "rag_success": rag_success,
            "llm_only_answer": llm_answer,
        })

        print(f"  RAG  ({rag_chunks} chunks): {rag_answer[:120].strip()}...")
        print(f"  LLM-only:            {llm_answer[:120].strip()}...")
        print()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"comparison_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, ensure_ascii=False, indent=2)

    print(f"Comparison saved to: {out}")


if __name__ == "__main__":
    main()
