"""
generator.py — Claude API integration for JLegal-ChatBot.

Builds an Arabic legal context from retrieved chunks and sends it to
claude-sonnet-4-6 with a strict Arabic system prompt that enforces
citation of article numbers and prohibits hallucination.
"""

import os
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLM_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """أنت مستشار قانوني أردني متخصص. مهمتك الإجابة على الأسئلة القانونية بدقة واحترافية.

القواعد الصارمة:
1. أجب فقط بناءً على النصوص القانونية المقدمة في السياق أدناه
2. اذكر رقم المادة واسم القانون لكل معلومة تستند إليها
3. إذا لم تجد إجابة واضحة في النصوص المقدمة، قل بوضوح: "لم أجد في النصوص القانونية المتاحة ما يجيب على هذا السؤال تحديداً"
4. لا تضف أي معلومات من خارج النصوص المقدمة
5. استخدم لغة قانونية واضحة ومفهومة"""

NO_RESULT_MESSAGE = (
    "لم أجد في النصوص القانونية المتاحة ما يجيب على هذا السؤال تحديداً. "
    "يُرجى إعادة صياغة السؤال أو اختيار نطاق قانوني أكثر تحديداً."
)

# ---------------------------------------------------------------------------
# Client factory (lazy init)
# ---------------------------------------------------------------------------

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    """Return the shared Anthropic client, initialising it on first call."""
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Please copy .env.example to .env and add your key."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered Arabic legal context block.

    Each chunk is labelled with its law name and article number so the
    model can cite them precisely in its answer.
    """
    lines: list[str] = ["النصوص القانونية ذات الصلة:\n"]
    for i, chunk in enumerate(chunks, start=1):
        law_name = chunk.get("law_name_ar") or chunk.get("law_domain", "")
        article = chunk.get("article_number")
        article_label = f"المادة {article} — " if article else ""
        lines.append(
            f"[{i}] {article_label}{law_name}\n"
            f"{chunk['chunk_text']}\n"
            f"{'─' * 60}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def generate_answer(
    query_text: str,
    chunks: list[dict],
) -> dict:
    """Call the Claude API to produce an Arabic legal answer.

    Parameters
    ----------
    query_text:
        The user's original question.
    chunks:
        Retrieved and ranked chunk dicts from retriever.retrieve().
        If empty, returns a polite no-result message without an API call.

    Returns
    -------
    dict with keys:
        response_text  (str)   — Arabic answer or no-result message
        tokens_used    (int)   — Total input + output tokens consumed
        is_no_result   (bool)  — True when no relevant chunks were found
        model_used     (str)   — LLM model identifier
        chunks_used    (list)  — The chunk list that was passed in
    """
    # ---- Fast path: no relevant chunks ------------------------------------
    if not chunks:
        return {
            "response_text": NO_RESULT_MESSAGE,
            "tokens_used": 0,
            "is_no_result": True,
            "model_used": LLM_MODEL,
            "chunks_used": [],
        }

    # ---- Build user message -----------------------------------------------
    context = build_context(chunks)
    user_message = (
        f"{context}\n\n"
        f"السؤال القانوني:\n{query_text}"
    )

    # ---- Call Claude API ---------------------------------------------------
    try:
        client = _get_client()
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        response_text: str = response.content[0].text
        tokens_used: int = response.usage.input_tokens + response.usage.output_tokens
    except Exception:
        return {
            "response_text": (
                "عذراً، حدث خطأ أثناء الاتصال بالنظام. "
                "يرجى التحقق من مفتاح API والمحاولة مرة أخرى."
            ),
            "tokens_used": 0,
            "is_no_result": True,
            "model_used": LLM_MODEL,
            "chunks_used": chunks,
        }

    return {
        "response_text": response_text,
        "tokens_used": tokens_used,
        "is_no_result": False,
        "model_used": LLM_MODEL,
        "chunks_used": chunks,
    }
