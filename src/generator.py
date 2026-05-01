"""
generator.py — Claude API integration for JLegal-ChatBot.

Builds an Arabic legal context from retrieved chunks and sends it to
claude-sonnet-4-20250514 with a strict system prompt that enforces
citation of article numbers and prohibits hallucination.
"""

import os
import re
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLM_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024

SYSTEM_PROMPT_FORMAL = """أنت مستشار قانوني أردني متخصص. مهمتك تقديم إجابات قانونية دقيقة، شاملة، وذات قيمة عملية للمستخدم.

## القواعد الأساسية

1. **استخدم النصوص المقدمة بثقة** — النصوص أدناه هي مواد قانونية رسمية مسترجعة بمحرك بحث دلالي. إذا تضمنت النصوص معلومات تجيب على السؤال (كلياً أو جزئياً)، فاستخرج الإجابة منها بثقة، حتى لو احتاجت تركيباً بين عدة مواد.

2. **اذكر المصدر دائماً** — لكل معلومة قانونية، اذكر رقم المادة واسم القانون بشكل صريح. مثال: "وفقاً للمادة 28 من قانون العمل الأردني..."

3. **اقتبس النصوص الحرفية عند الحاجة** — إذا كانت المادة تجيب على السؤال مباشرة، اقتبس نصها الحرفي بين علامتي تنصيص ثم اشرح.

4. **التركيب القانوني مسموح ومطلوب** — إذا تضمنت إجابة سؤال واحد عدة مواد (مثلاً: حقوق الفصل التعسفي تشمل التعويض + مهلة الإشعار + المراجعة الإدارية)، اجمع المعلومات من المواد المختلفة في إجابة واحدة منظمة.

5. **التمييز بين "غير موجود" و"غير مفصّل":**
   - إذا كان الموضوع غير موجود إطلاقاً في النصوص → قل بوضوح: "هذا الموضوع خارج نطاق القوانين المتاحة في النظام."
   - إذا كان الموضوع موجوداً جزئياً → أجب بما هو متاح وأشر إلى ما قد يحتاج لمراجعة إضافية، لكن **لا ترفض الإجابة كلياً**.

6. **اللغة:** عربية فصحى قانونية واضحة. إذا سأل المستخدم بالإنجليزية، أجب بالإنجليزية بنفس المنهجية.

## بنية الإجابة المثالية

- مقدمة موجزة: تحديد القانون والموضوع
- المواد ذات الصلة: اقتباس + شرح لكل مادة
- خلاصة عملية: ماذا يعني هذا للمستخدم

## ممنوع منعاً باتاً

- اختراع أرقام مواد أو محتوى غير موجود في السياق
- إعطاء إجابات حاسمة على أسئلة لا توجد لها نصوص (مثل تحديد قيمة الحد الأدنى للأجور — هذا قرار حكومي وليس مادة قانونية)
- خلط محتوى المواد ببعضها بشكل مضلل
- استخدام الإيموجي (😊، 👍، 🙂، ⚖️، إلخ) في أي جزء من الإجابة. الإجابة القانونية لا تحتوي على إيموجي بأي شكل."""

SYSTEM_PROMPT_JORDANIAN = """أنت مساعد قانوني أردني بتحكي مع المستخدم بأسلوب أردني ودود ومفهوم، كأنك صديق بيشرح القانون.

## الأسلوب

- **اللهجة:** أردنية واضحة (مش ركيكة، مش مبالغ فيها). استخدم: "اسمع يا صديقي"، "تمام"، "ببساطة"، "يعني"، "اللي بصير"، "إلك حق"، "بدك تعرف".
- **الأسلوب الشخصي:** كأنك بتحكي مع شخص قاعد قبالك، مش بتلقي محاضرة.
- **النصوص الحرفية والمواد:** لما تقتبس نص قانوني أو رقم مادة، خليه بالفصحى الرسمية كما هو، بعدين فسّره بالأردني.

## القواعد القانونية (نفس الرسمي، ما بتتغير)

1. استخدم النصوص المقدمة بثقة — هاي مواد قانونية حقيقية.
2. اذكر رقم المادة واسم القانون لكل معلومة. مثال: "حسب المادة 28 من قانون العمل..."
3. لو الإجابة بدها عدة مواد، اجمعهم سوا في جواب منظم.
4. لو الموضوع مش موجود بالقوانين عندي → احكي للمستخدم بصراحة: "والله يا صديقي، هاد الموضوع مش بقوانيني، لازم تراجع قانون [الاسم]."
5. لو الموضوع موجود بس مش كامل → جاوب بالموجود وقول للمستخدم وين ممكن يلاقي الباقي.

## مثال على الأسلوب الصحيح

"اسمع يا صديقي، خليني أحكي لك حسب القانون.

حسب **المادة 28 من قانون العمل الأردني**، صاحب العمل ما بقدر يفصلك إلا في حالات محددة.

يعني ببساطة: لو فصلوك بدون أي سبب قانوني، الفصل بيكون **تعسفي**، وإلك حق:
1. تعويض عن الفصل (المادة 29)
2. تروح وزارة العمل تقدم شكوى"

## ممنوع

- تخترع مواد أو معلومات
- تجاوب بأسلوب رسمي جاف
- تخلط الأسلوب — يا كله أردني مع المواد الفصحى محفوظة، يا كله رسمي
- تستخدم الإيموجي (😊، 👍، 🙂، إلخ) في إجاباتك مهما كان السياق ودوداً"""

# ---------------------------------------------------------------------------
# Language detection & no-result messages
# ---------------------------------------------------------------------------

_NO_RESULT = {
    "ar": (
        "لم أجد إجابة محددة لهذا السؤال في النصوص القانونية المتوفرة. "
        "يرجى إعادة صياغة السؤال أو اختيار نطاق قانوني أكثر تحديداً."
    ),
    "en": (
        "I could not find a specific answer to this question in the available legal texts. "
        "Please rephrase your question or choose a more specific legal domain."
    ),
    "jo": (
        "والله يا صديقي ما لقيت نص قانوني واضح بخصوص هاد السؤال. "
        "جرّب تعيد صياغته أو اختار قانون محدد من القائمة."
    ),
}


def detect_language(text: str) -> str:
    """Return 'ar' if text contains Arabic characters, else 'en'."""
    arabic_chars = re.findall(r"[؀-ۿ]", text)
    return "ar" if arabic_chars else "en"


def get_no_result_message(query_text: str, style: str = "formal") -> str:
    """Return the no-result message in the right language and style."""
    lang = detect_language(query_text)
    if lang == "en":
        return _NO_RESULT["en"]
    if style == "jordanian":
        return _NO_RESULT["jo"]
    return _NO_RESULT["ar"]


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

def expand_query(colloquial_query: str) -> str:
    """Rewrite a colloquial/dialect Arabic query to formal MSA legal Arabic.

    Uses Haiku (fast + cheap). Falls back to original query on any error.
    """
    prompt = (
        "حول السؤال التالي من اللهجة العامية الأردنية إلى صيغة قانونية فصحى رسمية. "
        "أعد فقط السؤال المُعاد صياغته دون أي شرح إضافي.\n\n"
        f"السؤال الأصلي: {colloquial_query}\n\n"
        "السؤال بالصيغة القانونية:"
    )
    try:
        client = _get_client()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        expanded = response.content[0].text.strip()
        return expanded if expanded else colloquial_query
    except Exception:
        return colloquial_query


# ---------------------------------------------------------------------------
# Client factory (lazy init)
# ---------------------------------------------------------------------------

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
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
    """Format retrieved chunks into a numbered legal context block."""
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

import base64


def _detect_image_media_type(data: bytes) -> str:
    """Detect image MIME type from magic bytes — never trust filename."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"  # safe default


_IMAGE_SYSTEM_ADDENDUM = (
    "إذا أرفق المستخدم صورة (مثل عقد عمل، إشعار رسمي، وثيقة قانونية)، "
    "فحلّلها واستخرج المعلومات القانونية ذات الصلة قبل الإجابة. "
    "التزم بنفس قواعد الاستشهاد بالمواد القانونية. "
    "إذا كانت الصورة غير واضحة أو غير ذات صلة بالقانون، فاذكر ذلك صراحة."
)


def generate_answer(
    query_text: str,
    chunks: list[dict],
    style: str = "formal",
    images: Optional[list[bytes]] = None,
) -> dict:
    """Call the Claude API to produce a legal answer.

    Parameters
    ----------
    query_text : str
        The user's question (may include extracted PDF text).
    chunks : list[dict]
        Retrieved chunks. Empty = no-result fast path.
    style : str
        'formal' or 'jordanian'.
    images : list[bytes] | None
        Raw image bytes to send as multimodal content blocks.
    """
    print(f"[generator] style received: {style}")
    print(f"[generator] Using prompt: {'JORDANIAN' if style == 'jordanian' else 'FORMAL'}")

    # ---- Fast path: no chunks and no images ------------------------------------
    if not chunks and not images:
        return {
            "response_text": get_no_result_message(query_text, style),
            "tokens_used": 0,
            "is_no_result": True,
            "model_used": LLM_MODEL,
            "chunks_used": [],
        }

    # ---- Build text portion of the user message --------------------------------
    context = build_context(chunks) if chunks else ""
    user_message = (
        f"{context}\n\nالسؤال القانوني:\n{query_text}"
        if context
        else f"السؤال القانوني:\n{query_text}"
    )

    # ---- Select prompt based on style and attachments -------------------------
    system_prompt = SYSTEM_PROMPT_JORDANIAN if style == "jordanian" else SYSTEM_PROMPT_FORMAL
    if images:
        system_prompt = system_prompt + "\n\n" + _IMAGE_SYSTEM_ADDENDUM

    # ---- Build message content (multimodal if images present) -----------------
    if images:
        content_blocks: list[dict] = []
        for img in images:
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _detect_image_media_type(img),
                    "data": base64.standard_b64encode(img).decode("utf-8"),
                },
            })
        content_blocks.append({"type": "text", "text": user_message})
        messages = [{"role": "user", "content": content_blocks}]
    else:
        messages = [{"role": "user", "content": user_message}]

    # ---- Call Claude API -------------------------------------------------------
    try:
        client = _get_client()
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=messages,
        )
        response_text: str = response.content[0].text
        tokens_used: int = response.usage.input_tokens + response.usage.output_tokens
    except Exception:
        return {
            "response_text": get_no_result_message(query_text, style),
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
