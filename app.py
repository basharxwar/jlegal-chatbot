import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import streamlit as st

st.set_page_config(
    page_title="JLegal-ChatBot | المساعد القانوني الأردني",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports after set_page_config
# ---------------------------------------------------------------------------

import uuid
import tempfile
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from src.pipeline import run_query, ensure_session
    from src.vector_store import VECTOR_STORE_DIR
    from src.embedder import get_model
    from src.database import get_chunk_count
    _startup_error: str | None = None
except Exception as _e:
    run_query = ensure_session = VECTOR_STORE_DIR = get_model = get_chunk_count = None  # type: ignore
    _startup_error = str(_e)

# ---------------------------------------------------------------------------
# Cached resource loaders — all triggered at startup
# ---------------------------------------------------------------------------

@st.cache_resource
def load_arabert():
    if get_model is None:
        return None
    return get_model()


@st.cache_resource
def preload_all_collections():
    from src.vector_store import _load_domain
    for domain in [
        "Labor", "Commercial", "PersonalStatus", "Cybercrime", "CivilService",
        "CivilStatus", "SocialSecurity", "PersonalStatus2019", "TrafficLaw", "HRManagement",
    ]:
        _load_domain(domain)
    return True


@st.cache_resource
def load_whisper_model():
    try:
        import whisper
        return whisper.load_model("base")
    except Exception as e:
        logger.exception("Whisper failed to load")
        return None


load_arabert()
preload_all_collections()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_IMAGE_BYTES = 5 * 1024 * 1024   # 5 MB
MAX_PDF_BYTES   = 10 * 1024 * 1024  # 10 MB

DOMAIN_OPTIONS: dict[str, str | None] = {
    "جميع القوانين | All Laws": None,
    "قانون العمل | Labor": "Labor",
    "قانون التجارة | Commercial": "Commercial",
    "قانون الأحوال الشخصية | Personal Status": "PersonalStatus",
    "قانون الجرائم الإلكترونية | Cybercrime": "Cybercrime",
    "نظام الخدمة المدنية | Civil Service": "CivilService",
    "قانون الأحوال المدنية | Civil Status": "CivilStatus",
    "قانون الأحوال الشخصية 2019 | Personal Status 2019": "PersonalStatus2019",
    "قانون السير | Traffic Law": "TrafficLaw",
    "نظام إدارة الموارد البشرية | HR Management": "HRManagement",
}

LAW_NAMES_AR: dict[str, str] = {
    "Labor": "قانون العمل الأردني",
    "Commercial": "قانون التجارة الأردني",
    "PersonalStatus": "قانون الأحوال الشخصية",
    "Cybercrime": "قانون الجرائم الإلكترونية الأردني",
    "CivilService": "نظام الخدمة المدنية",
    "CivilStatus": "قانون الأحوال المدنية",
    "SocialSecurity": "قانون الضمان الاجتماعي",
    "PersonalStatus2019": "قانون الأحوال الشخصية 2019",
    "TrafficLaw": "قانون السير الأردني",
    "HRManagement": "نظام إدارة الموارد البشرية",
}

EXAMPLES = [
    ("⚖️", "ما هي حقوق العامل عند الفصل التعسفي؟"),
    ("📅", "ما هي مدة الإجازة السنوية في قانون العمل؟"),
    ("💑", "ما هي شروط الزواج في قانون الأحوال الشخصية؟"),
    ("📱", "ما هي عقوبة الابتزاز الإلكتروني في الأردن؟"),
    ("🚗", "ما هي مخالفات السير والمرور؟"),
    ("💼", "ما هي حقوق الموظف الحكومي في الإجازات؟"),
]

# ---------------------------------------------------------------------------
# CSS — Bug #2: navy background for main chat area + existing v10 theme
# ---------------------------------------------------------------------------

RTL_CSS = """
<style>
/* ── Bug #2: Full navy background ─────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    background-color: #1B3A57 !important;
    color: #F5F5F5 !important;
}

/* ── Sidebar gradient ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B3A57 0%, #2D7D8E 100%);
}
[data-testid="stSidebar"] * { color: #FAF7F2 !important; }
[data-testid="stSidebar"] .stSelectbox div,
[data-testid="stSidebar"] .stRadio div {
    background: rgba(255,255,255,0.08);
    border-radius: 6px;
}

/* ── Chat messages ────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background-color: #234866 !important;
    border: 1px solid #2D7D8E !important;
    color: #F5F5F5 !important;
    border-radius: 8px !important;
}
[data-testid="stChatMessageContent"] {
    direction: rtl;
    text-align: right;
    color: #F5F5F5 !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatAvatarIcon-user"]) {
    border-right: 4px solid #1B3A57 !important;
    background-color: #1E3F5A !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatAvatarIcon-assistant"]) {
    border-right: 4px solid #C9A961 !important;
}

/* ── Chat input ───────────────────────────────────────────── */
[data-testid="stChatInput"],
[data-testid="stChatInput"] textarea {
    background-color: #234866 !important;
    color: #F5F5F5 !important;
    border: 1px solid #C9A961 !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton button {
    background-color: #C9A961;
    color: #1B3A57;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton button:hover {
    background-color: #B89752;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(201,169,97,0.3);
}

/* ── Source cards ─────────────────────────────────────────── */
.source-card {
    background: #234866 !important;
    border-right: 4px solid #C9A961 !important;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 8px;
    color: #F5F5F5 !important;
    direction: rtl;
}
.source-article { font-size: 1rem; font-weight: 700; color: #C9A961; }
.source-law     { font-size: 0.88rem; color: #C5CDD8; margin-top: 2px; }
.score-badge    { background: #C9A961 !important; color: #1B3A57 !important;
                  padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.chunk-preview  { color: #C5CDD8 !important; font-size: 0.80rem; margin-top: 6px;
                  line-height: 1.5; direction: rtl; }

/* ── Expanders ────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: #234866 !important;
    border: 1px solid #2D7D8E !important;
}
[data-testid="stExpander"] summary { color: #C9A961 !important; }

/* ── Example buttons ──────────────────────────────────────── */
div[data-testid="stButton"] button {
    direction: rtl;
    text-align: right;
    background: #1E3F5A;
    color: #F5F5F5;
    border: 1px solid #2D7D8E;
    border-radius: 10px;
    font-size: 0.88rem;
}
div[data-testid="stButton"] button:hover {
    border-color: #C9A961;
    background: #234866;
}

/* ── Miscellaneous ────────────────────────────────────────── */
body, .stMarkdown, .stCaption {
    font-family: 'Segoe UI', 'Arial', sans-serif;
    direction: rtl;
}
.stDivider { border-color: rgba(255,255,255,0.15) !important; }
p, li, span { color: #F5F5F5 !important; }
</style>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF byte buffer using PyMuPDF."""
    import fitz
    parts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts).strip()[:30_000]  # cap to protect context budget


def _generate_pdf(question: str, answer: str) -> bytes:
    from fpdf import FPDF
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        def ar(t): return get_display(arabic_reshaper.reshape(t))
        has_arabic = True
    except ImportError:
        def ar(t): return t
        has_arabic = False

    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=20)

    font_path = Path(__file__).resolve().parent / "fonts" / "Amiri-Regular.ttf"
    if font_path.exists() and has_arabic:
        pdf.add_font("Arabic", fname=str(font_path))
        mf, align = "Arabic", "R"
    else:
        mf, align = "Helvetica", "L"

    pdf.set_font(mf, size=16)
    pdf.set_text_color(27, 58, 87)
    pdf.cell(0, 12, ar("استشارة قانونية - JLegal-ChatBot"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, f"JLegal-ChatBot  |  {date.today().strftime('%Y-%m-%d')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3); pdf.line(15, pdf.get_y(), 195, pdf.get_y()); pdf.ln(6)
    for label, body in [("السؤال القانوني:", question), ("الإجابة القانونية:", answer)]:
        pdf.set_font(mf, size=12); pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 8, ar(label), align=align, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(mf, size=10); pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 6, ar(body), align=align); pdf.ln(5)
    pdf.set_y(-20); pdf.set_font("Helvetica", size=8); pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "JLegal-ChatBot - Yarmouk University", align="C")
    return bytes(pdf.output())


def _render_sources(chunks: list[dict]) -> None:
    with st.expander("📚 المصادر القانونية"):
        for chunk in chunks:
            law_name = LAW_NAMES_AR.get(chunk.get("law_domain", ""), "") or chunk.get("law_name_ar", "")
            article = chunk.get("article_number")
            article_label = f"المادة {article}" if article else "نص قانوني"
            score = chunk.get("score", 0.0)
            rank = chunk.get("rank", "—")
            preview = (chunk.get("chunk_text", "")[:130] + "…") if chunk.get("chunk_text") else ""
            st.markdown(
                f'<div class="source-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span class="source-article">{article_label}</span>'
                f'<span class="score-badge">تشابه: {score:.0%}</span>'
                f'</div>'
                f'<div class="source-law">{law_name} — ترتيب: {rank}</div>'
                f'<div class="chunk-preview">{preview}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_assistant_message(msg: dict, key_suffix: str) -> None:
    """Render a stored assistant message (history replay)."""
    st.markdown(msg["content"])

    if msg.get("chunks"):
        _render_sources(msg["chunks"])

    if msg.get("question"):
        try:
            pdf_bytes = _generate_pdf(msg["question"], msg["content"])
            st.download_button(
                label="تحميل الاستشارة القانونية PDF",
                data=pdf_bytes,
                file_name=f"استشارة_{date.today()}.pdf",
                mime="application/pdf",
                key=f"pdf_{key_suffix}",
            )
        except Exception as e:
            logger.exception("PDF generation failed for history message")

    # Show attachment metadata stored with this message
    for att in msg.get("attachments", []):
        icon = "🖼️" if att["type"] == "image" else "📄"
        st.caption(f"{icon} {att['name']}")

    fb_key = msg.get("query_id", key_suffix)
    fc = st.columns([1, 1, 8])
    with fc[0]:
        if st.button("👍", key=f"up_{fb_key}"):
            st.toast("شكراً لتقييمك!")
    with fc[1]:
        if st.button("👎", key=f"dn_{fb_key}"):
            st.toast("سنعمل على التحسين!")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        ensure_session(st.session_state.session_id)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_domain_label" not in st.session_state:
        st.session_state.selected_domain_label = list(DOMAIN_OPTIONS.keys())[0]
    if "response_style" not in st.session_state:
        st.session_state.response_style = "formal"


# ---------------------------------------------------------------------------
# App start
# ---------------------------------------------------------------------------

st.markdown(RTL_CSS, unsafe_allow_html=True)

if _startup_error:
    st.error("حدث خطأ عند تشغيل النظام. تأكد من تثبيت جميع المكتبات وتشغيل run_ingestion.py أولاً.")
    st.stop()

_init_state()

# Get chunk count before sidebar (needed in main header too)
db_chunk_count = get_chunk_count() if get_chunk_count else 0

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Compact logo
    st.markdown(
        '<div style="text-align:center;padding:8px 0;">'
        '<div style="font-size:1.6rem;">⚖️</div>'
        '<div style="font-size:1.15rem;font-weight:700;">JLegal-ChatBot</div>'
        '<div style="font-size:0.78rem;opacity:0.85;">المساعد القانوني الأردني</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Voice input
    st.markdown("##### 🎤 الإدخال الصوتي")
    _voice_import_ok = True
    try:
        from streamlit_mic_recorder import mic_recorder
    except ImportError as e:
        logger.error("streamlit_mic_recorder not installed: %s", e)
        st.caption("⚠️ الإدخال الصوتي غير مثبّت")
        _voice_import_ok = False

    if _voice_import_ok:
        audio = mic_recorder(
            start_prompt="🎤 اضغط للتحدث",
            stop_prompt="⏹️ إيقاف التسجيل",
            just_once=True,
            use_container_width=True,
            key="voice_input",
        )
        if audio and isinstance(audio, dict) and audio.get("bytes"):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio["bytes"])
                    tmp_path = tmp.name
                with st.spinner("جارٍ تحويل الصوت إلى نص..."):
                    wmodel = load_whisper_model()
                    if wmodel is None:
                        raise RuntimeError("Whisper model failed to load")
                    result_w = wmodel.transcribe(tmp_path, language="ar")
                    voice_text = (result_w.get("text") or "").strip()
                if voice_text:
                    st.session_state.voice_query = voice_text
                    st.success(f"✓ {voice_text}")
                    st.rerun()
                else:
                    st.warning("لم يُسمع كلام واضح. حاول مرة أخرى.")
            except Exception as e:
                logger.exception("Voice transcription failed")
                st.error(f"تعذّر تحويل الصوت: {type(e).__name__}: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

    st.markdown("---")

    # Response style — horizontal to save vertical space
    st.markdown("##### 💬 أسلوب الرد")
    response_style = st.radio(
        "أسلوب",
        options=["formal", "jordanian"],
        format_func=lambda x: "رسمي" if x == "formal" else "ودود (أردني)",
        index=0 if st.session_state.response_style == "formal" else 1,
        label_visibility="collapsed",
        key="style_radio",
        horizontal=True,
    )
    st.session_state.response_style = response_style

    st.markdown("---")

    # Law domain filter
    st.markdown("##### 📖 نطاق البحث")
    selected_label = st.selectbox(
        label="القانون",
        options=list(DOMAIN_OPTIONS.keys()),
        index=list(DOMAIN_OPTIONS.keys()).index(st.session_state.selected_domain_label),
        key="domain_selectbox",
        label_visibility="collapsed",
    )
    st.session_state.selected_domain_label = selected_label
    active_domain: str | None = DOMAIN_OPTIONS[selected_label]

    st.markdown("---")

    if st.button("🔄 جلسة جديدة", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        ensure_session(st.session_state.session_id)
        st.rerun()

    # Stats + status collapsed by default — saves vertical space
    with st.expander("📊 معلومات النظام", expanded=False):
        c1, c2 = st.columns(2)
        c1.metric("النصوص", str(db_chunk_count))
        c2.metric("القوانين", "9")
        st.caption("AraBERTv02 — بحث دلالي")

        store_ready = bool(VECTOR_STORE_DIR and VECTOR_STORE_DIR.exists() and any(VECTOR_STORE_DIR.iterdir()))
        model_loaded = load_arabert() is not None
        if store_ready and db_chunk_count > 0 and model_loaded:
            st.success("النظام جاهز ✓")
        elif not store_ready or db_chunk_count == 0:
            st.warning("شغّل run_ingestion.py")
        else:
            st.warning("النموذج لم يُحمَّل")
        st.caption(f"جلسة: `{st.session_state.session_id[:8]}…`")

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.markdown(
    '<div style="background:linear-gradient(135deg,#1B3A57 0%,#2D7D8E 100%);"'
    ' class="hdr"><style>.hdr{padding:24px;border-radius:16px;margin-bottom:24px;'
    'box-shadow:0 4px 20px rgba(27,58,87,0.4);}</style>'
    '<h1 style="color:#FAF7F2;margin:0;font-size:2rem;">⚖️ JLegal-ChatBot</h1>'
    '<p style="color:#C9A961;margin:8px 0 0 0;font-size:1.05rem;">'
    'مساعدك القانوني الأردني الذكي — مدعوم بالذكاء الاصطناعي</p>'
    '<div style="margin-top:12px;display:flex;gap:12px;flex-wrap:wrap;">'
    '<span style="background:rgba(201,169,97,0.2);color:#C9A961;padding:4px 12px;border-radius:20px;font-size:0.85rem;">9 قوانين أردنية</span>'
    f'<span style="background:rgba(201,169,97,0.2);color:#C9A961;padding:4px 12px;border-radius:20px;font-size:0.85rem;">{db_chunk_count} نصاً قانونياً</span>'
    '<span style="background:rgba(201,169,97,0.2);color:#C9A961;padding:4px 12px;border-radius:20px;font-size:0.85rem;">بحث دلالي AraBERTv02</span>'
    '</div></div>',
    unsafe_allow_html=True,
)
st.caption("يستند هذا النظام حصراً إلى النصوص القانونية الأردنية المُدرجة. لا يُعدّ بديلاً عن الاستشارة القانونية المتخصصة.")

# ---------------------------------------------------------------------------
# Example questions (empty state only)
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    st.markdown("### 💡 جرّب هذه الأسئلة")
    cols = st.columns(3)
    for i, (icon, question) in enumerate(EXAMPLES):
        with cols[i % 3]:
            if st.button(f"{icon}  {question}", key=f"ex_{i}", use_container_width=True):
                st.session_state.example_query = question
                st.rerun()
    st.divider()

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_assistant_message(msg, key_suffix=str(idx))
        else:
            st.markdown(msg["content"])
            for att in msg.get("attachments", []):
                icon = "🖼️" if att["type"] == "image" else "📄"
                st.caption(f"{icon} {att['name']}")

# ---------------------------------------------------------------------------
# Bug #3: File-accepting chat input
# ---------------------------------------------------------------------------

chat_input_obj = st.chat_input(
    "اكتب سؤالك القانوني هنا... (يمكنك إرفاق صور أو ملفات PDF)",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp", "pdf"],
)

# Resolve input source: typed/file > voice > example
user_text: str | None = None
uploaded_files: list = []

if chat_input_obj:
    user_text = chat_input_obj.text if hasattr(chat_input_obj, "text") else str(chat_input_obj)
    uploaded_files = chat_input_obj.files if hasattr(chat_input_obj, "files") else []
elif not user_text:
    user_text = st.session_state.pop("voice_query", None)
if not user_text:
    user_text = st.session_state.pop("example_query", None)

# ---------------------------------------------------------------------------
# Process query (Bug #1 fix: spinner OUTSIDE assistant chat message)
# ---------------------------------------------------------------------------

if user_text or uploaded_files:
    user_text = (user_text or "").strip()

    # --- Validate and process attachments ---
    images_bytes: list[bytes] = []
    pdf_texts: list[str] = []
    attachment_meta: list[dict] = []
    validation_error: str | None = None

    for f in uploaded_files:
        data = f.read()
        if f.type and f.type.startswith("image/"):
            if len(data) > MAX_IMAGE_BYTES:
                validation_error = f"الصورة '{f.name}' تتجاوز الحد المسموح (5 ميجابايت)."
                break
            images_bytes.append(data)
            attachment_meta.append({"type": "image", "name": f.name})
        else:  # PDF
            if len(data) > MAX_PDF_BYTES:
                validation_error = f"الملف '{f.name}' يتجاوز الحد المسموح (10 ميجابايت)."
                break
            try:
                extracted = _extract_pdf_text(data)
                pdf_texts.append(extracted)
                attachment_meta.append({"type": "pdf", "name": f.name})
            except Exception as e:
                logger.exception(f"PDF extraction failed for {f.name}")
                validation_error = f"تعذّر قراءة الملف '{f.name}'. تأكد أنه ملف PDF صالح."
                break

    if validation_error:
        st.error(validation_error)
    elif user_text or images_bytes or pdf_texts:

        # Render user message
        with st.chat_message("user"):
            if user_text:
                st.markdown(user_text)
            for att in attachment_meta:
                icon = "🖼️" if att["type"] == "image" else "📄"
                st.caption(f"{icon} {att['name']}")

        st.session_state.messages.append({
            "role": "user",
            "content": user_text or "(مرفق فقط)",
            "attachments": attachment_meta,
        })

        # ── Bug #1 fix: spinner runs BETWEEN messages, NOT inside assistant bubble ──
        print(f"[app] Calling run_query with style={st.session_state.get('response_style')}")
        with st.spinner("⏳ جارٍ البحث في النصوص القانونية..."):
            result = run_query(
                query_text=user_text or "حلّل المرفق",
                session_id=st.session_state.session_id,
                law_domain=active_domain,
                style=st.session_state.response_style,
                images=images_bytes or None,
                pdf_texts=pdf_texts or None,
            )

        response_text: str = result["response_text"]
        chunks: list = result.get("chunks", [])
        query_id: str | None = result.get("query_id")

        # Render assistant message cleanly — no spinner nesting
        with st.chat_message("assistant"):
            try:
                st.markdown(response_text)
            except Exception as e:
                logger.exception("Failed to render response")
                st.error(f"خطأ في عرض الإجابة: {type(e).__name__}")

            if chunks:
                _render_sources(chunks)
            elif not result["success"]:
                st.error("عذراً، حدث خطأ أثناء معالجة سؤالك. يُرجى المحاولة مرة أخرى.")

            try:
                if user_text:
                    pdf_bytes = _generate_pdf(user_text, response_text)
                    st.download_button(
                        label="تحميل الاستشارة القانونية PDF",
                        data=pdf_bytes,
                        file_name=f"استشارة_{date.today()}.pdf",
                        mime="application/pdf",
                        key="pdf_new",
                    )
            except Exception:
                pass

            fc2 = st.columns([1, 1, 8])
            with fc2[0]:
                if st.button("👍", key=f"up_new_{query_id}"):
                    st.toast("شكراً لتقييمك!")
            with fc2[1]:
                if st.button("👎", key=f"dn_new_{query_id}"):
                    st.toast("سنعمل على التحسين!")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "chunks": chunks,
            "question": user_text,
            "query_id": query_id,
            "attachments": [],
        })
