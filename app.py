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
    initial_sidebar_state="collapsed",   # Feature 5: collapsed by default
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
    from src.database import (
        get_chunk_count,
        list_sessions,
        rename_session,
        delete_session,
        load_session_messages,
    )
    _startup_error: str | None = None
except Exception as _e:
    run_query = ensure_session = VECTOR_STORE_DIR = get_model = None  # type: ignore
    get_chunk_count = list_sessions = rename_session = delete_session = load_session_messages = None  # type: ignore
    _startup_error = str(_e)

# ---------------------------------------------------------------------------
# Cached resource loaders
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

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PDF_BYTES   = 10 * 1024 * 1024

DOMAIN_OPTIONS: dict[str, str | None] = {
    "جميع القوانين": None,
    "قانون العمل": "Labor",
    "قانون التجارة": "Commercial",
    "قانون الأحوال الشخصية": "PersonalStatus",
    "قانون الجرائم الإلكترونية": "Cybercrime",
    "نظام الخدمة المدنية": "CivilService",
    "قانون الأحوال المدنية": "CivilStatus",
    "قانون الأحوال الشخصية 2019": "PersonalStatus2019",
    "قانون السير": "TrafficLaw",
    "نظام إدارة الموارد البشرية": "HRManagement",
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
    ("⚠️", "متى يحق لصاحب العمل فصل العامل دون إنذار؟"),
    ("💼", "ما هي حقوق الموظف الحكومي في الإجازات؟"),
]

# ---------------------------------------------------------------------------
# CSS — Feature 7-9: dark blue/gray palette
# ---------------------------------------------------------------------------

RTL_CSS = """
<style>
/* ── Base ───────────────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    background-color: #1A2332 !important;
    color: #E5E7EB !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #243447 !important;
    border-left: 1px solid #2D3748;
}
[data-testid="stSidebar"] * { color: #E5E7EB !important; }
[data-testid="stSidebar"] .stButton button {
    background-color: #1A2332;
    color: #E5E7EB;
    border: 1px solid #2D3748;
    border-radius: 6px;
    text-align: right;
    direction: rtl;
    font-size: 0.83rem;
}
[data-testid="stSidebar"] .stButton button:hover {
    background-color: #3B82F6;
    color: #FFFFFF;
    border-color: #3B82F6;
}

/* ── Chat messages ───────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin: 8px 0 !important;
    border: none !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatAvatarIcon-user"]) {
    background-color: #3B82F6 !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatAvatarIcon-user"]) * {
    color: #FFFFFF !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatAvatarIcon-assistant"]) {
    background-color: #2D3748 !important;
    color: #E5E7EB !important;
}
[data-testid="stChatMessageContent"] { direction: rtl; text-align: right; }

/* ── Chat input ──────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background-color: #243447 !important;
    border: 1px solid #2D3748 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #E5E7EB !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton button {
    background-color: #3B82F6;
    color: #FFFFFF;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton button:hover {
    background-color: #2563EB;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59,130,246,0.3);
}

/* ── Source cards ────────────────────────────────────────── */
.source-card {
    background: #243447 !important;
    border-right: 4px solid #3B82F6 !important;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 8px;
    color: #E5E7EB !important;
    direction: rtl;
}
.source-article { font-size: 1rem; font-weight: 700; color: #3B82F6; }
.source-law { font-size: 0.88rem; color: #A0AEC0; margin-top: 2px; }
.score-badge {
    background: #3B82F6 !important;
    color: #FFFFFF !important;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.chunk-preview {
    color: #A0AEC0 !important;
    font-size: 0.80rem;
    margin-top: 6px;
    line-height: 1.5;
    direction: rtl;
}

/* ── Confidence card ─────────────────────────────────────── */
.confidence-card {
    background: #243447;
    border: 1px solid #2D3748;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
}
.confidence-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    direction: rtl;
}
.confidence-label { color: #A0AEC0; font-size: 0.9rem; }
.confidence-value { font-weight: 700; font-size: 1rem; }
.confidence-bar-bg {
    background: #1A2332;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}

/* ── Expanders ───────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: #243447 !important;
    border: 1px solid #2D3748 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary { color: #3B82F6 !important; }

/* ── Text ────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 { color: #E5E7EB !important; }
a { color: #3B82F6 !important; }
body, .stMarkdown, .stCaption {
    font-family: 'Segoe UI', 'Tajawal', 'Arial', sans-serif;
    direction: rtl;
}
.stDivider { border-color: #2D3748 !important; }
p, li, span { color: #E5E7EB !important; }
</style>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_bytes: bytes) -> str:
    import fitz
    parts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts).strip()[:30_000]


class _LegalPDF:
    """FPDF2 subclass with per-page header and footer for legal consultations."""

    def __init__(self, question: str, answer: str):
        from fpdf import FPDF

        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            def ar(t): return get_display(arabic_reshaper.reshape(t))
            self._has_arabic = True
        except ImportError:
            def ar(t): return t
            self._has_arabic = False
        self._ar = ar

        font_path = Path(__file__).resolve().parent / "fonts" / "Amiri-Regular.ttf"

        class PDF(FPDF):
            def __init__(self_, *a, **k):
                super().__init__(*a, **k)
                self_.set_auto_page_break(auto=True, margin=30)
                self_.set_margins(15, 28, 15)
                if font_path.exists() and self._has_arabic:
                    self_.add_font("Arabic", fname=str(font_path))
                    self_._mf = "Arabic"
                else:
                    self_._mf = "Helvetica"

            def header(self_):
                mf = self_._mf
                align = "R" if self._has_arabic else "C"
                self_.set_font(mf, size=16)
                self_.set_text_color(27, 58, 87)
                self_.cell(0, 8, ar("استشارة قانونية"), align="C", new_x="LMARGIN", new_y="NEXT")
                self_.set_font(mf, size=10)
                self_.set_text_color(80, 80, 80)
                self_.cell(0, 6, ar("JLegal-ChatBot — المساعد القانوني الأردني"), align="C", new_x="LMARGIN", new_y="NEXT")
                self_.set_font("Helvetica", size=9)
                self_.set_text_color(130, 130, 130)
                self_.cell(0, 5, ar("جامعة اليرموك — كلية تكنولوجيا المعلومات"), align="C", new_x="LMARGIN", new_y="NEXT")
                self_.cell(0, 5, date.today().strftime("%Y-%m-%d"), align=align, new_x="LMARGIN", new_y="NEXT")
                self_.ln(2)
                self_.line(15, self_.get_y(), 195, self_.get_y())
                self_.ln(4)

            def footer(self_):
                self_.set_y(-22)
                self_.line(15, self_.get_y(), 195, self_.get_y())
                self_.ln(1)
                self_.set_font("Helvetica", size=7)
                self_.set_text_color(130, 130, 130)
                disclaimer = ar(
                    "إخلاء مسؤولية: هذه الاستشارة مُولّدة بواسطة نظام ذكاء اصطناعي بناءً على نصوص قانونية أردنية مُدرجة في النظام. "
                    "لا تُعدّ بديلاً عن الاستشارة القانونية المتخصصة من محامٍ مرخّص."
                )
                self_.multi_cell(170, 3.5, disclaimer, align="C")
                self_.set_y(-8)
                self_.set_font("Helvetica", size=8)
                self_.cell(0, 5, f"صفحة {self_.page_no()} من {{nb}}", align="C")

        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()

        mf = pdf._mf
        align_body = "R" if self._has_arabic else "L"

        pdf.set_font(mf, size=12)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 8, ar("السؤال القانوني:"), align=align_body, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(mf, size=10)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 6, ar(question), align=align_body)
        pdf.ln(5)

        pdf.set_font(mf, size=12)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 8, ar("الإجابة القانونية:"), align=align_body, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(mf, size=10)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 6, ar(answer), align=align_body)

        self._bytes = bytes(pdf.output())

    def output(self) -> bytes:
        return self._bytes


def _generate_pdf(question: str, answer: str) -> bytes:
    return _LegalPDF(question, answer).output()


def _render_confidence(conf: dict) -> None:
    """Render the confidence card above an assistant answer."""
    if not conf or conf.get("score", 0) <= 0:
        return
    score_pct = int(conf["score"] * 100)
    st.markdown(
        f'<div class="confidence-card">'
        f'<div class="confidence-header">'
        f'<span class="confidence-label">ثقة الإجابة</span>'
        f'<span class="confidence-value" style="color:{conf["label_color"]}">'
        f'{score_pct}% — {conf["label"]}'
        f'</span></div>'
        f'<div class="confidence-bar-bg">'
        f'<div class="confidence-bar-fill" style="width:{score_pct}%;background:{conf["label_color"]}"></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _render_sources(chunks: list[dict]) -> None:
    with st.expander("المصادر القانونية"):
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
    _render_confidence(msg.get("confidence", {}))
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
        except Exception:
            logger.exception("PDF generation failed")

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

db_chunk_count = get_chunk_count() if get_chunk_count else 0

# ---------------------------------------------------------------------------
# Sidebar — Feature 6: chat history + compact controls
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

    # --- Feature 6: Chat history ---
    st.markdown("##### محادثاتي")
    if st.button("+ محادثة جديدة", use_container_width=True, key="new_chat_btn"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        ensure_session(st.session_state.session_id)
        st.rerun()

    if list_sessions:
        sessions = list_sessions(limit=20)
        for sess in sessions:
            sess_id = sess["session_id"]
            first_q = sess.get("first_query") or ""
            name = sess["display_name"] or (first_q[:32] + "…" if first_q else "محادثة فارغة")
            is_active = (sess_id == st.session_state.session_id)
            prefix = "● " if is_active else "  "

            cols = st.columns([7, 1, 1])
            with cols[0]:
                if st.button(prefix + name, key=f"sess_{sess_id}", use_container_width=True):
                    st.session_state.session_id = sess_id
                    st.session_state.messages = load_session_messages(sess_id)
                    st.rerun()
            with cols[1]:
                if st.button("✏", key=f"rename_{sess_id}"):
                    st.session_state[f"renaming_{sess_id}"] = True
            with cols[2]:
                if st.button("🗑", key=f"del_{sess_id}"):
                    delete_session(sess_id)
                    if sess_id == st.session_state.session_id:
                        st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.messages = []
                        ensure_session(st.session_state.session_id)
                    st.rerun()

            if st.session_state.get(f"renaming_{sess_id}"):
                new_name = st.text_input(
                    "اسم جديد",
                    value=name,
                    key=f"rename_input_{sess_id}",
                    label_visibility="collapsed",
                )
                rc = st.columns(2)
                with rc[0]:
                    if st.button("حفظ", key=f"save_{sess_id}", use_container_width=True):
                        rename_session(sess_id, new_name)
                        st.session_state[f"renaming_{sess_id}"] = False
                        st.rerun()
                with rc[1]:
                    if st.button("إلغاء", key=f"cancel_{sess_id}", use_container_width=True):
                        st.session_state[f"renaming_{sess_id}"] = False
                        st.rerun()

    st.markdown("---")

    # Style toggle — horizontal
    st.markdown("##### اسلوب الرد")
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

    # Domain filter
    st.markdown("##### نطاق البحث")
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

    # Stats + status collapsed
    with st.expander("معلومات النظام", expanded=False):
        c1, c2 = st.columns(2)
        c1.metric("النصوص", str(db_chunk_count))
        c2.metric("القوانين", "9")
        st.caption("AraBERTv02 — بحث دلالي")
        store_ready = bool(VECTOR_STORE_DIR and VECTOR_STORE_DIR.exists() and any(VECTOR_STORE_DIR.iterdir()))
        model_loaded = load_arabert() is not None
        if store_ready and db_chunk_count > 0 and model_loaded:
            st.success("النظام جاهز")
        elif not store_ready or db_chunk_count == 0:
            st.warning("شغّل run_ingestion.py")
        else:
            st.warning("النموذج لم يُحمَّل")
        st.caption(f"جلسة: `{st.session_state.session_id[:8]}…`")

# ---------------------------------------------------------------------------
# Main header — updated blue/gray palette
# ---------------------------------------------------------------------------

st.markdown(
    '<div style="background:linear-gradient(135deg,#1A2332 0%,#3B82F6 100%);"'
    ' class="hdr"><style>.hdr{padding:24px;border-radius:16px;margin-bottom:24px;'
    'box-shadow:0 4px 20px rgba(59,130,246,0.2);}</style>'
    '<h1 style="color:#FFFFFF;margin:0;font-size:2rem;">⚖️ JLegal-ChatBot</h1>'
    '<p style="color:#BFDBFE;margin:8px 0 0 0;font-size:1.05rem;">'
    'مساعدك القانوني الأردني الذكي — مدعوم بالذكاء الاصطناعي</p>'
    '<div style="margin-top:12px;display:flex;gap:12px;flex-wrap:wrap;">'
    '<span style="background:rgba(59,130,246,0.25);color:#BFDBFE;padding:4px 12px;border-radius:20px;font-size:0.85rem;">9 قوانين أردنية</span>'
    f'<span style="background:rgba(59,130,246,0.25);color:#BFDBFE;padding:4px 12px;border-radius:20px;font-size:0.85rem;">{db_chunk_count} نصاً قانونياً</span>'
    '<span style="background:rgba(59,130,246,0.25);color:#BFDBFE;padding:4px 12px;border-radius:20px;font-size:0.85rem;">بحث دلالي AraBERTv02</span>'
    '</div></div>',
    unsafe_allow_html=True,
)
st.caption("يستند هذا النظام حصراً إلى النصوص القانونية الأردنية المُدرجة. لا يُعدّ بديلاً عن الاستشارة القانونية المتخصصة.")

# ---------------------------------------------------------------------------
# Example questions (empty state)
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    st.markdown("### جرّب هذه الأسئلة")
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
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
            for att in msg.get("attachments", []):
                icon = "🖼️" if att["type"] == "image" else "📄"
                st.caption(f"{icon} {att['name']}")
    else:
        with st.chat_message("assistant", avatar="⚖️"):
            _render_assistant_message(msg, key_suffix=str(idx))

# ---------------------------------------------------------------------------
# Feature 3: Voice button inline — row above chat input
# ---------------------------------------------------------------------------

voice_cols = st.columns([1, 20])
with voice_cols[0]:
    _voice_import_ok = True
    try:
        from streamlit_mic_recorder import mic_recorder
    except ImportError as e:
        logger.error("streamlit_mic_recorder not installed: %s", e)
        st.caption("🎤")
        _voice_import_ok = False

    if _voice_import_ok:
        audio = mic_recorder(
            start_prompt="🎤",
            stop_prompt="⏹",
            just_once=True,
            use_container_width=True,
            key="voice_inline",
        )
        if audio and isinstance(audio, dict) and audio.get("bytes"):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio["bytes"])
                    tmp_path = tmp.name
                with st.spinner("تحويل الصوت..."):
                    wmodel = load_whisper_model()
                    if wmodel is None:
                        raise RuntimeError("Whisper model failed to load")
                    result_w = wmodel.transcribe(tmp_path, language="ar")
                    voice_text = (result_w.get("text") or "").strip()
                if voice_text:
                    st.session_state.voice_query = voice_text
                    st.rerun()
                else:
                    st.warning("لم يُسمع كلام واضح.")
            except Exception as e:
                logger.exception("Voice transcription failed")
                st.error(f"خطأ: {type(e).__name__}: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

# ---------------------------------------------------------------------------
# Chat input with file support
# ---------------------------------------------------------------------------

chat_input_obj = st.chat_input(
    "اكتب سؤالك القانوني هنا... (يمكنك إرفاق صور أو ملفات PDF)",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp", "pdf"],
)

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
# Process query
# ---------------------------------------------------------------------------

if user_text or uploaded_files:
    user_text = (user_text or "").strip()

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
        else:
            if len(data) > MAX_PDF_BYTES:
                validation_error = f"الملف '{f.name}' يتجاوز الحد المسموح (10 ميجابايت)."
                break
            try:
                pdf_texts.append(_extract_pdf_text(data))
                attachment_meta.append({"type": "pdf", "name": f.name})
            except Exception as e:
                logger.exception(f"PDF extraction failed for {f.name}")
                validation_error = f"تعذّر قراءة الملف '{f.name}'."
                break

    if validation_error:
        st.error(validation_error)
    elif user_text or images_bytes or pdf_texts:
        with st.chat_message("user", avatar="👤"):
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

        print(f"[app] Calling run_query with style={st.session_state.get('response_style')}")
        with st.spinner("جارٍ البحث في النصوص القانونية..."):
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
        conf: dict = result.get("confidence", {})

        with st.chat_message("assistant", avatar="⚖️"):
            try:
                _render_confidence(conf)
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
            "confidence": conf,
        })
