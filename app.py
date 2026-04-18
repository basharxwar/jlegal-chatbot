import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import streamlit as st

# set_page_config MUST be the very first Streamlit call — nothing else before it
st.set_page_config(
    page_title="JLegal-ChatBot | المساعد القانوني الأردني",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# All remaining imports come AFTER set_page_config
# ---------------------------------------------------------------------------

import uuid
from datetime import date
from pathlib import Path

try:
    from src.pipeline import run_query, ensure_session
    from src.vector_store import VECTOR_STORE_DIR, _load_domain
    _startup_error: str | None = None
except Exception as _e:
    run_query = ensure_session = VECTOR_STORE_DIR = _load_domain = None  # type: ignore[assignment]
    _startup_error = str(_e)

# ---------------------------------------------------------------------------
# Preload all vector collections into memory once at startup
# ---------------------------------------------------------------------------

@st.cache_resource
def preload_collections():
    """Load all domain JSON files into memory so the first query is fast."""
    if _load_domain is None:
        return False
    for domain in ["Labor", "Commercial", "PersonalStatus", "Cybercrime", "CivilService"]:
        _load_domain(domain)
    return True

preload_collections()

# ---------------------------------------------------------------------------
# RTL + custom CSS
# ---------------------------------------------------------------------------

RTL_CSS = """
<style>
/* Force RTL for all Streamlit text elements */
body, .stApp, .stChatMessage, .stMarkdown, .stExpander,
.stSelectbox, .stTextInput, .stButton, .stSidebar {
    direction: rtl;
    text-align: right;
    font-family: 'Segoe UI', 'Arial', sans-serif;
}

/* Chat message bubbles */
.stChatMessage [data-testid="stChatMessageContent"] {
    direction: rtl;
    text-align: right;
}

/* Source card */
.source-card {
    border-right: 3px solid #1f77b4;
    padding: 8px 12px;
    margin-bottom: 8px;
    background: #f8fbfe;
    border-radius: 4px;
    font-size: 0.85rem;
}

/* Score badge */
.score-badge {
    background: #e8f4f8;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 0.78rem;
    color: #1f77b4;
    font-weight: bold;
}

/* Chunk text preview */
.chunk-preview {
    color: #555;
    font-size: 0.80rem;
    margin-top: 4px;
    line-height: 1.4;
    direction: rtl;
}

/* Sidebar header */
.sidebar-logo {
    font-size: 1.6rem;
    font-weight: bold;
    color: #1f3c88;
    text-align: center;
    padding: 10px 0;
}
</style>
"""

# ---------------------------------------------------------------------------
# Domain mapping (display label -> vector store collection name)
# ---------------------------------------------------------------------------

DOMAIN_OPTIONS: dict[str, str | None] = {
    "جميع القوانين | All Laws": None,
    "قانون العمل | Labor": "Labor",
    "قانون التجارة | Commercial": "Commercial",
    "قانون الأحوال الشخصية | Personal Status": "PersonalStatus",
    "قانون الجرائم الإلكترونية | Cybercrime": "Cybercrime",
    "نظام الخدمة المدنية | Civil Service": "CivilService",
}

LAW_NAMES_AR: dict[str, str] = {
    "Labor": "قانون العمل الأردني",
    "Commercial": "قانون التجارة الأردني",
    "PersonalStatus": "قانون الأحوال الشخصية",
    "Cybercrime": "قانون الجرائم الإلكترونية الأردني",
    "CivilService": "نظام الخدمة المدنية",
}

# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def _generate_pdf(question: str, answer: str) -> bytes:
    """Generate a simple PDF for the Q&A consultation using fpdf2."""
    from fpdf import FPDF

    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        def ar(text: str) -> str:
            return get_display(arabic_reshaper.reshape(text))

        has_arabic = True
    except ImportError:
        def ar(text: str) -> str:
            return text

        has_arabic = False

    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=20)

    # Try to load an Arabic font from the fonts/ directory
    font_path = Path(__file__).resolve().parent / "fonts" / "Amiri-Regular.ttf"
    if font_path.exists() and has_arabic:
        pdf.add_font("Arabic", fname=str(font_path))
        mf = "Arabic"
    else:
        mf = "Helvetica"

    align = "R" if has_arabic else "L"

    # Header
    pdf.set_font(mf, size=16)
    pdf.set_text_color(31, 60, 136)
    pdf.cell(0, 12, ar("استشارة قانونية - JLegal-ChatBot"), align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, f"JLegal-ChatBot  |  {date.today().strftime('%Y-%m-%d')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # Question
    pdf.set_font(mf, size=12)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, ar("السؤال القانوني:"), align=align, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(mf, size=10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, ar(question), align=align)
    pdf.ln(5)

    # Answer
    pdf.set_font(mf, size=12)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, ar("الإجابة القانونية:"), align=align, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(mf, size=10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, ar(answer), align=align)

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Helvetica", size=8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "JLegal-ChatBot - Yarmouk University", align="C")

    return bytes(pdf.output())


def _render_sources(chunks: list[dict]) -> None:
    """Render the sources expander with article info and text preview."""
    with st.expander("📚 المصادر القانونية"):
        for chunk in chunks:
            law_name = (
                LAW_NAMES_AR.get(chunk.get("law_domain", ""), "")
                or chunk.get("law_name_ar", "")
            )
            article = chunk.get("article_number")
            article_label = f"المادة {article}" if article else "نص قانوني"
            score = chunk.get("score", 0.0)
            rank = chunk.get("rank", "—")
            preview = (chunk.get("chunk_text", "")[:120] + "…") if chunk.get("chunk_text") else ""

            st.markdown(
                f'<div class="source-card">'
                f"<strong>{article_label}</strong> — {law_name} "
                f'<span class="score-badge">تشابه: {score:.0%}</span> '
                f"(ترتيب: {rank})"
                f'<div class="chunk-preview">{preview}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        ensure_session(st.session_state.session_id)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_domain_label" not in st.session_state:
        st.session_state.selected_domain_label = list(DOMAIN_OPTIONS.keys())[0]


# ---------------------------------------------------------------------------
# Main app — everything below runs on every Streamlit script rerun
# ---------------------------------------------------------------------------

st.markdown(RTL_CSS, unsafe_allow_html=True)

# Surface any startup import error in Arabic only
if _startup_error:
    st.error("حدث خطأ عند تشغيل النظام. تأكد من تثبيت جميع المكتبات وتشغيل run_ingestion.py أولاً.")
    st.stop()

_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">⚖️ JLegal-ChatBot<br>'
        '<small style="font-size:0.9rem">المساعد القانوني الأردني</small></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Law domain filter
    st.markdown("**نطاق البحث القانوني**")
    selected_label = st.selectbox(
        label="اختر القانون",
        options=list(DOMAIN_OPTIONS.keys()),
        index=list(DOMAIN_OPTIONS.keys()).index(
            st.session_state.selected_domain_label
        ),
        key="domain_selectbox",
        label_visibility="collapsed",
    )
    st.session_state.selected_domain_label = selected_label
    active_domain: str | None = DOMAIN_OPTIONS[selected_label]

    st.divider()

    # New session button
    if st.button("🔄 جلسة جديدة", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        ensure_session(st.session_state.session_id)
        st.rerun()

    st.divider()

    # System status
    st.markdown("**حالة النظام**")
    store_ready = bool(VECTOR_STORE_DIR and VECTOR_STORE_DIR.exists() and any(VECTOR_STORE_DIR.iterdir()))
    if store_ready:
        st.success("قاعدة البيانات القانونية جاهزة ✓")
    else:
        st.warning("قاعدة البيانات غير مُهيأة — شغّل run_ingestion.py أولاً")

    st.caption(f"معرّف الجلسة: `{st.session_state.session_id[:8]}…`")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.markdown("## ⚖️ المساعد القانوني الأردني")
st.caption(
    "يستند هذا النظام حصراً إلى النصوص القانونية الأردنية المُدرجة. "
    "لا يُعدّ بديلاً عن الاستشارة القانونية المتخصصة."
)
st.divider()

# Render existing chat history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("chunks"):
                _render_sources(msg["chunks"])

            # PDF download button for each past answer
            if msg.get("question"):
                try:
                    pdf_bytes = _generate_pdf(msg["question"], msg["content"])
                    st.download_button(
                        label="تحميل الاستشارة القانونية PDF",
                        data=pdf_bytes,
                        file_name=f"استشارة_قانونية_{date.today()}.pdf",
                        mime="application/pdf",
                        key=f"pdf_history_{idx}",
                    )
                except Exception:
                    pass  # PDF generation failure should not break the UI

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

user_input = st.chat_input("اكتب سؤالك القانوني هنا...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append(
        {"role": "user", "content": user_input, "chunks": [], "question": None}
    )

    with st.chat_message("assistant"):
        with st.spinner("جارٍ البحث في النصوص القانونية..."):
            result = run_query(
                query_text=user_input,
                session_id=st.session_state.session_id,
                law_domain=active_domain,
            )

        response_text: str = result["response_text"]
        chunks: list = result.get("chunks", [])

        st.markdown(response_text)

        if chunks:
            _render_sources(chunks)
        elif not result["success"]:
            st.error("عذراً، حدث خطأ أثناء معالجة سؤالك. يُرجى المحاولة مرة أخرى.")

        # PDF download button for the new answer
        try:
            pdf_bytes = _generate_pdf(user_input, response_text)
            st.download_button(
                label="تحميل الاستشارة القانونية PDF",
                data=pdf_bytes,
                file_name=f"استشارة_قانونية_{date.today()}.pdf",
                mime="application/pdf",
                key="pdf_new",
            )
        except Exception:
            pass

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "chunks": chunks,
            "question": user_input,
        }
    )
