"""Streamlit UI for AI Research Assistant Pro.

This file only handles product interface, file upload, chat history, and calls
the Agent entry points. AI logic lives in agent/, rag/, llm/, and tools/.
"""

from pathlib import Path

import streamlit as st

from agent.core_agent import ingest_pdf, run_agent


BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "data" / "docs"
DEMO_PDF_PATH = DOCS_DIR / "on_chip_test_infrastructure_dft.pdf"
DEMO_QUESTION = "\u8fd9\u7bc7\u8bba\u6587\u7684\u6838\u5fc3\u8d21\u732e\u662f\u4ec0\u4e48\uff1f"
MAX_HISTORY = 12
APP_VERSION = "2026-06-29-answer-v5"


def init_state():
    """Initialize Streamlit session state."""
    if st.session_state.get("app_version") != APP_VERSION:
        st.session_state.clear()
        st.session_state["app_version"] = APP_VERSION

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("rag_ready", False)
    st.session_state.setdefault("current_pdf", "")
    st.session_state.setdefault("chunk_count", 0)
    st.session_state.setdefault("retrieval_backend", "\u672a\u6784\u5efa")
    st.session_state.setdefault("pdf_data", None)


def add_message(role, content):
    """Append a chat message and keep recent history."""
    st.session_state["messages"].append({"role": role, "content": content})
    st.session_state["messages"] = st.session_state["messages"][-MAX_HISTORY:]


def save_uploaded_pdf(uploaded_file):
    """Save uploaded PDF to the project docs folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DOCS_DIR / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


def set_kb_status(file_name, info):
    """Update knowledge-base status shown in the UI."""
    st.session_state["rag_ready"] = True
    st.session_state["current_pdf"] = file_name
    st.session_state["chunk_count"] = info.get("chunk_count", 0)
    st.session_state["retrieval_backend"] = info.get("backend", "unknown")
    st.session_state["pdf_data"] = info.get("pdf_data")


def build_pdf_knowledge_base(file_path, display_name):
    """Ask the Agent layer to ingest a PDF."""
    info = ingest_pdf(str(file_path))
    set_kb_status(display_name, info)
    return info


def run_user_query(prompt):
    """Run Agent and append user/assistant messages."""
    add_message("user", prompt)
    with st.spinner("Agent \u6b63\u5728\u62c6\u89e3\u4efb\u52a1\u3001\u8c03\u5ea6\u591a\u6a21\u6001\u5de5\u5177\u5e76\u878d\u5408\u7ed3\u679c..."):
        response = run_agent(prompt, st.session_state.get("pdf_data"))
    add_message("assistant", response)


def load_demo_pdf():
    """Load the DFT demo paper and build the RAG index."""
    if not DEMO_PDF_PATH.exists():
        st.error("\u672a\u627e\u5230\u793a\u4f8b\u8bba\u6587\uff0c\u8bf7\u786e\u8ba4 data/docs \u76ee\u5f55\u4e2d\u5b58\u5728 DFT PDF\u3002")
        return False

    with st.spinner("\u6b63\u5728\u52a0\u8f7d\u793a\u4f8b\u8bba\u6587\u5e76\u6784\u5efa\u77e5\u8bc6\u5e93..."):
        try:
            info = build_pdf_knowledge_base(DEMO_PDF_PATH, DEMO_PDF_PATH.name)
            st.success(
                f"\u793a\u4f8b\u8bba\u6587\u5df2\u52a0\u8f7d\uff1a{info.get('chunk_count', 0)} \u4e2a\u7247\u6bb5\uff0c"
                f"\u68c0\u7d22\u540e\u7aef\uff1a{info.get('backend', 'unknown')}"
            )
            return True
        except Exception as exc:
            st.error(f"Demo \u77e5\u8bc6\u5e93\u6784\u5efa\u5931\u8d25\uff1a{exc}")
            return False


def render_sidebar():
    """Render sidebar navigation and system status."""
    with st.sidebar:
        st.markdown("## AI Research Assistant Pro")
        st.caption("\u9762\u5411\u8bba\u6587\u9605\u8bfb\u3001\u79d1\u7814\u5b66\u4e60\u4e0e\u6bd4\u8d5b\u7b54\u8fa9\u7684\u667a\u80fd\u52a9\u624b")
        st.caption(f"Version: {APP_VERSION}")

        st.divider()
        st.markdown("### \u529f\u80fd\u6a21\u5757")
        st.markdown("- \U0001F4AC \u667a\u80fd\u5bf9\u8bdd")
        st.markdown("- \U0001F4C4 \u8bba\u6587\u603b\u7ed3")
        st.markdown("- \U0001F9E0 Agent\u4efb\u52a1\u62c6\u89e3")
        st.markdown("- \U0001F4DA RAG\u77e5\u8bc6\u5e93\u95ee\u7b54")
        st.markdown("- \U0001F9EE \u56fe\u8868/\u516c\u5f0f\u89e3\u6790")

        st.divider()
        st.markdown("### \u7cfb\u7edf\u72b6\u6001")
        st.markdown("**\u6a21\u578b\uff1a** \u901a\u4e49\u5343\u95ee qwen-turbo")
        st.markdown("**RAG\uff1a** \u5df2\u542f\u7528")
        st.markdown("**Agent\uff1a** \u591a\u6a21\u6001\u878d\u5408")

        if st.session_state.get("rag_ready"):
            st.success("\u77e5\u8bc6\u5e93\uff1a\u5df2\u6784\u5efa")
            st.caption(f"\u5f53\u524d\u8bba\u6587\uff1a{st.session_state.get('current_pdf')}")
            st.caption(f"\u6587\u672c\u7247\u6bb5\uff1a{st.session_state.get('chunk_count')}")
            st.caption(f"\u68c0\u7d22\u540e\u7aef\uff1a{st.session_state.get('retrieval_backend')}")
        else:
            st.warning("\u77e5\u8bc6\u5e93\uff1a\u5f85\u4e0a\u4f20PDF")

        st.divider()
        if st.button("\u4e00\u952eDemo\u6f14\u793a", use_container_width=True):
            if load_demo_pdf():
                run_user_query(DEMO_QUESTION)
                st.rerun()

        if st.button("\u6e05\u7a7a\u5bf9\u8bdd", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()


def inject_style():
    """Add lightweight product styling."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
        }
        .info-card {
            border: 1px solid rgba(49, 51, 63, 0.16);
            border-radius: 8px;
            padding: 16px;
            min-height: 112px;
            background: rgba(250, 250, 250, 0.76);
        }
        .card-title {
            font-weight: 700;
            font-size: 1.02rem;
            margin-bottom: 8px;
        }
        .card-text {
            color: rgba(49, 51, 63, 0.72);
            line-height: 1.55;
            font-size: 0.94rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status_cards():
    """Render compact capability cards."""
    col1, col2, col3 = st.columns(3)

    cards = [
        ("\U0001F4DA \u6587\u672cRAG", "\u4e0a\u4f20\u8bba\u6587\u540e\u5bf9\u6b63\u6587\u5207\u5206\u3001\u5411\u91cf\u5316\uff0c\u5e76\u4f7f\u7528 FAISS \u52a8\u6001\u68c0\u7d22\u76f8\u5173\u5185\u5bb9\u3002"),
        ("\U0001F9EE \u591a\u6a21\u6001\u5de5\u5177", "\u63d0\u53d6\u4f5c\u8005\u3001\u56fe\u8868\u3001\u8868\u683c\u3001\u516c\u5f0f\u548c\u5b57\u6570\u7ed3\u6784\u4fe1\u606f\uff0c\u8f85\u52a9\u8bba\u6587\u5206\u6790\u3002"),
        ("\U0001F9E0 Agent\u878d\u5408", "\u81ea\u52a8\u62c6\u89e3\u4efb\u52a1\uff0c\u9009\u62e9 RAG \u4e0e\u5de5\u5177\uff0c\u5e76\u7edf\u4e00\u751f\u6210\u7ed3\u6784\u5316\u79d1\u7814\u56de\u7b54\u3002"),
    ]

    for col, (title, text) in zip((col1, col2, col3), cards):
        with col:
            st.markdown(
                f"""
                <div class=\"info-card\">
                    <div class=\"card-title\">{title}</div>
                    <div class=\"card-text\">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_upload_area():
    """Render PDF upload area."""
    st.markdown("### \U0001F4E4 \u4e0a\u4f20\u8bba\u6587\u533a")
    with st.container(border=True):
        uploaded_pdf = st.file_uploader("\u4e0a\u4f20\u8bba\u6587PDF", type=["pdf"])

        if uploaded_pdf is not None:
            with st.spinner("\u6b63\u5728\u89e3\u6790 PDF \u5e76\u6784\u5efa RAG \u77e5\u8bc6\u5e93..."):
                try:
                    file_path = save_uploaded_pdf(uploaded_pdf)
                    info = build_pdf_knowledge_base(file_path, uploaded_pdf.name)
                    st.success(
                        f"\u77e5\u8bc6\u5e93\u6784\u5efa\u5b8c\u6210\uff1a{info.get('chunk_count', 0)} \u4e2a\u6587\u672c\u7247\u6bb5\uff0c"
                        f"\u68c0\u7d22\u540e\u7aef\uff1a{info.get('backend', 'unknown')}\uff1b"
                        f"\u56fe\u8868\uff1a{info.get('figure_count', 0) + info.get('table_count', 0)}\uff1b"
                        f"\u516c\u5f0f\uff1a{info.get('formula_count', 0)}"
                    )
                except Exception as exc:
                    st.session_state["rag_ready"] = False
                    st.error(f"\u77e5\u8bc6\u5e93\u6784\u5efa\u5931\u8d25\uff1a{exc}")

        if st.session_state.get("rag_ready"):
            st.markdown(
                f"""
                **\u5f53\u524d\u6587\u6863\uff1a** {st.session_state.get("current_pdf")}  
                **\u6587\u672c\u7247\u6bb5\uff1a** {st.session_state.get("chunk_count")}  
                **\u68c0\u7d22\u540e\u7aef\uff1a** {st.session_state.get("retrieval_backend")}
                """
            )
        else:
            st.info("\u4e0a\u4f20 PDF \u540e\uff0c\u7cfb\u7edf\u4f1a\u81ea\u52a8\u6784\u5efa\u53ef\u68c0\u7d22\u7684\u8bba\u6587\u77e5\u8bc6\u5e93\u3002")


def render_assistant_message(content):
    """Render Agent output as structured markdown."""
    sections = [part.strip() for part in content.split("\n\n---\n\n") if part.strip()]
    for section in sections:
        st.markdown(section)
        st.divider()


def render_chat_area():
    """Render ChatGPT-style conversation area."""
    st.markdown("### \U0001F4AC \u95ee\u7b54\u533a")

    if not st.session_state["messages"]:
        with st.chat_message("assistant"):
            st.markdown(
                """
                \u4f60\u597d\uff0c\u6211\u662f **AI Research Assistant Pro**\u3002  
                \u4f60\u53ef\u4ee5\u5148\u4e0a\u4f20\u8bba\u6587 PDF\uff0c\u7136\u540e\u95ee\u6211\uff1a

                - \u8fd9\u7bc7\u8bba\u6587\u7684\u6838\u5fc3\u8d21\u732e\u662f\u4ec0\u4e48\uff1f
                - \u603b\u7ed3\u8fd9\u7bc7\u8bba\u6587\u5e76\u7ed9\u6211\u5b66\u4e60\u8ba1\u5212
                - \u7edf\u8ba1\u8fd9\u7bc7\u8bba\u6587\u5927\u6982\u6709\u591a\u5c11\u8bcd
                - \u56fe1\u8bb2\u7684\u662f\u4ec0\u4e48\uff1f
                - \u8fd9\u7bc7\u8bba\u6587\u6709\u54ea\u4e9b\u56fe\u8868\u6216\u516c\u5f0f\uff1f
                - \u4f5c\u8005\u662f\u8c01\uff1f\u8fd9\u7bc7\u8bba\u6587\u591a\u5c11\u5b57\uff1f
                """
            )

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message["content"])
            else:
                st.markdown(message["content"])


def main():
    """Render the product UI."""
    st.set_page_config(page_title="AI Research Assistant Pro", layout="wide")
    init_state()
    inject_style()
    render_sidebar()

    st.title("\U0001F9E0 AI Research Assistant Pro")
    st.caption("Multimodal Scientific AI Assistant\uff1aRAG + \u56fe\u8868/\u516c\u5f0f/\u5143\u4fe1\u606f\u5de5\u5177 + Agent\u878d\u5408")

    render_status_cards()
    st.divider()

    render_upload_area()
    st.divider()

    st.markdown("### \U0001F9E0 Agent\u6267\u884c\u7ed3\u679c\u5c55\u793a\u533a")
    render_chat_area()

    prompt = st.chat_input("\u8bf7\u8f93\u5165\u4f60\u7684\u95ee\u9898\uff0c\u4f8b\u5982\uff1a\u603b\u7ed3\u8fd9\u7bc7\u8bba\u6587\u5e76\u751f\u6210\u5b66\u4e60\u8ba1\u5212")
    if prompt:
        run_user_query(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
