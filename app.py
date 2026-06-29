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
DEMO_QUESTION = "这篇论文的核心贡献是什么？"
MAX_HISTORY = 12


def init_state():
    """Initialize Streamlit session state."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("rag_ready", False)
    st.session_state.setdefault("current_pdf", "")
    st.session_state.setdefault("chunk_count", 0)
    st.session_state.setdefault("retrieval_backend", "未构建")
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
    with st.spinner("Agent 正在拆解任务、调度多模态工具并融合结果..."):
        response = run_agent(prompt, st.session_state.get("pdf_data"))
    add_message("assistant", response)


def load_demo_pdf():
    """Load the DFT demo paper and build the RAG index."""
    if not DEMO_PDF_PATH.exists():
        st.error("未找到示例论文，请确认 data/docs 目录中存在 DFT PDF。")
        return False

    with st.spinner("正在加载示例论文并构建知识库..."):
        try:
            info = build_pdf_knowledge_base(DEMO_PDF_PATH, DEMO_PDF_PATH.name)
            st.success(
                f"示例论文已加载：{info.get('chunk_count', 0)} 个片段，"
                f"检索后端：{info.get('backend', 'unknown')}"
            )
            return True
        except Exception as exc:
            st.error(f"Demo 知识库构建失败：{exc}")
            return False


def render_sidebar():
    """Render sidebar navigation and system status."""
    with st.sidebar:
        st.markdown("## AI Research Assistant Pro")
        st.caption("面向论文阅读、科研学习与比赛答辩的智能助手")

        st.divider()
        st.markdown("### 功能模块")
        st.markdown("- 💬 智能对话")
        st.markdown("- 📄 论文总结")
        st.markdown("- 🧠 Agent任务拆解")
        st.markdown("- 📚 RAG知识库问答")
        st.markdown("- 📊 图表/公式解析")

        st.divider()
        st.markdown("### 系统状态")
        st.markdown("**模型：** 通义千问 qwen-turbo")
        st.markdown("**RAG：** 已启用")
        st.markdown("**Agent：** 多模态融合")

        if st.session_state.get("rag_ready"):
            st.success("知识库：已构建")
            st.caption(f"当前论文：{st.session_state.get('current_pdf')}")
            st.caption(f"文本片段：{st.session_state.get('chunk_count')}")
            st.caption(f"检索后端：{st.session_state.get('retrieval_backend')}")
        else:
            st.warning("知识库：待上传PDF")

        st.divider()
        if st.button("一键Demo演示", use_container_width=True):
            if load_demo_pdf():
                run_user_query(DEMO_QUESTION)
                st.rerun()

        if st.button("清空对话", use_container_width=True):
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
        ("📚 文本RAG", "上传论文后对正文切分、向量化，并使用FAISS动态检索相关内容。"),
        ("📊 多模态工具", "提取作者、图表、表格、公式和字数结构信息，辅助论文分析。"),
        ("🧠 Agent融合", "自动拆解任务，选择RAG与工具，并统一生成结构化科研回答。"),
    ]

    for col, (title, text) in zip((col1, col2, col3), cards):
        with col:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-title">{title}</div>
                    <div class="card-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_upload_area():
    """Render PDF upload area."""
    st.markdown("### 📤 上传论文区")
    with st.container(border=True):
        uploaded_pdf = st.file_uploader("上传论文PDF", type=["pdf"])

        if uploaded_pdf is not None:
            with st.spinner("正在解析 PDF 并构建 RAG 知识库..."):
                try:
                    file_path = save_uploaded_pdf(uploaded_pdf)
                    info = build_pdf_knowledge_base(file_path, uploaded_pdf.name)
                    st.success(
                        f"知识库构建完成：{info.get('chunk_count', 0)} 个文本片段，"
                        f"检索后端：{info.get('backend', 'unknown')}，"
                        f"图表：{info.get('figure_count', 0) + info.get('table_count', 0)}，"
                        f"公式：{info.get('formula_count', 0)}"
                    )
                except Exception as exc:
                    st.session_state["rag_ready"] = False
                    st.error(f"知识库构建失败：{exc}")

        if st.session_state.get("rag_ready"):
            st.markdown(
                f"""
                **当前文档：** {st.session_state.get("current_pdf")}  
                **文本片段：** {st.session_state.get("chunk_count")}  
                **检索后端：** {st.session_state.get("retrieval_backend")}
                """
            )
        else:
            st.info("上传 PDF 后，系统会自动构建可检索的论文知识库。")


def render_assistant_message(content):
    """Render Agent output as structured markdown."""
    sections = [part.strip() for part in content.split("\n\n---\n\n") if part.strip()]
    for section in sections:
        st.markdown(section)
        st.divider()


def render_chat_area():
    """Render ChatGPT-style conversation area."""
    st.markdown("### 💬 问答区")

    if not st.session_state["messages"]:
        with st.chat_message("assistant"):
            st.markdown(
                """
                你好，我是 **AI Research Assistant Pro**。  
                你可以上传论文 PDF，然后问我：

                - 这篇论文的核心贡献是什么？
                - 总结这篇论文并给我学习计划
                - 统计这篇论文大概有多少词
                - 图1讲的是什么？
                - 这篇论文有哪些图表或公式？
                - 作者是谁？这篇论文多少字？
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

    st.title("🧠 AI Research Assistant Pro")
    st.caption("Multimodal Scientific AI Assistant：RAG + 图表/公式/元信息工具 + Agent融合")

    render_status_cards()
    st.divider()

    render_upload_area()
    st.divider()

    st.markdown("### 🧠 Agent执行结果展示区")
    render_chat_area()

    prompt = st.chat_input("请输入你的问题，例如：总结这篇论文并生成学习计划")
    if prompt:
        run_user_query(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
