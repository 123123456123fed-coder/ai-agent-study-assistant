"""Streamlit UI for AI Research Assistant Pro.

The UI owns paper management, chat history, and presentation. AI logic remains
in agent/, rag/, llm/, and tools/.
"""

from pathlib import Path
import re

import streamlit as st

st.set_page_config(
    page_title="AI Research Assistant Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

from agent.core_agent import ingest_pdf, run_agent
from rag import retriever


BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "data" / "docs"
DEMO_PDF_PATH = DOCS_DIR / "on_chip_test_infrastructure_dft.pdf"
DEMO_QUESTION = "这篇论文的核心贡献是什么？"
MAX_HISTORY = 16
APP_VERSION = "2026-07-01-product-v25"


def init_state():
    """Initialize Streamlit session state."""
    if st.session_state.get("app_version") != APP_VERSION:
        st.session_state.clear()
        st.session_state["app_version"] = APP_VERSION

    st.session_state.setdefault("papers", {})
    st.session_state.setdefault("current_paper", "")
    st.session_state.setdefault("current_paper_id", "")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_query", "")
    st.session_state.setdefault("last_retrieval", [])
    st.session_state.setdefault("rag_ready", False)
    st.session_state.setdefault("current_pdf", "")
    st.session_state.setdefault("chunk_count", 0)
    st.session_state.setdefault("retrieval_backend", "未构建")
    st.session_state.setdefault("pdf_data", None)
    st.session_state.setdefault("show_left_panel", True)
    st.session_state.setdefault("show_right_panel", True)


def current_paper_record():
    """Return the selected paper record."""
    name = st.session_state.get("current_paper")
    if not name:
        return None
    return st.session_state["papers"].get(name)


def current_pdf_data():
    """Return selected paper data for Agent/tools."""
    record = current_paper_record()
    if not record:
        return None
    return record.get("pdf_data")


def sync_current_status(record):
    """Sync selected paper record to legacy UI state fields."""
    if not record:
        st.session_state["rag_ready"] = False
        st.session_state["current_pdf"] = ""
        st.session_state["current_paper_id"] = ""
        st.session_state["chunk_count"] = 0
        st.session_state["retrieval_backend"] = "未构建"
        st.session_state["pdf_data"] = None
        return

    st.session_state["rag_ready"] = True
    st.session_state["current_pdf"] = record.get("name", "")
    st.session_state["current_paper_id"] = record.get("paper_id", record.get("name", ""))
    st.session_state["chunk_count"] = record.get("chunk_count", 0)
    st.session_state["retrieval_backend"] = record.get("backend", "unknown")
    st.session_state["pdf_data"] = record.get("pdf_data")


def add_message(role, content):
    """Append a chat message and keep recent history."""
    paper = st.session_state.get("current_paper", "")
    st.session_state["messages"].append({"role": role, "content": content, "paper": paper})
    st.session_state["messages"] = st.session_state["messages"][-MAX_HISTORY:]


def save_uploaded_pdf(uploaded_file):
    """Save uploaded PDF to the project docs folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DOCS_DIR / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


def paper_counts(pdf_data):
    """Return basic counts from unified paper data."""
    if not isinstance(pdf_data, dict):
        return {"figures": 0, "tables": 0, "formulas": 0, "words": 0}

    text = pdf_data.get("text", "")
    words = re.findall(r"[A-Za-z0-9_+\-/.]+|[\u4e00-\u9fff]", text)
    return {
        "figures": len(pdf_data.get("figures", [])),
        "tables": len(pdf_data.get("tables", [])),
        "formulas": len(pdf_data.get("formulas", [])),
        "words": len(words),
    }


def build_paper_record(file_path, display_name):
    """Ingest a PDF and create a session paper record."""
    paper_id = display_name
    info = ingest_pdf(str(file_path), paper_id=paper_id)
    pdf_data = info.get("pdf_data", {})
    counts = paper_counts(pdf_data)
    return {
        "name": display_name,
        "paper_id": paper_id,
        "file_path": str(file_path),
        "pdf_data": pdf_data,
        "chunk_count": info.get("chunk_count", 0),
        "backend": info.get("backend", "unknown"),
        "index_size": info.get("index_size", 0),
        "figure_count": counts["figures"],
        "table_count": counts["tables"],
        "formula_count": counts["formulas"],
        "word_count": counts["words"],
    }


def select_paper(name):
    """Switch current paper and activate its isolated RAG index."""
    if not name:
        sync_current_status(None)
        return

    record = st.session_state["papers"].get(name)
    if not record:
        sync_current_status(None)
        return

    paper_id = record.get("paper_id", name)
    if not retriever.has_index(paper_id):
        record = build_paper_record(Path(record["file_path"]), name)
        st.session_state["papers"][name] = record
        paper_id = record.get("paper_id", name)

    retriever.set_active_paper(paper_id)
    st.session_state["current_paper"] = name
    st.session_state["current_paper_id"] = paper_id
    st.session_state["last_retrieval"] = []
    sync_current_status(record)


def load_demo_pdf():
    """Load the DFT demo paper and build the RAG index."""
    if not DEMO_PDF_PATH.exists():
        st.error("未找到示例论文，请确认 data/docs 目录中存在 DFT PDF。")
        return

    with st.spinner("正在加载示例论文并构建知识库..."):
        record = build_paper_record(DEMO_PDF_PATH, DEMO_PDF_PATH.name)
        st.session_state["papers"][DEMO_PDF_PATH.name] = record
        st.session_state["current_paper"] = DEMO_PDF_PATH.name
        st.session_state["current_paper_id"] = record["paper_id"]
        sync_current_status(record)
        add_message("user", DEMO_QUESTION)
        add_message("assistant", run_agent(DEMO_QUESTION, record["pdf_data"], paper_id=record["paper_id"]))
        st.rerun()


def run_user_query(prompt):
    """Run Agent and append user/assistant messages."""
    record = current_paper_record()
    if not record:
        add_message("user", prompt)
        add_message("assistant", "请先在左侧上传或选择一篇论文，然后再开始提问。")
        return

    select_paper(record["name"])
    pdf_data = current_pdf_data()
    paper_id = st.session_state.get("current_paper_id") or record.get("paper_id", record["name"])
    st.session_state["last_query"] = prompt
    st.session_state["last_retrieval"] = retriever.search(prompt, top_k=3, paper_id=paper_id)

    add_message("user", prompt)
    with st.spinner("AI正在分析论文..."):
        response = run_agent(prompt, pdf_data, paper_id=paper_id)
    add_message("assistant", response)


def extract_keywords(pdf_data, limit=8):
    """Extract lightweight keywords for the right-side panel."""
    if not isinstance(pdf_data, dict):
        return []

    text = pdf_data.get("text", "")
    preferred = [
        "multi-site testing",
        "ATE",
        "E-RPCT",
        "RAG",
        "SOC",
        "test throughput",
        "vector memory",
        "abort-on-fail",
        "stimuli broadcast",
        "contact yield",
    ]
    keywords = [kw for kw in preferred if kw.lower() in text.lower()]
    if len(keywords) >= limit:
        return keywords[:limit]

    words = re.findall(r"\b[A-Za-z][A-Za-z\-]{4,}\b", text.lower())
    stop = {"which", "there", "their", "these", "those", "with", "from", "that", "this", "have", "will", "test"}
    counts = {}
    for word in words:
        if word in stop:
            continue
        counts[word] = counts.get(word, 0) + 1
    for word, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        if word not in [kw.lower() for kw in keywords]:
            keywords.append(word)
        if len(keywords) >= limit:
            break
    return keywords[:limit]


def render_assistant_message(content):
    """Render Agent output as structured markdown."""
    sections = [part.strip() for part in content.split("\n\n---\n\n") if part.strip()]
    for index, section in enumerate(sections):
        st.markdown(section)
        if index < len(sections) - 1:
            st.divider()


def inject_style():
    """Add product-level styling."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 100% !important;
            padding-top: 1.4rem;
            padding-left: 2rem;
            padding-right: 2rem;
            padding-bottom: 7.5rem;
        }
        main {
            max-width: 100% !important;
        }
        div[data-testid="stDecoration"],
        div[data-testid="stToolbar"],
        div[data-testid="stStatusWidget"],
        div[data-testid="stAppDeployButton"],
        .stAppDeployButton,
        #MainMenu,
        footer,
        [data-testid="manage-app-button"],
        [aria-label="Manage app"],
        [title="Manage app"],
        iframe[src*="streamlit"],
        iframe[title="streamlit_app_badge"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            overflow: hidden !important;
        }
        header[data-testid="stHeader"] {
            background: transparent !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] {
            background-color: #f5f6f8 !important;
            height: 100vh !important;
        }
        [data-testid="stSidebar"] > div {
            height: 100vh !important;
            overflow-y: auto !important;
        }
        section[data-testid="stSidebar"] {
            min-width: 310px;
            overflow: hidden !important;
        }
        [data-testid="stSidebar"]::-webkit-scrollbar,
        [data-testid="stSidebar"] > div::-webkit-scrollbar {
            width: 6px;
        }
        [data-testid="stSidebar"]::-webkit-scrollbar-thumb,
        [data-testid="stSidebar"] > div::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 10px;
        }
        [data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover,
        [data-testid="stSidebar"] > div::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        .left-panel-title {
            color: #111111;
            font-weight: 750;
            line-height: 1.12;
            margin-bottom: 8px;
        }
        .left-panel-subtitle {
            color: #666666;
            font-size: 0.88rem;
            line-height: 1.5;
            margin-bottom: 8px;
        }
        .left-panel-meta {
            color: #999999;
            font-size: 0.78rem;
        }
        div[data-testid="column"]:has(.left-panel-root) > div[data-testid="stVerticalBlock"] {
            max-height: calc(100vh - 2rem);
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding-right: 8px;
        }
        div[data-testid="column"]:has(.left-panel-root) > div[data-testid="stVerticalBlock"]::-webkit-scrollbar {
            width: 6px;
        }
        div[data-testid="column"]:has(.left-panel-root) > div[data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 10px;
        }
        div[data-testid="column"]:has(.left-panel-root) > div[data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border-color: #eaeaea !important;
            border-radius: 10px !important;
        }
        .workspace-header {
            padding: 10px 0 14px 0;
            border-bottom: 1px solid rgba(49, 51, 63, 0.14);
            margin-bottom: 14px;
        }
        .metric-card {
            border: 1px solid rgba(49, 51, 63, 0.14);
            border-radius: 8px;
            padding: 12px;
            background: rgba(250, 250, 250, 0.82);
        }
        .panel-title {
            font-weight: 700;
            margin-bottom: 6px;
        }
        .chunk-card {
            border-left: 3px solid #3b82f6;
            padding: 10px 12px;
            margin-bottom: 10px;
            background: rgba(59, 130, 246, 0.06);
            border-radius: 6px;
            font-size: 0.9rem;
            line-height: 1.48;
        }
        .paper-chip {
            border: 1px solid rgba(49, 51, 63, 0.16);
            border-radius: 8px;
            padding: 8px 10px;
            margin-bottom: 8px;
            background: rgba(255, 255, 255, 0.9);
        }
        div[data-testid="stChatInput"] {
            position: fixed;
            left: 50%;
            bottom: 18px;
            transform: translateX(-50%);
            width: min(760px, calc(100vw - 420px));
            z-index: 9999;
            padding: 8px 10px;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.96);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
            backdrop-filter: blur(10px);
        }
        @media (max-width: 900px) {
            div[data-testid="stChatInput"] {
                left: 50%;
                width: calc(100vw - 32px);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render left document management sidebar."""
    with st.sidebar:
        st.markdown("## AI Research Assistant Pro")
        st.caption("NotebookLM 式论文库 + ChatGPT 式对话 + 科研分析面板")
        st.caption(f"Version: {APP_VERSION}")

        st.divider()
        st.markdown("### 📄 我的论文库")
        papers = list(st.session_state["papers"].keys())
        if papers:
            selected = st.selectbox(
                "选择当前论文",
                options=papers,
                index=papers.index(st.session_state["current_paper"]) if st.session_state["current_paper"] in papers else 0,
            )
            if selected != st.session_state.get("current_paper"):
                select_paper(selected)
                st.rerun()
        else:
            st.info("还没有上传论文。")

        for name in papers:
            record = st.session_state["papers"][name]
            marker = "🧠 当前" if name == st.session_state.get("current_paper") else "📄 论文"
            st.markdown(
                f"""
                <div class="paper-chip">
                    <strong>{marker}</strong><br/>
                    {name}<br/>
                    <small>{record.get('chunk_count', 0)} chunks · {record.get('word_count', 0)} tokens</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("### ➕ 上传新论文")
        uploaded_files = st.file_uploader("上传 PDF", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name in st.session_state["papers"]:
                    continue
                with st.spinner(f"正在解析 {uploaded_file.name}..."):
                    file_path = save_uploaded_pdf(uploaded_file)
                    record = build_paper_record(file_path, uploaded_file.name)
                    st.session_state["papers"][uploaded_file.name] = record
                    st.session_state["current_paper"] = uploaded_file.name
                    sync_current_status(record)
            st.success("论文已加入论文库。")

        if st.button("加载示例论文", use_container_width=True):
            load_demo_pdf()

        if st.button("清空对话", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["last_retrieval"] = []
            st.rerun()

        st.divider()
        st.markdown("### 📊 论文基本信息")
        record = current_paper_record()
        if record:
            meta = record.get("pdf_data", {}).get("metadata", {})
            st.markdown(f"**当前论文：** {record.get('name')}")
            st.caption(f"标题：{meta.get('title') or '未识别'}")
            st.caption(f"作者：{meta.get('authors') or '未识别'}")
            st.caption(f"字数/Token：{record.get('word_count', 0)}")
            st.caption(f"图表：{record.get('figure_count', 0) + record.get('table_count', 0)}")
        else:
            st.warning("请选择或上传论文。")


def render_left_panel():
    """Render the custom left document panel."""
    st.markdown('<div class="left-panel-root"></div>', unsafe_allow_html=True)
    title_col, toggle_col = st.columns([0.86, 0.14], vertical_alignment="center")
    with title_col:
        st.markdown('<h2 class="left-panel-title">AI Research Assistant Pro</h2>', unsafe_allow_html=True)
    with toggle_col:
        if st.button("«", key="toggle_left_panel_close_v2", help="收起论文库"):
            st.session_state.show_left_panel = False
            st.rerun()

    st.markdown(
        f"""
        <div class="left-panel-subtitle">NotebookLM 式论文库 + ChatGPT 式对话 + 科研分析面板</div>
        <div class="left-panel-meta">Version: {APP_VERSION}</div>
        """,
        unsafe_allow_html=True,
    )

    papers = list(st.session_state["papers"].keys())

    with st.container(height=720, border=False):
        with st.container(border=True):
            st.markdown("### 📄 我的论文库")
            if papers:
                selected = st.selectbox(
                    "选择当前论文",
                    options=papers,
                    index=papers.index(st.session_state["current_paper"]) if st.session_state["current_paper"] in papers else 0,
                    key="left_current_paper_v2",
                )
                if selected != st.session_state.get("current_paper"):
                    select_paper(selected)
                    st.rerun()
            else:
                st.info("还没有上传论文。")

            for name in papers:
                record = st.session_state["papers"][name]
                marker = "当前" if name == st.session_state.get("current_paper") else "论文"
                st.markdown(
                    f"""
                    <div class="paper-chip">
                        <strong>{marker}</strong><br/>
                        {name}<br/>
                        <small>{record.get('chunk_count', 0)} chunks · {record.get('word_count', 0)} tokens</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.write("")
        with st.container(border=True):
            st.markdown("### 上传新论文")
            uploaded_files = st.file_uploader("上传 PDF", type=["pdf"], accept_multiple_files=True, key="left_pdf_upload_v2")
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if uploaded_file.name in st.session_state["papers"]:
                        continue
                    with st.spinner(f"正在解析 {uploaded_file.name}..."):
                        file_path = save_uploaded_pdf(uploaded_file)
                        record = build_paper_record(file_path, uploaded_file.name)
                        st.session_state["papers"][uploaded_file.name] = record
                        st.session_state["current_paper"] = uploaded_file.name
                        sync_current_status(record)
                st.success("论文已加入论文库。")

            if st.button("加载示例论文", use_container_width=True, key="left_load_demo_v2"):
                load_demo_pdf()

            if st.button("清空对话", use_container_width=True, key="left_clear_chat_v2"):
                st.session_state["messages"] = []
                st.session_state["last_retrieval"] = []
                st.rerun()

        st.write("")
        with st.container(border=True):
            st.markdown("### 论文基本信息")
            record = current_paper_record()
            if record:
                meta = record.get("pdf_data", {}).get("metadata", {})
                st.markdown(f"**当前论文：** {record.get('name')}")
                st.caption(f"标题：{meta.get('title') or '未识别'}")
                st.caption(f"作者：{meta.get('authors') or '未识别'}")
                st.caption(f"字数/Token：{record.get('word_count', 0)}")
                st.caption(f"图表：{record.get('figure_count', 0) + record.get('table_count', 0)}")
            else:
                st.warning("请选择或上传论文。")
    return

    st.markdown('<div class="left-panel-root"></div>', unsafe_allow_html=True)
    title_col, toggle_col = st.columns([0.86, 0.14], vertical_alignment="center")
    with title_col:
        st.markdown('<h2 class="left-panel-title">AI Research Assistant Pro</h2>', unsafe_allow_html=True)
    with toggle_col:
        if st.button("«", key="toggle_left_panel_close", help="收起论文库"):
            st.session_state.show_left_panel = False
            st.rerun()

    st.markdown(
        f"""
        <div class="left-panel-subtitle">NotebookLM 式论文库 + ChatGPT 式对话 + 科研分析面板</div>
        <div class="left-panel-meta">Version: {APP_VERSION}</div>
        """,
        unsafe_allow_html=True,
    )

    papers = list(st.session_state["papers"].keys())

    st.write("")
    with st.container(border=True):
        st.markdown("### 📄 我的论文库")
        if papers:
            selected = st.selectbox(
                "选择当前论文",
                options=papers,
                index=papers.index(st.session_state["current_paper"]) if st.session_state["current_paper"] in papers else 0,
                key="left_current_paper",
            )
            if selected != st.session_state.get("current_paper"):
                select_paper(selected)
                st.rerun()
        else:
            st.info("还没有上传论文。")

        for name in papers:
            record = st.session_state["papers"][name]
            marker = "当前" if name == st.session_state.get("current_paper") else "论文"
            st.markdown(
                f"""
                <div class="paper-chip">
                    <strong>{marker}</strong><br/>
                    {name}<br/>
                    <small>{record.get('chunk_count', 0)} chunks · {record.get('word_count', 0)} tokens</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    with st.container(border=True):
        st.markdown("### 上传新论文")
        uploaded_files = st.file_uploader("上传 PDF", type=["pdf"], accept_multiple_files=True, key="left_pdf_upload")
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name in st.session_state["papers"]:
                    continue
                with st.spinner(f"正在解析 {uploaded_file.name}..."):
                    file_path = save_uploaded_pdf(uploaded_file)
                    record = build_paper_record(file_path, uploaded_file.name)
                    st.session_state["papers"][uploaded_file.name] = record
                    st.session_state["current_paper"] = uploaded_file.name
                    sync_current_status(record)
            st.success("论文已加入论文库。")

        if st.button("加载示例论文", use_container_width=True, key="left_load_demo"):
            load_demo_pdf()

        if st.button("清空对话", use_container_width=True, key="left_clear_chat"):
            st.session_state["messages"] = []
            st.session_state["last_retrieval"] = []
            st.rerun()

    st.write("")
    with st.container(border=True):
        st.markdown("### 论文基本信息")
        record = current_paper_record()
        if record:
            meta = record.get("pdf_data", {}).get("metadata", {})
            st.markdown(f"**当前论文：** {record.get('name')}")
            st.caption(f"标题：{meta.get('title') or '未识别'}")
            st.caption(f"作者：{meta.get('authors') or '未识别'}")
            st.caption(f"字数/Token：{record.get('word_count', 0)}")
            st.caption(f"图表：{record.get('figure_count', 0) + record.get('table_count', 0)}")
        else:
            st.warning("请选择或上传论文。")


def render_left_panel_toggle():
    """Render the compact left panel toggle."""
    if st.button("»", key="toggle_left_panel_open", help="展开论文库"):
        st.session_state.show_left_panel = True
        st.rerun()


def render_info_panel():
    """Render right-side paper analysis panel."""
    record = current_paper_record()
    st.markdown("### 论文分析")

    if not record:
        st.info("上传论文后，这里会显示作者、图表、公式和 RAG 摘要。")
        return

    pdf_data = record.get("pdf_data", {})
    meta = pdf_data.get("metadata", {})
    counts = paper_counts(pdf_data)

    st.markdown(f"**作者**  \n{meta.get('authors') or '未识别'}")
    metric_col_1, metric_col_2 = st.columns(2)
    with metric_col_1:
        st.metric("图表", counts["figures"] + counts["tables"])
    with metric_col_2:
        st.metric("公式", counts["formulas"])

    st.divider()
    st.markdown("### RAG 摘要")
    chunks = st.session_state.get("last_retrieval", [])
    if not chunks:
        st.caption("提问后展示 Top-3 检索摘要。")
        return

    for index, item in enumerate(chunks[:3], start=1):
        score = item.get("score", 0.0)
        text = item.get("text", "").strip()
        preview = text[:260] + ("..." if len(text) > 260 else "")
        st.markdown(
            f"""
            <div class="chunk-card">
                <strong>Top {index}</strong> · {score:.3f}<br/>
                {preview}
            </div>
            """,
            unsafe_allow_html=True,
        )
    return

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 论文分析面板")

    if not record:
        st.info("上传论文后，这里会显示元信息、关键词和 RAG 检索结果。")
        return

    pdf_data = record.get("pdf_data", {})
    meta = pdf_data.get("metadata", {})
    counts = paper_counts(pdf_data)
    keywords = extract_keywords(pdf_data)

    st.markdown(f"**📄 论文标题**  \n{meta.get('title') or record.get('name')}")
    st.markdown(f"**👤 作者**  \n{meta.get('authors') or '未识别'}")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("图表", counts["figures"] + counts["tables"])
        st.metric("文本片段", record.get("chunk_count", 0))
    with c2:
        st.metric("公式", counts["formulas"])
        st.metric("词/Token", counts["words"])

    st.markdown("**📚 关键词**")
    if keywords:
        st.markdown(" · ".join(f"`{kw}`" for kw in keywords))
    else:
        st.caption("暂无关键词。")

    st.divider()
    st.markdown("### 当前RAG检索结果")
    chunks = st.session_state.get("last_retrieval", [])
    if not chunks:
        st.caption("提出问题后，这里会展示 Top-3 检索片段和相似度。")
        return

    for index, item in enumerate(chunks, start=1):
        score = item.get("score", 0.0)
        text = item.get("text", "").strip()
        preview = text[:560] + ("..." if len(text) > 560 else "")
        st.markdown(
            f"""
            <div class="chunk-card">
                <strong>Chunk {index}</strong> · similarity {score:.3f}<br/>
                <small>来源：当前论文 / FAISS Top-{index}</small><br/><br/>
                {preview}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_right_panel_toggle():
    """Render the single custom control for the right analysis panel."""
    label = "»" if st.session_state.show_right_panel else "«"
    help_text = "收起论文分析面板" if st.session_state.show_right_panel else "展开论文分析面板"
    if st.button(label, key="toggle_right_panel", help=help_text):
        st.session_state.show_right_panel = not st.session_state.show_right_panel
        st.rerun()


def render_chat_area():
    """Render central ChatGPT-style conversation area."""
    st.markdown(
        """
        <div class="workspace-header">
            <h1 style="margin:0;">AI Research Assistant Pro</h1>
            <p style="margin:4px 0 0 0;color:rgba(49,51,63,.72);">
            面向论文阅读、公式图表解释、RAG问答与答辩准备的科研AI工作台
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    record = current_paper_record()
    if record:
        st.success(f"当前论文：{record.get('name')}")
    else:
        st.warning("请先在左侧上传或加载一篇论文。")

    if not st.session_state["messages"]:
        with st.chat_message("assistant"):
            st.markdown(
                """
                你好，我可以围绕当前论文进行连续追问。你可以试试：

                - 这篇论文有什么创新点？
                - 总结这篇论文，并列出核心公式和图表
                - 图5和图7分别说明什么？
                - D_th 公式是什么意思？
                - 帮我生成答辩讲解思路
                """
            )

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message["content"])
            else:
                paper = message.get("paper")
                if paper:
                    st.caption(f"基于论文：{paper}")
                st.markdown(message["content"])

    prompt = st.chat_input("请输入你的问题，例如：总结这篇论文，并解释其中的公式和图表")
    if prompt:
        run_user_query(prompt)
        st.rerun()


def main():
    """Render the product UI."""
    init_state()
    inject_style()

    left_open = st.session_state.show_left_panel
    right_open = st.session_state.show_right_panel

    if left_open and right_open:
        left_col, main_col, right_col = st.columns([1.05, 2.6, 1.0], gap="small")
        with left_col:
            render_left_panel()
        with main_col:
            render_chat_area()
        with right_col:
            render_right_panel_toggle()
            render_info_panel()
    elif left_open and not right_open:
        left_col, main_col, right_toggle_col = st.columns([1.05, 3.0, 0.08], gap="small")
        with left_col:
            render_left_panel()
        with main_col:
            render_chat_area()
        with right_toggle_col:
            render_right_panel_toggle()
    elif not left_open and right_open:
        left_toggle_col, main_col, right_col = st.columns([0.08, 2.9, 1.0], gap="small")
        with left_toggle_col:
            render_left_panel_toggle()
        with main_col:
            render_chat_area()
        with right_col:
            render_right_panel_toggle()
            render_info_panel()
    else:
        left_toggle_col, main_col, right_toggle_col = st.columns([0.08, 3.6, 0.08], gap="small")
        with left_toggle_col:
            render_left_panel_toggle()
        with main_col:
            render_chat_area()
        with right_toggle_col:
            render_right_panel_toggle()


if __name__ == "__main__":
    main()
