"""Core Agent scheduler for the Multimodal Scientific AI Assistant."""

from agent.classifier import classify_query
from llm.qwen import ask_llm
from rag import retriever
from tools.author_tool import extract_authors
from tools.figure_tool import analyze_figures
from tools.formula_tool import analyze_formulas
from tools.stats_tool import count_words
from utils.pdf_loader import load_pdf


NO_CONTEXT = "未在文档中找到相关内容"


def ingest_pdf(file_path):
    """Load a PDF, build text RAG index, and return UI status plus pdf_data."""
    pdf_data = load_pdf(file_path)
    index_info = retriever.build_from_data(pdf_data)
    return {
        **index_info,
        "pdf_data": pdf_data,
        "figure_count": len(pdf_data.get("figures", [])),
        "table_count": len(pdf_data.get("tables", [])),
        "formula_count": len(pdf_data.get("formulas", [])),
    }


def _wants_any(query, keywords):
    """Return whether a query contains any keyword."""
    query_lower = query.lower()
    return any(keyword.lower() in query_lower for keyword in keywords)


def _detect_tasks(query):
    """Detect one or more tasks for tool-augmented fusion."""
    primary = classify_query(query)
    tasks = {primary}

    if _wants_any(query, ["论文", "文档", "文章", "这篇", "总结", "贡献", "方法", "实验", "结论", "核心", "paper", "summary"]):
        tasks.add("rag")
    if _wants_any(query, ["总结", "概括", "摘要", "学习计划", "summary", "summarize"]):
        tasks.add("summary")
    if _wants_any(query, ["作者", "谁写", "元信息", "标题", "author", "authors", "metadata"]):
        tasks.add("author")
    if _wants_any(query, ["字数", "多少字", "词数", "统计", "结构", "word count", "statistics"]):
        tasks.add("word_count")
    if _wants_any(query, ["图", "图表", "表格", "figure", "fig.", "table"]):
        tasks.add("figure")
    if _wants_any(query, ["公式", "方程", "表达式", "formula", "equation"]):
        tasks.add("formula")

    if "general" in tasks and len(tasks) > 1:
        tasks.remove("general")

    order = ["rag", "summary", "author", "word_count", "figure", "formula", "general"]
    return sorted(tasks, key=order.index)


def _format_rag_results(chunks):
    """Format top-k RAG chunks with similarity scores."""
    if not chunks:
        return NO_CONTEXT

    lines = []
    for index, item in enumerate(chunks, start=1):
        score = item.get("score", 0.0)
        text = item.get("text", "").strip()
        lines.append(f"片段 {index}（相似度：{score:.3f}）：\n{text}")
    return "\n\n".join(lines)


def _ensure_pdf_data(pdf_data):
    """Use provided pdf_data or fallback to the latest ingested document."""
    if pdf_data:
        return pdf_data
    return retriever.get_pdf_data()


def _collect_results(query, tasks, pdf_data):
    """Run RAG and tools selected by the Agent."""
    results = {}
    data = _ensure_pdf_data(pdf_data)

    if any(task in tasks for task in ["rag", "summary"]):
        results["rag"] = _format_rag_results(retriever.search(query, top_k=3))

    if not data:
        if any(task in tasks for task in ["author", "word_count", "figure", "formula"]):
            results["tool_status"] = "请先上传论文PDF并构建知识库。"
        return results

    if "author" in tasks:
        results["author"] = extract_authors(data)
    if "word_count" in tasks:
        results["word_count"] = count_words(data)
    if "figure" in tasks:
        results["figure"] = analyze_figures(data)
    if "formula" in tasks:
        results["formula"] = analyze_formulas(data)

    return results


def _build_final_prompt(query, tasks, results):
    """Build the unified Tool-Augmented LLM prompt."""
    return f"""
你是科研论文分析AI助手，请基于以下信息回答问题：

【任务类型】
{", ".join(tasks)}

【论文文本】
{results.get("rag", "")}

【图表信息】
{results.get("figure", "")}

【公式信息】
{results.get("formula", "")}

【元信息】
{results.get("author", "")}

【字数/结构统计】
{results.get("word_count", "")}

【工具状态】
{results.get("tool_status", "正常")}

【用户问题】
{query}

要求：
- 不允许编造
- 必须基于提供内容
- 工具结果优先于模型推测
- 如果内容中没有答案，回答“文档中未提及”
- 不要重复输出相同信息
- 输出结构化回答，包含“任务判断、依据、最终回答”
"""


def _fallback_answer(tasks, results):
    """Return deterministic structured output when the LLM is unavailable."""
    sections = ["📌 任务判断\n- " + "、".join(tasks)]
    evidence = []

    for key in ["author", "word_count", "figure", "formula"]:
        if results.get(key):
            evidence.append(results[key])
    if results.get("rag"):
        evidence.append("📚 RAG检索依据\n" + results["rag"])
    if results.get("tool_status"):
        evidence.append(results["tool_status"])

    sections.append("📎 依据\n" + ("\n\n".join(evidence) if evidence else "暂无可用论文信息。"))
    sections.append("✍️ 最终回答\n已根据当前可用的 RAG 与工具结果完成整合。")
    return "\n\n---\n\n".join(sections)


def run_agent(query, pdf_data=None):
    """Run multimodal task decomposition, tool selection, and answer fusion."""
    tasks = _detect_tasks(query)
    results = _collect_results(query, tasks, pdf_data)

    if tasks == ["general"]:
        final_answer = ask_llm(query)
    else:
        final_answer = ask_llm(_build_final_prompt(query, tasks, results))
        if final_answer.startswith("调用失败") or "填写真实的 DASHSCOPE_API_KEY" in final_answer:
            final_answer = _fallback_answer(tasks, results)

    return (
        "📌 Agent任务拆解\n"
        f"- 识别任务：{', '.join(tasks)}\n"
        "- 执行路径：RAG文本检索 + 多模态工具 + LLM融合输出\n\n"
        "---\n\n"
        "🧠 多工具融合回答\n"
        f"{final_answer}"
    )
