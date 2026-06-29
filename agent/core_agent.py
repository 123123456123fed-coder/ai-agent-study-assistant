"""Core Agent scheduler for the Multimodal Scientific AI Assistant."""

from agent.classifier import classify_query
from llm.qwen import ask_llm
from rag import retriever
from tools.author_tool import extract_authors
from tools.figure_tool import analyze_figures
from tools.formula_tool import analyze_formulas
from tools.stats_tool import count_words
from utils.pdf_loader import load_pdf


NO_CONTEXT = "\u672a\u5728\u6587\u6863\u4e2d\u627e\u5230\u76f8\u5173\u5185\u5bb9\u3002"

ZH_RAG_HINTS = ["\u8bba\u6587", "\u6587\u6863", "\u6587\u7ae0", "\u8fd9\u7bc7", "\u603b\u7ed3", "\u8d21\u732e", "\u65b9\u6cd5", "\u5b9e\u9a8c", "\u7ed3\u8bba", "\u6838\u5fc3"]
ZH_SUMMARY_HINTS = ["\u603b\u7ed3", "\u6982\u62ec", "\u6458\u8981", "\u5b66\u4e60\u8ba1\u5212"]
ZH_AUTHOR_HINTS = ["\u4f5c\u8005", "\u8c01\u5199\u7684", "\u5143\u4fe1\u606f", "\u6807\u9898"]
ZH_WORD_COUNT_HINTS = ["\u5b57\u6570", "\u591a\u5c11\u5b57", "\u8bcd\u6570", "\u7edf\u8ba1", "\u7ed3\u6784"]
ZH_FIGURE_HINTS = ["\u56fe", "\u56fe\u8868", "\u8868\u683c"]
ZH_FORMULA_HINTS = ["\u516c\u5f0f", "\u65b9\u7a0b", "\u8868\u8fbe\u5f0f"]


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

    if _wants_any(query, ZH_RAG_HINTS + ["paper", "summary"]):
        tasks.add("rag")
    if _wants_any(query, ZH_SUMMARY_HINTS + ["summary", "summarize"]):
        tasks.add("summary")
    if _wants_any(query, ZH_AUTHOR_HINTS + ["author", "authors", "metadata"]):
        tasks.add("author")
    if _wants_any(query, ZH_WORD_COUNT_HINTS + ["word count", "statistics"]):
        tasks.add("word_count")
    if _wants_any(query, ZH_FIGURE_HINTS + ["figure", "fig.", "table"]):
        tasks.add("figure")
    if _wants_any(query, ZH_FORMULA_HINTS + ["formula", "equation"]):
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
        lines.append(f"\u7247\u6bb5 {index}\uff08\u76f8\u4f3c\u5ea6\uff1a{score:.3f}\uff09\uff1a\n{text}")
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
            results["tool_status"] = "\u8bf7\u5148\u4e0a\u4f20\u8bba\u6587 PDF \u5e76\u6784\u5efa\u77e5\u8bc6\u5e93\u3002"
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
\u4f60\u662f\u79d1\u7814\u8bba\u6587\u5206\u6790 AI \u52a9\u624b\uff0c\u8bf7\u57fa\u4e8e\u4ee5\u4e0b\u4fe1\u606f\u56de\u7b54\u95ee\u9898\uff1a

\u3010\u4efb\u52a1\u7c7b\u578b\u3011
{", ".join(tasks)}

\u3010\u8bba\u6587\u6587\u672c\u3011
{results.get("rag", "")}

\u3010\u56fe\u8868\u4fe1\u606f\u3011
{results.get("figure", "")}

\u3010\u516c\u5f0f\u4fe1\u606f\u3011
{results.get("formula", "")}

\u3010\u5143\u4fe1\u606f\u3011
{results.get("author", "")}

\u3010\u5b57\u6570/\u7ed3\u6784\u7edf\u8ba1\u3011
{results.get("word_count", "")}

\u3010\u5de5\u5177\u72b6\u6001\u3011
{results.get("tool_status", "\u6b63\u5e38")}

\u3010\u7528\u6237\u95ee\u9898\u3011
{query}

\u8981\u6c42\uff1a
- \u4e0d\u5141\u8bb8\u7f16\u9020
- \u5fc5\u987b\u57fa\u4e8e\u63d0\u4f9b\u5185\u5bb9
- \u5de5\u5177\u7ed3\u679c\u4f18\u5148\u4e8e\u6a21\u578b\u63a8\u6f14
- \u5982\u679c\u5185\u5bb9\u4e2d\u6ca1\u6709\u7b54\u6848\uff0c\u56de\u7b54\u201c\u6587\u6863\u4e2d\u672a\u63d0\u53ca\u201d
- \u4e0d\u8981\u91cd\u590d\u8f93\u51fa\u76f8\u540c\u4fe1\u606f
- \u8f93\u51fa\u7ed3\u6784\u5316\u56de\u7b54\uff0c\u5305\u542b\u201c\u4efb\u52a1\u5224\u65ad\u3001\u4f9d\u636e\u3001\u6700\u7ec8\u56de\u7b54\u201d
"""


def _fallback_answer(tasks, results):
    """Return deterministic structured output when the LLM is unavailable."""
    sections = ["\U0001F4CC \u4efb\u52a1\u5224\u65ad\n- " + "\u3001".join(tasks)]
    evidence = []

    for key in ["author", "word_count", "figure", "formula"]:
        if results.get(key):
            evidence.append(results[key])
    if results.get("rag"):
        evidence.append("\U0001F4DA RAG \u68c0\u7d22\u4f9d\u636e\n" + results["rag"])
    if results.get("tool_status"):
        evidence.append(results["tool_status"])

    sections.append("\U0001F4CE \u4f9d\u636e\n" + ("\n\n".join(evidence) if evidence else "\u6682\u65e0\u53ef\u7528\u8bba\u6587\u4fe1\u606f\u3002"))
    sections.append("\u2705 \u6700\u7ec8\u56de\u7b54\n\u5df2\u6839\u636e\u5f53\u524d\u53ef\u7528\u7684 RAG \u4e0e\u5de5\u5177\u7ed3\u679c\u5b8c\u6210\u6574\u5408\u3002")
    return "\n\n---\n\n".join(sections)


def run_agent(query, pdf_data=None):
    """Run multimodal task decomposition, tool selection, and answer fusion."""
    tasks = _detect_tasks(query)
    results = _collect_results(query, tasks, pdf_data)

    if tasks == ["general"]:
        final_answer = ask_llm(query)
    elif "formula" in tasks and results.get("formula"):
        final_answer = results["formula"]
    else:
        final_answer = ask_llm(_build_final_prompt(query, tasks, results))
        if final_answer.startswith("\u8c03\u7528\u5931\u8d25") or "DASHSCOPE_API_KEY" in final_answer:
            final_answer = _fallback_answer(tasks, results)

    return (
        "\U0001F9ED Agent\u4efb\u52a1\u62c6\u89e3\n"
        f"- \u8bc6\u522b\u4efb\u52a1\uff1a{', '.join(tasks)}\n"
        "- \u6267\u884c\u8def\u5f84\uff1aRAG \u6587\u672c\u68c0\u7d22 + \u591a\u6a21\u6001\u5de5\u5177 + LLM \u878d\u5408\u8f93\u51fa\n\n"
        "---\n\n"
        "\U0001F9E0 \u591a\u5de5\u5177\u878d\u5408\u56de\u7b54\n"
        f"{final_answer}"
    )
