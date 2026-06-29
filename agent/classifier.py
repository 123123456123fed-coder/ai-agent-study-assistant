"""Rule-based task classifier for the multimodal scientific Agent."""


def classify_query(query):
    """Classify the user query into one primary task type."""
    normalized = query.lower()

    if any(keyword in query for keyword in ["作者", "谁写", "元信息", "标题"]):
        return "author"
    if any(keyword in normalized for keyword in ["author", "authors", "metadata", "title"]):
        return "author"

    if any(keyword in query for keyword in ["字数", "多少字", "词数", "统计", "结构"]):
        return "word_count"
    if any(keyword in normalized for keyword in ["word count", "statistics", "structure"]):
        return "word_count"

    if any(keyword in query for keyword in ["图", "图表", "表格"]):
        return "figure"
    if any(keyword in normalized for keyword in ["figure", "fig.", "table", "chart"]):
        return "figure"

    if any(keyword in query for keyword in ["公式", "方程", "表达式"]):
        return "formula"
    if any(keyword in normalized for keyword in ["formula", "equation", "loss", "constraint"]):
        return "formula"

    if any(keyword in query for keyword in ["总结", "概括", "摘要", "学习计划"]):
        return "summary"
    if any(keyword in normalized for keyword in ["summary", "summarize", "abstract"]):
        return "summary"

    if any(keyword in query for keyword in ["论文", "文档", "文章", "这篇", "贡献", "方法", "实验", "结论", "核心"]):
        return "rag"
    if any(keyword in normalized for keyword in ["paper", "contribution", "method", "experiment", "conclusion"]):
        return "rag"

    return "general"
