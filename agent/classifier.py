"""Rule-based task classifier for the multimodal scientific Agent."""

AUTHOR_KEYWORDS_ZH = ["\u4f5c\u8005", "\u8c01\u5199\u7684", "\u5143\u4fe1\u606f", "\u6807\u9898"]
WORD_COUNT_KEYWORDS_ZH = ["\u5b57\u6570", "\u591a\u5c11\u5b57", "\u8bcd\u6570", "\u7edf\u8ba1", "\u7ed3\u6784"]
FIGURE_KEYWORDS_ZH = ["\u56fe", "\u56fe\u8868", "\u8868\u683c"]
FORMULA_KEYWORDS_ZH = ["\u516c\u5f0f", "\u65b9\u7a0b", "\u8868\u8fbe\u5f0f"]
SUMMARY_KEYWORDS_ZH = ["\u603b\u7ed3", "\u6982\u62ec", "\u6458\u8981", "\u5b66\u4e60\u8ba1\u5212"]
RAG_KEYWORDS_ZH = ["\u8bba\u6587", "\u6587\u6863", "\u6587\u7ae0", "\u8fd9\u7bc7", "\u8d21\u732e", "\u65b9\u6cd5", "\u5b9e\u9a8c", "\u7ed3\u8bba", "\u6838\u5fc3"]


def classify_query(query):
    """Classify the user query into one primary task type."""
    normalized = query.lower()

    if any(keyword in query for keyword in AUTHOR_KEYWORDS_ZH):
        return "author"
    if any(keyword in normalized for keyword in ["author", "authors", "metadata", "title"]):
        return "author"

    if any(keyword in query for keyword in WORD_COUNT_KEYWORDS_ZH):
        return "word_count"
    if any(keyword in normalized for keyword in ["word count", "statistics", "structure"]):
        return "word_count"

    if any(keyword in query for keyword in FIGURE_KEYWORDS_ZH):
        return "figure"
    if any(keyword in normalized for keyword in ["figure", "fig.", "table", "chart"]):
        return "figure"

    if any(keyword in query for keyword in FORMULA_KEYWORDS_ZH):
        return "formula"
    if any(keyword in normalized for keyword in ["formula", "equation", "loss", "constraint"]):
        return "formula"

    if any(keyword in query for keyword in SUMMARY_KEYWORDS_ZH):
        return "summary"
    if any(keyword in normalized for keyword in ["summary", "summarize", "abstract"]):
        return "summary"

    if any(keyword in query for keyword in RAG_KEYWORDS_ZH):
        return "rag"
    if any(keyword in normalized for keyword in ["paper", "contribution", "method", "experiment", "conclusion"]):
        return "rag"

    return "general"
