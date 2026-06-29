"""Document statistics tool."""

import re


def _get_text(pdf_data):
    """Return text from old text input or new unified PDF data."""
    if isinstance(pdf_data, dict):
        return pdf_data.get("text", "")
    return str(pdf_data)


def count_words(pdf_data):
    """Return word, character, and structure statistics."""
    text = _get_text(pdf_data)
    words = re.findall(r"[A-Za-z0-9_+\-/.]+|[\u4e00-\u9fff]", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    english_words = re.findall(r"[A-Za-z0-9_+\-/.]+", text)

    figures = len(pdf_data.get("figures", [])) if isinstance(pdf_data, dict) else 0
    tables = len(pdf_data.get("tables", [])) if isinstance(pdf_data, dict) else 0
    formulas = len(pdf_data.get("formulas", [])) if isinstance(pdf_data, dict) else 0

    return (
        "📊 字数/结构统计\n"
        f"- 总 token/词数量：{len(words)}\n"
        f"- 中文字符数：{len(chinese_chars)}\n"
        f"- 英文词数量：{len(english_words)}\n"
        f"- 原始字符数：{len(text)}\n"
        f"- 图：{figures} 个\n"
        f"- 表：{tables} 个\n"
        f"- 公式候选：{formulas} 条"
    )
