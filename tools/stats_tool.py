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
        "\U0001F4CA \u5b57\u6570\u4e0e\u7ed3\u6784\u7edf\u8ba1\n"
        f"- \u603b token/\u8bcd\u6570\uff1a{len(words)}\n"
        f"- \u4e2d\u6587\u5b57\u7b26\u6570\uff1a{len(chinese_chars)}\n"
        f"- \u82f1\u6587\u8bcd\u6570\uff1a{len(english_words)}\n"
        f"- \u539f\u59cb\u5b57\u7b26\u6570\uff1a{len(text)}\n"
        f"- \u56fe\uff1a{figures} \u4e2a\n"
        f"- \u8868\uff1a{tables} \u4e2a\n"
        f"- \u516c\u5f0f\u5019\u9009\uff1a{formulas} \u6761"
    )
