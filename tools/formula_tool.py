"""Formula extraction and explanation tool."""

import re


def _as_pdf_data(pdf_data):
    """Normalize old text input and new unified PDF data input."""
    if isinstance(pdf_data, dict):
        return pdf_data
    return {"text": str(pdf_data), "formulas": []}


def _looks_garbled(text):
    """Detect obvious mojibake or damaged formula text."""
    bad_markers = [
        "\ufffd", "\u25a1", "\u25c6", "\ufe49", "\ufe5f",
        "\u7c73", "\u8003", "\u6c50", "\u6c55", "\u6c61", "\u7af9", "\u725f"
    ]
    if any(marker in text for marker in bad_markers):
        return True
    weird_chars = re.findall(r"[^\w\s.,;:!?%()\-/=<>+\[\]*]", text)
    if len(text) > 0 and len(weird_chars) / len(text) > 0.12:
        return True
    return False


def analyze_formulas(pdf_data):
    """Return formula list and lightweight explanations."""
    data = _as_pdf_data(pdf_data)
    formulas = []

    for item in data.get("formulas", []):
        if not item:
            continue
        cleaned = re.sub(r"\s+", " ", item).strip()
        if not cleaned or _looks_garbled(cleaned):
            continue
        formulas.append(cleaned)

    if not formulas:
        return "\U0001F9EE \u516c\u5f0f\u89e3\u6790\n\u6587\u6863\u4e2d\u672a\u8bc6\u522b\u5230\u53ef\u76f4\u63a5\u89e3\u6790\u7684\u6e05\u6670\u516c\u5f0f\uff0c\u6216 PDF \u6587\u672c\u5c42\u5b58\u5728\u7f16\u7801\u95ee\u9898\u3002"

    lines = ["\U0001F9EE \u516c\u5f0f\u89e3\u6790"]
    for index, formula in enumerate(formulas[:10], start=1):
        short_formula = re.sub(r"\s+", " ", formula)
        lines.append(f"- \u516c\u5f0f {index}\uff1a{short_formula}")
        lines.append("  \u89e3\u91ca\uff1a\u8fd9\u662f\u8bba\u6587\u4e2d\u7684\u6570\u5b66\u5173\u7cfb\u6216\u7ea6\u675f\u6761\u4ef6\uff0c\u5177\u4f53\u542b\u4e49\u9700\u8981\u7ed3\u5408\u6240\u5728\u6bb5\u843d\u4e00\u8d77\u7406\u89e3\u3002")

    return "\n".join(lines)
