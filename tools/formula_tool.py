"""Formula extraction and explanation tool."""


def _as_pdf_data(pdf_data):
    """Normalize old text input and new unified PDF data input."""
    if isinstance(pdf_data, dict):
        return pdf_data
    return {"text": str(pdf_data), "formulas": []}


def analyze_formulas(pdf_data):
    """Return formula list and lightweight explanations."""
    data = _as_pdf_data(pdf_data)
    formulas = data.get("formulas", [])

    if not formulas:
        return "🧮 公式解析\n文档中未识别到明显公式。"

    lines = ["🧮 公式解析"]
    for index, formula in enumerate(formulas[:10], start=1):
        lines.append(f"- 公式 {index}：{formula}")
        lines.append("  解释：该式是论文中的数学关系或约束条件，具体含义需要结合所在段落理解。")

    return "\n".join(lines)
