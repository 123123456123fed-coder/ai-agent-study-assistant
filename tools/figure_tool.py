"""Figure and table analysis tool."""


def _as_pdf_data(pdf_data):
    """Normalize old text input and new unified PDF data input."""
    if isinstance(pdf_data, dict):
        return pdf_data
    return {"text": str(pdf_data), "figures": [], "tables": []}


def analyze_figures(pdf_data):
    """Return figure/table list and caption summary."""
    data = _as_pdf_data(pdf_data)
    figures = data.get("figures", [])
    tables = data.get("tables", [])

    if not figures and not tables:
        return "📊 图表解析\n文档中未识别到明显的 Figure/Table 标题。"

    lines = ["📊 图表解析"]
    if figures:
        lines.append("Figures:")
        for item in figures[:8]:
            lines.append(f"- {item.get('id', 'Figure')}：{item.get('caption', '')}")
    if tables:
        lines.append("Tables:")
        for item in tables[:8]:
            lines.append(f"- {item.get('id', 'Table')}：{item.get('content', '')}")

    return "\n".join(lines)
