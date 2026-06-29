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
        return "\U0001F5BC \u56fe\u8868\u89e3\u6790\n\u6587\u6863\u4e2d\u672a\u8bc6\u522b\u5230\u660e\u786e\u7684 Figure/Table \u6807\u9898\u3002"

    lines = ["\U0001F5BC \u56fe\u8868\u89e3\u6790"]
    if figures:
        lines.append("Figures:")
        for item in figures[:8]:
            lines.append(f"- {item.get('id', 'Figure')}\uff1a{item.get('caption', '')}")
    if tables:
        lines.append("Tables:")
        for item in tables[:8]:
            lines.append(f"- {item.get('id', 'Table')}\uff1a{item.get('content', '')}")

    return "\n".join(lines)
