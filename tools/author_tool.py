"""Author and metadata extraction tool."""

import re


def _as_pdf_data(pdf_data):
    """Normalize old text input and new unified PDF data input."""
    if isinstance(pdf_data, dict):
        return pdf_data
    return {"text": str(pdf_data), "metadata": {"title": "", "authors": ""}}


def extract_authors(pdf_data):
    """Extract authors from metadata, with a text-based fallback."""
    data = _as_pdf_data(pdf_data)
    metadata = data.get("metadata", {})
    authors = metadata.get("authors", "").strip()
    title = metadata.get("title", "").strip()

    if authors:
        lines = ["\U0001F464 \u4f5c\u8005\u4e0e\u5143\u4fe1\u606f"]
        if title:
            lines.append(f"- \u6807\u9898\uff1a{title}")
        lines.append(f"- \u4f5c\u8005\uff1a{authors}")
        return "\n".join(lines)

    text = data.get("text", "")
    candidates = []
    for line in [line.strip() for line in text.splitlines() if line.strip()][:80]:
        lower = line.lower()
        if re.search(r"\b(author|authors)\b", line, re.IGNORECASE):
            candidates.append(line)
        elif (
            re.match(r"^([A-Z][A-Za-z.\-]+\s+){2,9}[A-Z][A-Za-z.\-]+$", line)
            and not re.search(
                r"@|\d|prof\.|university|laborator|department|institute|street|road|"
                r"design|testing|test|system|chip|chips|optimal|infrastructure|conventional|resources",
                lower,
            )
        ):
            candidates.append(line)
        elif "," in line and len(line.split()) <= 18 and not lower.startswith(("abstract", "keywords")):
            candidates.append(line)

    if not candidates:
        return "\U0001F464 \u4f5c\u8005\u4e0e\u5143\u4fe1\u606f\n\u6587\u6863\u4e2d\u672a\u8bc6\u522b\u5230\u660e\u786e\u4f5c\u8005\u4fe1\u606f\u3002"

    return "\U0001F464 \u4f5c\u8005\u4e0e\u5143\u4fe1\u606f\n" + "\n".join(f"- {item}" for item in candidates[:5])
