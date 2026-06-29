"""PDF loading utilities with a unified multimodal paper data structure."""

import re

from pypdf import PdfReader


def _extract_figures(text):
    """Extract figure captions from plain PDF text."""
    pattern = re.compile(r"(?im)\b(Fig\.?|Figure)\s*(\d+)[\s.:;-]+([^\n]{1,260})")
    figures = []
    seen = set()

    for match in pattern.finditer(text):
        figure_id = f"Figure {match.group(2)}"
        caption = re.sub(r"\s+", " ", match.group(3)).strip()
        key = (figure_id, caption)
        if caption and key not in seen:
            figures.append({"id": figure_id, "caption": caption})
            seen.add(key)

    return figures


def _extract_tables(text):
    """Extract table captions or short table descriptors from plain PDF text."""
    pattern = re.compile(r"(?im)\b(Table)\s*(\d+)[\s.:;-]+([^\n]{1,300})")
    tables = []
    seen = set()

    for match in pattern.finditer(text):
        table_id = f"Table {match.group(2)}"
        content = re.sub(r"\s+", " ", match.group(3)).strip()
        key = (table_id, content)
        if content and key not in seen:
            tables.append({"id": table_id, "content": content})
            seen.add(key)

    return tables


def _extract_formulas(text):
    """Extract likely formula lines from plain PDF text."""
    formulas = []
    seen = set()

    def is_garbled_formula(line):
        bad_markers = ["\ufffd", "\u25a1", "\u25c6", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?"]
        if any(marker in line for marker in bad_markers):
            return True
        weird_chars = re.findall(r"[^\w\s.,;:!?%()\-/=<>+\[\]*]", line)
        if len(line) > 0 and len(weird_chars) / len(line) > 0.12:
            return True
        return False

    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if not cleaned or len(cleaned) > 220:
            continue
        if is_garbled_formula(cleaned):
            continue
        if re.search(r"[=<>+\-*/]|\b(eq\.?|equation)\b", cleaned, re.IGNORECASE):
            if cleaned not in seen:
                formulas.append(cleaned)
                seen.add(cleaned)

    return formulas[:30]


def _extract_metadata(text):
    """Extract a best-effort title and author string from the first page text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = ""
    authors = ""

    def looks_like_author_line(line):
        lower = line.lower()
        if not line or lower.startswith(("abstract", "keywords", "introduction")):
            return False
        if re.search(r"@|\d|\b(prof\.|university|laborator|department|institute|street|road|eindhoven|netherlands|china|usa)\b", lower):
            return False
        if re.search(r"\b(design|testing|test|system|chip|chips|optimal|infrastructure|conventional|resources|model|method|algorithm)\b", lower):
            return False
        words = re.findall(r"[A-Z][A-Za-z.\-]+", line)
        return 2 <= len(words) <= 10 and len(line) <= 120

    def looks_like_title(line):
        lower = line.lower()
        return (
            len(line) > 8
            and not lower.startswith(("abstract", "keywords", "introduction"))
            and not re.search(r"@|\b(prof\.|university|laborator|department|institute|street|road)\b", lower)
        )

    title_parts = []
    for line in lines[:12]:
        if title_parts and looks_like_author_line(line):
            break
        if looks_like_title(line):
            title_parts.append(line)
            continue
        break

    if title_parts:
        title = " ".join(title_parts[:3])

    start_index = max(len(title_parts), 1)
    for line in lines[start_index : start_index + 12]:
        if looks_like_author_line(line):
            authors = line
            break
        if re.search(r"\b(author|authors)\b", line, re.IGNORECASE):
            authors = line
            break

    if not authors:
        for line in lines[:50]:
            if looks_like_author_line(line) or re.search(r"\b(author|authors)\b", line, re.IGNORECASE):
                authors = line
                break

    return {"title": title, "authors": authors}


def load_pdf(file_path):
    """Read a PDF file and return a unified multimodal paper structure."""
    reader = PdfReader(file_path)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    full_text = "\n\n".join(pages)

    return {
        "text": full_text,
        "figures": _extract_figures(full_text),
        "tables": _extract_tables(full_text),
        "formulas": _extract_formulas(full_text),
        "metadata": _extract_metadata(full_text),
    }
