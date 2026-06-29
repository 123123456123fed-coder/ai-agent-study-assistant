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
        bad_markers = ["\ufffd", "\u25a1", "\u25c6", "\ufe49", "\ufe5f"]
        if any(marker in line for marker in bad_markers):
            return True
        if any(ord(char) < 32 and char not in "\n\t" for char in line):
            return True
        weird_chars = re.findall(r"[^\w\s.,;:!?%()\-/=<>+\[\]*]", line)
        if len(line) > 0 and len(weird_chars) / len(line) > 0.12:
            return True
        return False

    def is_likely_formula(line):
        if len(line.strip()) < 5 or line.strip() in {"=", "<", ">"}:
            return False
        lower = line.lower()
        if lower.startswith(("fig", "figure", "table", "abstract", "keywords", "references")):
            return False
        if re.search(r"\b(prof\.|university|department|institute|road|street|holstlaan)\b", lower):
            return False

        has_relation = bool(re.search(r"(<=|>=|=|<|>)", line))
        has_equation_label = bool(re.search(r"\b(eq\.?|equation)\s*\d*", lower))
        if not has_relation and not has_equation_label:
            return False
        if has_relation:
            pieces = re.split(r"<=|>=|=|<|>", line, maxsplit=1)
            if len(pieces) == 2 and (not pieces[0].strip() or not pieces[1].strip()):
                return False

        words = re.findall(r"[A-Za-z]+", line)
        math_tokens = re.findall(r"(<=|>=|=|<|>|\+|\*|/|\(|\)|\d+)", line)
        if has_relation and (len(words) <= 16 or len(math_tokens) >= 3):
            return True
        if has_equation_label and has_relation:
            return True
        return False

    def compact_number(value):
        return re.sub(r"\s+", "", value)

    def normalize_formula(line):
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"(\d)\s*\.\s*((?:\d\s*)+)", lambda match: match.group(1) + "." + compact_number(match.group(2)), line)
        if ", i.e." in line.lower():
            before_comma = re.split(r",\s*i\.e\.", line, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            if re.search(r"(<=|>=|=|<|>)", before_comma):
                line = before_comma
        line = re.sub(r"\s*(<=|>=|=|<|>)\s*", r" \1 ", line)
        return line.strip()

    def expand_formula(line):
        separated = re.sub(r"(?<=\d)(?=[A-Za-z][A-Za-z0-9_]*\s*=)", "; ", line)
        parts = [part.strip() for part in separated.split(";") if part.strip()]
        assignments = []
        for part in parts:
            match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)\s*=\s*((?:\d\s*)+(?:\.\s*(?:\d\s*)+)?)", part)
            if match:
                assignments.append((match.group(1), match.group(2)))
        if len(assignments) >= 2:
            return [f"{name} = {compact_number(value)}" for name, value in assignments]
        return [normalize_formula(line)]

    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if not cleaned or len(cleaned) > 220:
            continue
        if is_garbled_formula(cleaned):
            continue
        if is_likely_formula(cleaned):
            for formula in expand_formula(cleaned):
                if formula not in seen and is_likely_formula(formula):
                    formulas.append(formula)
                    seen.add(formula)

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
