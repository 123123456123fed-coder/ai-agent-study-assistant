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
        "\ufffd", "\u25a1", "\u25c6", "\ufe49", "\ufe5f"
    ]
    if any(marker in text for marker in bad_markers):
        return True
    if any(ord(char) < 32 and char not in "\n\t" for char in text):
        return True
    weird_chars = re.findall(r"[^\w\s.,;:!?%()\-/=<>+\[\]*]", text)
    if len(text) > 0 and len(weird_chars) / len(text) > 0.12:
        return True
    return False


def _is_likely_formula(text):
    """Avoid presenting titles, addresses, and prose as formulas."""
    if len(text.strip()) < 5 or text.strip() in {"=", "<", ">"}:
        return False
    lower = text.lower()
    if lower.startswith(("fig", "figure", "table", "abstract", "keywords", "references")):
        return False
    if re.search(r"\b(prof\.|university|department|institute|road|street|holstlaan)\b", lower):
        return False

    has_relation = bool(re.search(r"(<=|>=|=|<|>)", text))
    has_equation_label = bool(re.search(r"\b(eq\.?|equation)\s*\d*", lower))
    if has_relation:
        pieces = re.split(r"<=|>=|=|<|>", text, maxsplit=1)
        if len(pieces) == 2 and (not pieces[0].strip() or not pieces[1].strip()):
            return False
    words = re.findall(r"[A-Za-z]+", text)
    math_tokens = re.findall(r"(<=|>=|=|<|>|\+|\*|/|\(|\)|\d+)", text)
    return (has_relation and (len(words) <= 16 or len(math_tokens) >= 3)) or (has_equation_label and has_relation)


def _compact_number(value):
    return re.sub(r"\s+", "", value)


def _normalize_formula(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(\d)\s*\.\s*((?:\d\s*)+)", lambda match: match.group(1) + "." + _compact_number(match.group(2)), text)
    if ", i.e." in text.lower():
        before_comma = re.split(r",\s*i\.e\.", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if re.search(r"(<=|>=|=|<|>)", before_comma):
            text = before_comma
    text = re.sub(r"\s*(<=|>=|=|<|>)\s*", r" \1 ", text)
    return text.strip()


def _expand_formula(text):
    separated = re.sub(r"(?<=\d)(?=[A-Za-z][A-Za-z0-9_]*\s*=)", "; ", text)
    parts = [part.strip() for part in separated.split(";") if part.strip()]
    assignments = []
    for part in parts:
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)\s*=\s*((?:\d\s*)+(?:\.\s*(?:\d\s*)+)?)", part)
        if match:
            assignments.append((match.group(1), match.group(2)))
    if len(assignments) >= 2:
        return [f"{name} = {_compact_number(value)}" for name, value in assignments]
    return [_normalize_formula(text)]


def _explain_formula(formula):
    """Give a domain-aware explanation for a DFT/test formula."""
    lower = formula.lower()
    parts = []
    numeric_inequality = re.fullmatch(r"\s*([0-9.]+)\s*(<=|>=|<|>)\s*([0-9.]+)\s*", formula)
    if numeric_inequality:
        left, op, right = numeric_inequality.groups()
        parts.append(f"\u8fd9\u662f\u4e00\u4e2a\u6570\u503c\u7ea6\u675f\uff1a\u5de6\u4fa7 {left} \u8868\u793a\u8bba\u6587\u63d0\u53d6\u5230\u7684\u8d44\u6e90\u9700\u6c42\u6216\u6d4b\u8bd5\u6307\u6807\uff0c\u53f3\u4fa7 {right} \u8868\u793a\u53ef\u7528\u8d44\u6e90\u6216\u4e0a\u9650\u3002")
        if op in {"<", "<="}:
            parts.append("\u5b83\u8bf4\u660e\u8be5\u9700\u6c42\u6ca1\u6709\u8d85\u8fc7\u7cfb\u7edf\u53ef\u63d0\u4f9b\u7684\u8d44\u6e90\uff0c\u56e0\u800c\u5728\u8fd9\u4e2a\u7ea6\u675f\u4e0b\u662f\u53ef\u884c\u7684\u3002")
    if any(term in lower for term in ["ate", "channel", "memory", "resource"]):
        parts.append("\u5b83\u63cf\u8ff0 ATE \u6d4b\u8bd5\u8d44\u6e90\uff08\u5982\u901a\u9053\u6570\u6216\u5411\u91cf\u5b58\u50a8\u6df1\u5ea6\uff09\u4e0e\u88ab\u6d4b SOC \u9700\u6c42\u4e4b\u95f4\u7684\u7ea6\u675f\u5173\u7cfb\u3002")
    if any(term in lower for term in ["site", "multi", "parallel"]):
        parts.append("\u5b83\u7528\u6765\u5224\u65ad\u5728\u591a\u7ad9\u70b9\u5e76\u884c\u6d4b\u8bd5\u65f6\uff0c\u540c\u65f6\u6d4b\u8bd5\u591a\u5c11\u4e2a SOC \u4e0d\u4f1a\u8d85\u51fa\u8d44\u6e90\u4e0a\u9650\u3002")
    if any(term in lower for term in ["time", "throughput", "index", "abort", "yield"]):
        parts.append("\u5b83\u548c\u6d4b\u8bd5\u65f6\u95f4\u3001\u541e\u5410\u91cf\u6216\u826f\u7387\u76f8\u5173\uff0c\u7528\u4e8e\u8bc4\u4f30\u6d4b\u8bd5\u6548\u7387\u3002")
    if re.search(r"\bpc\b", lower):
        parts.append("`pc` \u5728\u8fd9\u7c7b\u6d4b\u8bd5\u8bba\u6587\u4e2d\u901a\u5e38\u8868\u793a\u5355\u4e2a\u5668\u4ef6\u6216\u5355\u4e2a\u82af\u7247\u7684\u63a5\u89e6/\u826f\u7387\u6982\u7387\uff1b\u5b83\u8d8a\u63a5\u8fd1 1\uff0c\u8bf4\u660e\u6d4b\u8bd5\u63a5\u89e6\u6216\u6d4b\u8bd5\u6210\u529f\u7684\u6982\u7387\u8d8a\u9ad8\u3002")
    if re.search(r"\bpm\b", lower):
        parts.append("`pm` \u53ef\u7406\u89e3\u4e3a\u591a\u7ad9\u70b9\u6216\u591a\u6a21\u5757\u6d4b\u8bd5\u573a\u666f\u4e0b\u7684\u7efc\u5408\u6210\u529f\u6982\u7387\uff1b\u5b83\u4e0b\u964d\u65f6\uff0c\u5e76\u884c\u6d4b\u8bd5\u7684\u6709\u6548\u541e\u5410\u91cf\u901a\u5e38\u4f1a\u53d7\u5230\u5f71\u54cd\u3002")
    if "<" in formula or ">" in formula:
        parts.append("\u5176\u4e2d\u7684\u4e0d\u7b49\u5f0f\u8868\u793a\u7ea6\u675f\u6761\u4ef6\uff1a\u5de6\u4fa7\u9700\u6c42\u5fc5\u987b\u5c0f\u4e8e\u6216\u4e0d\u8d85\u8fc7\u53f3\u4fa7\u53ef\u7528\u8d44\u6e90\u3002")
    if "=" in formula:
        parts.append("\u5176\u4e2d\u7684\u7b49\u5f0f\u8868\u793a\u8ba1\u7b97\u5173\u7cfb\uff1a\u7528\u5df2\u77e5\u53c2\u6570\u63a8\u5bfc\u6d4b\u8bd5\u65f6\u95f4\u3001\u8d44\u6e90\u9700\u6c42\u6216\u541e\u5410\u91cf\u3002")

    if not parts:
        parts.append("\u8fd9\u662f\u8bba\u6587\u4e2d\u7684\u6570\u5b66\u5173\u7cfb\uff0c\u7528\u4e8e\u8868\u8fbe\u65b9\u6cd5\u4e2d\u7684\u7ea6\u675f\u3001\u8d44\u6e90\u6216\u6027\u80fd\u8ba1\u7b97\u3002")
    return "\u89e3\u91ca\uff1a" + "".join(parts)


def analyze_formulas(pdf_data):
    """Return formula list and lightweight explanations."""
    data = _as_pdf_data(pdf_data)
    formulas = []

    for item in data.get("formulas", []):
        if not item:
            continue
        cleaned = re.sub(r"\s+", " ", item).strip()
        if not cleaned or _looks_garbled(cleaned) or not _is_likely_formula(cleaned):
            continue
        for formula in _expand_formula(cleaned):
            if formula and _is_likely_formula(formula) and formula not in formulas:
                formulas.append(formula)

    if not formulas:
        return "\U0001F9EE \u516c\u5f0f\u89e3\u6790\n\u672a\u4ece PDF \u6587\u672c\u5c42\u4e2d\u63d0\u53d6\u5230\u53ef\u9760\u7684\u6e05\u6670\u516c\u5f0f\u3002\u8fd9\u901a\u5e38\u8bf4\u660e\u516c\u5f0f\u5728 PDF \u4e2d\u662f\u56fe\u7247\u3001\u7279\u6b8a\u5b57\u4f53\u6216\u88ab\u63d0\u53d6\u5668\u7834\u574f\u4e86\uff1b\u7cfb\u7edf\u5df2\u907f\u514d\u628a\u6807\u9898\u3001\u5730\u5740\u6216\u666e\u901a\u53e5\u5b50\u8bef\u5f53\u6210\u516c\u5f0f\u3002"

    lines = ["\U0001F9EE \u516c\u5f0f\u89e3\u6790", "\u4ee5\u4e0b\u53ea\u5217\u51fa PDF \u6587\u672c\u5c42\u4e2d\u80fd\u53ef\u9760\u8bc6\u522b\u7684\u516c\u5f0f\u6216\u7ea6\u675f\uff1a"]
    for index, formula in enumerate(formulas[:15], start=1):
        short_formula = re.sub(r"\s+", " ", formula)
        lines.append(f"- \u516c\u5f0f {index}\uff1a`{short_formula}`")
        lines.append(f"  {_explain_formula(short_formula)}")

    return "\n".join(lines)
