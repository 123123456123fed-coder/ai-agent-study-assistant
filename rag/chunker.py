"""Text chunking for RAG."""

import re


def _tokenize(text):
    """Tokenize English words, numbers, and Chinese characters."""
    return re.findall(r"[A-Za-z0-9_+\-/.]+|[\u4e00-\u9fff]|[^\s]", text)


def _join_tokens(tokens):
    """Join tokens into readable chunk text."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([([%])\s+", r"\1", text)
    text = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)
    return text.strip()


def _split_sections(text):
    """Split text by common academic section headings when possible."""
    heading_pattern = re.compile(
        r"(?im)^(abstract|introduction|method|methods|methodology|"
        r"contribution|contributions|experiment|experiments|evaluation|"
        r"results|conclusion|future work|references)\s*:?\s*$"
    )
    matches = list(heading_pattern.finditer(text))
    if len(matches) < 2:
        return [text]

    sections = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)
    return sections


def split_text(text, chunk_size=420, overlap=80):
    """
    Split text into overlapping token chunks.

    Each chunk is about 300-500 tokens by default, with 50-100 tokens overlap.
    """
    chunks = []
    step = max(chunk_size - overlap, 1)

    for section in _split_sections(text.strip()):
        tokens = _tokenize(section)
        if not tokens:
            continue

        if len(tokens) <= chunk_size:
            chunks.append(_join_tokens(tokens))
            continue

        start = 0
        while start < len(tokens):
            chunk_tokens = tokens[start : start + chunk_size]
            chunk = _join_tokens(chunk_tokens)
            if chunk:
                chunks.append(chunk)
            start += step

    return chunks
