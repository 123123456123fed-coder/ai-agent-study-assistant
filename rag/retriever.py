"""FAISS retriever for text-only RAG over the loaded paper."""

import re

import faiss

from rag.chunker import split_text
from rag.embedder import build_embeddings, encode_query, get_backend, keyword_boost
from utils.pdf_loader import load_pdf


MIN_SIMILARITY = 0.2

_index = None
_chunks = []
_document_text = ""
_pdf_data = None


def _safe_preview(text, limit=240):
    """Return a console-safe one-line preview for debug output."""
    preview = re.sub(r"\s+", " ", text[:limit]).strip()
    return preview.encode("gbk", errors="replace").decode("gbk", errors="replace")


def _quality_penalty(chunk):
    """Penalize noisy formula/table/reference chunks that make poor RAG context."""
    text = chunk.strip()
    if not text:
        return 0.5

    control_chars = sum(1 for char in text if ord(char) < 32 and char not in "\n\t")
    digits = sum(1 for char in text if char.isdigit())
    letters = sum(1 for char in text if char.isalpha())
    total = max(len(text), 1)

    penalty = 0.0
    if control_chars / total > 0.01:
        penalty += 0.3
    if digits / total > 0.3 and letters / total < 0.4:
        penalty += 0.25
    normalized_lower = re.sub(r"\s+", "", text.lower())
    if ("table" in normalized_lower and digits > 40) or digits > 120:
        penalty += 0.55
    if text.lower().startswith("references"):
        penalty += 0.3
    if len(re.findall(r"[^\w\s.,;:!?%()\-/]", text)) / total > 0.18:
        penalty += 0.15

    return min(penalty, 0.6)


def build_index(text_chunks):
    """Build a fresh FAISS index from text chunks."""
    global _index, _chunks

    _chunks = [chunk for chunk in text_chunks if chunk.strip()]
    _index = None

    if not _chunks:
        return None

    vectors = build_embeddings(_chunks)
    _index = faiss.IndexFlatIP(vectors.shape[1])
    _index.add(vectors)

    if _index.ntotal == 0:
        raise RuntimeError("FAISS index build failed: index is empty.")

    return _index


def build_from_data(pdf_data):
    """Build the RAG index from the unified PDF data structure."""
    global _document_text, _pdf_data

    _pdf_data = pdf_data
    _document_text = pdf_data.get("text", "")
    if not _document_text.strip():
        raise ValueError("Failed to extract text from this PDF.")

    chunks = split_text(_document_text)
    build_index(chunks)
    return {"chunk_count": len(chunks), "backend": get_backend(), "index_size": get_index_size()}


def build_from_text(text):
    """Build the RAG index from raw text for compatibility."""
    return build_from_data(
        {
            "text": text,
            "figures": [],
            "tables": [],
            "formulas": [],
            "metadata": {"title": "", "authors": ""},
        }
    )


def build_from_pdf(file_path):
    """Load a PDF and build the text-only RAG index."""
    return build_from_data(load_pdf(file_path))


def search(query, top_k=3):
    """Search the FAISS index and return true top-k chunks with similarity scores."""
    if _index is None or not _chunks or _index.ntotal == 0:
        return []

    query_vector = encode_query(query)
    if query_vector is None:
        return []

    k = min(max(top_k * 5, 10), len(_chunks))
    scores, indices = _index.search(query_vector, k)
    candidates = []

    for score, index in zip(scores[0], indices[0]):
        if index < 0:
            continue

        raw_similarity = float(score)
        chunk = _chunks[index]
        similarity = min(raw_similarity + keyword_boost(query, chunk), 1.0)
        similarity = max(similarity - _quality_penalty(chunk), 0.0)
        print(f"RAG similarity={similarity:.4f}, raw={raw_similarity:.4f}, chunk={_safe_preview(chunk)}")

        if similarity >= MIN_SIMILARITY:
            candidates.append({"text": chunk, "score": similarity})

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]


def get_index_size():
    """Return current FAISS index size."""
    if _index is None:
        return 0
    return int(_index.ntotal)


def get_status():
    """Return retriever status for UI display."""
    return {"backend": get_backend(), "index_size": get_index_size(), "chunk_count": len(_chunks)}


def get_document_text():
    """Return the current loaded document text for compatibility."""
    return _document_text


def get_pdf_data():
    """Return the current unified PDF data structure."""
    return _pdf_data
