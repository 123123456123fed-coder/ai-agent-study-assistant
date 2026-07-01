"""FAISS retriever with one isolated index per paper."""

import re

import faiss

from rag.chunker import split_text
from rag.embedder import (
    build_embeddings,
    encode_query_with_state,
    get_backend,
    get_vectorizer,
    keyword_boost,
)
from utils.pdf_loader import load_pdf


MIN_SIMILARITY = 0.2
DEFAULT_PAPER_ID = "__default__"

paper_index_map = {}
_active_paper_id = None


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


def _resolve_paper_id(paper_id=None):
    """Return explicit paper_id or the active/default paper id."""
    if paper_id:
        return paper_id
    if _active_paper_id:
        return _active_paper_id
    return DEFAULT_PAPER_ID


def set_active_paper(paper_id):
    """Set the active paper for backward-compatible calls."""
    global _active_paper_id
    _active_paper_id = paper_id


def has_index(paper_id=None):
    """Return whether a paper has an index."""
    return _resolve_paper_id(paper_id) in paper_index_map


def build_index(text_chunks, paper_id=DEFAULT_PAPER_ID, pdf_data=None, document_text=""):
    """Build and store a FAISS index isolated by paper_id."""
    chunks = [chunk for chunk in text_chunks if chunk.strip()]
    if not chunks:
        return None

    vectors = build_embeddings(chunks)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    if index.ntotal == 0:
        raise RuntimeError("FAISS index build failed: index is empty.")

    paper_index_map[paper_id] = {
        "index": index,
        "chunks": chunks,
        "pdf_data": pdf_data,
        "document_text": document_text,
        "backend": get_backend(),
        "vectorizer": get_vectorizer(),
    }
    set_active_paper(paper_id)
    return index


def build_from_data(pdf_data, paper_id=DEFAULT_PAPER_ID):
    """Build the RAG index from the unified PDF data structure for one paper."""
    document_text = pdf_data.get("text", "")
    if not document_text.strip():
        raise ValueError("Failed to extract text from this PDF.")

    chunks = split_text(document_text)
    build_index(chunks, paper_id=paper_id, pdf_data=pdf_data, document_text=document_text)
    return {
        "paper_id": paper_id,
        "chunk_count": len(chunks),
        "backend": get_status(paper_id).get("backend"),
        "index_size": get_index_size(paper_id),
    }


def build_from_text(text, paper_id=DEFAULT_PAPER_ID):
    """Build the RAG index from raw text for compatibility."""
    return build_from_data(
        {
            "text": text,
            "figures": [],
            "tables": [],
            "formulas": [],
            "metadata": {"title": "", "authors": ""},
        },
        paper_id=paper_id,
    )


def build_from_pdf(file_path, paper_id=DEFAULT_PAPER_ID):
    """Load a PDF and build the text-only RAG index."""
    return build_from_data(load_pdf(file_path), paper_id=paper_id)


def search(query, top_k=3, paper_id=None):
    """Search one paper's FAISS index and return top-k chunks."""
    resolved_id = _resolve_paper_id(paper_id)
    state = paper_index_map.get(resolved_id)
    if not state:
        return []

    index = state["index"]
    chunks = state["chunks"]
    if index is None or not chunks or index.ntotal == 0:
        return []

    query_vector = encode_query_with_state(query, state.get("backend"), state.get("vectorizer"))
    if query_vector is None:
        return []

    k = min(max(top_k * 5, 10), len(chunks))
    scores, indices = index.search(query_vector, k)
    candidates = []

    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index < 0:
            continue

        raw_similarity = float(score)
        chunk = chunks[chunk_index]
        similarity = min(raw_similarity + keyword_boost(query, chunk), 1.0)
        similarity = max(similarity - _quality_penalty(chunk), 0.0)
        print(
            f"RAG paper={resolved_id} similarity={similarity:.4f}, "
            f"raw={raw_similarity:.4f}, chunk={_safe_preview(chunk)}"
        )

        if similarity >= MIN_SIMILARITY:
            candidates.append({"text": chunk, "score": similarity, "paper_id": resolved_id})

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]


def get_index_size(paper_id=None):
    """Return the FAISS index size for one paper."""
    state = paper_index_map.get(_resolve_paper_id(paper_id))
    if not state or state.get("index") is None:
        return 0
    return int(state["index"].ntotal)


def get_status(paper_id=None):
    """Return retriever status for UI display."""
    state = paper_index_map.get(_resolve_paper_id(paper_id))
    if not state:
        return {"backend": None, "index_size": 0, "chunk_count": 0}
    return {
        "backend": state.get("backend"),
        "index_size": get_index_size(paper_id),
        "chunk_count": len(state.get("chunks", [])),
    }


def get_document_text(paper_id=None):
    """Return one paper's loaded document text."""
    state = paper_index_map.get(_resolve_paper_id(paper_id))
    if not state:
        return ""
    return state.get("document_text", "")


def get_pdf_data(paper_id=None):
    """Return one paper's unified PDF data structure."""
    state = paper_index_map.get(_resolve_paper_id(paper_id))
    if not state:
        return None
    return state.get("pdf_data")
