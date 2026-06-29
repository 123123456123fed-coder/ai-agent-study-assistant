"""Backward-compatible wrapper for the modular RAG retriever.

New code should import rag.retriever, rag.embedder, and rag.chunker directly.
This wrapper keeps older imports working during development.
"""

from rag.chunker import split_text
from rag.embedder import get_backend
from rag.retriever import (
    build_from_pdf,
    build_from_text,
    build_index,
    get_document_text,
    get_index_size,
    get_status,
    search,
)


__all__ = [
    "split_text",
    "get_backend",
    "build_index",
    "build_from_text",
    "build_from_pdf",
    "search",
    "get_index_size",
    "get_status",
    "get_document_text",
]
