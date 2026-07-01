"""Embedding model management for RAG."""

import re

import streamlit as st
from sklearn.preprocessing import normalize


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

QUERY_EXPANSIONS = {
    "贡献": "contribution contributions novelty propose proposes proposed present presents model algorithm optimization optimal",
    "创新": "innovation novelty contribution proposed present presents",
    "核心": "core contribution objective problem",
    "方法": "method methodology approach design algorithm",
    "算法": "algorithm step optimization search",
    "实验": "experiment evaluation result results throughput",
    "结果": "experiment evaluation result results throughput",
    "结论": "conclusion future work",
    "摘要": "abstract summary",
    "总结": "abstract summary overview",
    "研究": "research problem objective",
    "芯片": "chip soc system-on-chip",
    "测试": "test testing throughput ATE",
    "图表": "figure table throughput vector memory",
    "公式": "equation formula constraint throughput probability",
    "DFT": "DFT design-for-test design for test testability",
}

_model = None
_tfidf_vectorizer = None
_backend = None


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the embedding model lazily and cache it for Streamlit Cloud."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _get_sentence_transformer():
    """Load one shared sentence-transformers model lazily."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


def expand_query(query):
    """Add small bilingual hints so Chinese questions can match English papers."""
    query_lower = query.lower()
    expansions = [extra for key, extra in QUERY_EXPANSIONS.items() if key.lower() in query_lower]
    if expansions:
        return query + " " + " ".join(expansions)
    return query


def build_embeddings(chunks):
    """Build normalized embeddings and remember the active backend."""
    global _backend, _tfidf_vectorizer

    try:
        model = _get_sentence_transformer()
        vectors = model.encode(chunks, convert_to_numpy=True).astype("float32")
        _backend = EMBEDDING_MODEL_NAME
    except Exception as exc:
        print(f"Sentence-transformers unavailable, using TF-IDF fallback: {exc}")
        from sklearn.feature_extraction.text import TfidfVectorizer

        _tfidf_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=4096,
            lowercase=True,
        )
        vectors = _tfidf_vectorizer.fit_transform(chunks).toarray().astype("float32")
        _backend = "tfidf-fallback"

    vectors = normalize(vectors)
    return vectors.astype("float32")


def encode_query(query):
    """Encode a query using the same backend as document embeddings."""
    return encode_query_with_state(query, _backend, _tfidf_vectorizer)


def encode_query_with_state(query, backend, vectorizer=None):
    """Encode a query with an explicit backend/vectorizer pair."""
    expanded_query = expand_query(query)

    if backend == EMBEDDING_MODEL_NAME:
        model = _get_sentence_transformer()
        vector = model.encode([expanded_query], convert_to_numpy=True).astype("float32")
        return normalize(vector).astype("float32")

    if backend == "tfidf-fallback" and vectorizer is not None:
        vector = vectorizer.transform([expanded_query]).toarray().astype("float32")
        return normalize(vector).astype("float32")

    return None


def keyword_boost(query, chunk):
    """Boost similarity when bilingual intent keywords appear in the chunk."""
    query_lower = query.lower()
    chunk_lower = chunk.lower()
    boost = 0.0

    for key, expansion in QUERY_EXPANSIONS.items():
        if key.lower() not in query_lower:
            continue

        terms = [term.lower() for term in expansion.split()]
        if any(re.search(rf"\b{re.escape(term)}\b", chunk_lower) for term in terms):
            boost += 0.25

    if any(term in query_lower for term in ["主要", "研究", "什么", "内容"]):
        if any(section in chunk_lower[:80] for section in ["abstract", "introduction"]):
            boost += 0.35
        if any(phrase in chunk_lower for phrase in ["we present", "this paper presents", "we propose"]):
            boost += 0.2

    if "贡献" in query_lower or "创新" in query_lower:
        if any(section in chunk_lower[:120] for section in ["abstract", "contribution", "contributions", "conclusion"]):
            boost += 0.25
        if any(phrase in chunk_lower for phrase in ["we present", "this paper presents", "we propose", "we focus"]):
            boost += 0.2

    return min(boost, 0.5)


def get_backend():
    """Return the active embedding backend name."""
    return _backend


def get_vectorizer():
    """Return the active TF-IDF vectorizer when fallback backend is used."""
    return _tfidf_vectorizer
