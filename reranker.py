"""Cross-encoder reranker — stage 2 of two-stage retrieval.

The embedder (embedder.py) is a BI-ENCODER: it turns the question and each chunk
into vectors SEPARATELY, then ranks by cosine distance. Fast (you can search all
34k chunks), but the query and a document never interact, so a chunk that merely
shares vocabulary can outrank the truly relevant one.

A CROSS-ENCODER reads (question, chunk) together as a single input and outputs a
relevance score. Much more accurate — and much slower, because there's no
reusable vector: every (question, chunk) pair is a fresh forward pass. So we
never run it over the whole corpus. Instead:

    bi-encoder retrieves top-N candidates (cheap)  ->  cross-encoder reranks -> top-k

Loaded once and cached, exactly like embedder._model(), so every caller shares a
single in-memory copy on the GPU.
"""
from functools import lru_cache

from sentence_transformers import CrossEncoder

import config


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    """Load (and cache) the cross-encoder on first call. Downloaded + cached by
    huggingface on first use, so later runs are fast and offline-friendly."""
    return CrossEncoder(config.RERANK_MODEL)


def rerank(question: str, hits, top_k: int = config.TOP_K):
    """Reorder `hits` by cross-encoder relevance and return the top_k.

    `hits` are (text, meta, distance) tuples from query.retrieve(). We keep the
    ORIGINAL cosine distance in each tuple untouched and only change the ORDER —
    so downstream code still sees the true bi-encoder distance, and you can watch
    cos_dist come back out of order as visible proof the reranker re-ranked.
    """
    if not hits:
        return hits
    # One (query, passage) pair per candidate — the cross-encoder scores each.
    pairs = [[question, text] for text, _meta, _dist in hits]
    scores = _model().predict(pairs)  # higher = more relevant
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    return [hits[i] for i in order[:top_k]]
