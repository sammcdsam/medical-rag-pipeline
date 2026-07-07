"""Local embedding model, loaded exactly once.

Why a module? So ingest / query / eval all share a single in-memory copy of
bge-small instead of each loading its own. The model weights are downloaded on
first use and cached by sentence-transformers (~/.cache/huggingface), so later
runs are fast and offline-friendly. We never re-download or re-embed on a whim.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load (and cache) the embedding model on first call."""
    return SentenceTransformer(config.EMBED_MODEL)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed passages/chunks for storage.

    normalize_embeddings=True returns unit vectors, so a plain dot product
    equals cosine similarity — which is the distance space we tell Chroma to use.
    """
    vecs = _model().encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 64,
    )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a search query. Note the bge query prefix — documents don't get it."""
    vec = _model().encode(config.QUERY_PREFIX + text, normalize_embeddings=True)
    return vec.tolist()


def tokenizer():
    """The model's tokenizer — used by the chunker to count real tokens."""
    return _model().tokenizer
