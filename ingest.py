"""Step 1 — download -> chunk -> embed -> store.

    python ingest.py              # full pqa_labeled (1,000 abstracts)
    python ingest.py --limit 50   # tiny sample for fast development

This script costs $0 in API usage: embeddings are computed locally. It is
idempotent — if the collection already holds at least as many chunks as we're
about to add, it exits without re-embedding.
"""
import argparse
import re

import chromadb
from datasets import load_dataset

import config
from embedder import embed_documents, tokenizer


# --- Chunking --------------------------------------------------------------
# A sentence-aware chunker: greedily pack whole sentences until adding the next
# one would exceed CHUNK_MAX_TOKENS, then start a new chunk (carrying a little
# overlap so a fact split across a boundary stays retrievable). We measure with
# the *actual* bge tokenizer rather than guessing from word counts.

def split_sentences(text: str) -> list[str]:
    """Naive sentence split — fine for abstracts. Real systems might use spaCy."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_text(text: str) -> list[str]:
    tok = tokenizer()

    def n_tokens(s: str) -> int:
        return len(tok.encode(s, add_special_tokens=False))

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in split_sentences(text):
        sent_tokens = n_tokens(sent)
        # If this sentence would overflow the current chunk, flush it first.
        if current and current_tokens + sent_tokens > config.CHUNK_MAX_TOKENS:
            chunks.append(" ".join(current))
            # Start the next chunk with the last N sentences for continuity.
            current = current[-config.CHUNK_OVERLAP_SENTENCES:] if config.CHUNK_OVERLAP_SENTENCES else []
            current_tokens = sum(n_tokens(s) for s in current)
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


# --- Corpus ----------------------------------------------------------------

def load_corpus(limit: int | None = None) -> list[dict]:
    """Return one document per abstract.

    PubMedQA stores each abstract split into labeled sections under
    context.contexts; we join them back into a single document and keep the
    `pubid` so eval.py can check whether the *right* abstract was retrieved.
    The question is kept only for reference/inspection.
    """
    ds = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split=config.DATASET_SPLIT)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    docs = []
    for row in ds:
        abstract = " ".join(row["context"]["contexts"])
        docs.append(
            {
                "pubid": str(row["pubid"]),
                "question": row["question"],
                "text": abstract,
            }
        )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PubMedQA contexts into Chroma.")
    parser.add_argument("--limit", type=int, default=None, help="Only ingest the first N abstracts.")
    args = parser.parse_args()

    print(f"Loading {config.DATASET_NAME} [{config.DATASET_CONFIG}] ...")
    docs = load_corpus(args.limit)

    # Chunk every document, keeping provenance (pubid) on each chunk.
    ids, texts, metadatas = [], [], []
    for d in docs:
        for i, chunk in enumerate(chunk_text(d["text"])):
            ids.append(f'{d["pubid"]}-{i}')
            texts.append(chunk)
            metadatas.append(
                {"pubid": d["pubid"], "question": d["question"], "chunk_index": i}
            )
    print(f"{len(docs)} abstracts -> {len(texts)} chunks")

    # Connect to (or create) the on-disk vector store. Cosine space pairs with
    # our normalized embeddings.
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_or_create_collection(
        config.COLLECTION_PUBMEDQA, metadata={"hnsw:space": "cosine"}
    )

    # Idempotency: skip the expensive embedding step if we're already ingested.
    if len(ids) > 0 and collection.count() >= len(ids):
        print(f"Already ingested ({collection.count()} chunks) — nothing to do.")
        return

    print(f"Embedding {len(texts)} chunks locally with {config.EMBED_MODEL} ...")
    embeddings = embed_documents(texts)  # no API cost — runs on your machine

    # upsert (not add) so re-running with a larger --limit just tops up the store.
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    print(f"Done. Collection '{config.COLLECTION_PUBMEDQA}' holds {collection.count()} chunks at {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
