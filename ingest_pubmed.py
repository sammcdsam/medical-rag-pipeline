"""Step 2 (local, offline, repeatable): local cache -> chunk -> embed -> Chroma.

    python download_corpus.py        # step 1 first: PubMed -> ortho_corpus.jsonl (once)
    python ingest_pubmed.py          # step 2: build the index from the cache (no network)
    python ingest_pubmed.py --limit 500   # use only the first 500 cached abstracts
    python ingest_pubmed.py --rebuild     # wipe the collection and rebuild

Reads the LOCAL corpus cache (no PubMed calls), so you can re-chunk / re-embed
as often as you like without re-downloading — and it works offline. Reuses the
same chunk->embed pipeline as the PubMedQA baseline; provenance is the real PMID
+ article title. Embeddings run on the GPU, so this costs $0 in API usage.
"""
import argparse
import os

import chromadb

import config
import pubmed
from embedder import embed_documents
from ingest import chunk_text   # same sentence-aware chunker as the baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the local orthopedic corpus into Chroma.")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N cached abstracts.")
    parser.add_argument("--rebuild", action="store_true", help="Delete the collection and rebuild.")
    args = parser.parse_args()

    # The cache is the source of truth. Download once if it's missing.
    if not os.path.exists(config.CORPUS_CACHE):
        print("No local corpus cache found — downloading once (resumable)...")
        pubmed.download_corpus()

    docs = pubmed.load_corpus(limit=args.limit)
    print(f"Loaded {len(docs)} abstracts from {config.CORPUS_CACHE}")

    # Chunk every abstract, carrying provenance (pmid + title) onto each chunk.
    ids, texts, metadatas = [], [], []
    for d in docs:
        for i, chunk in enumerate(chunk_text(d["text"])):
            ids.append(f'{d["pmid"]}-{i}')
            texts.append(chunk)
            metadatas.append({
                "pmid": d["pmid"], "title": d["title"],
                "journal": d.get("journal", ""), "year": d.get("year", ""), "chunk_index": i,
            })
    print(f"{len(docs)} abstracts -> {len(texts)} chunks")

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    if args.rebuild:
        try:
            client.delete_collection(config.COLLECTION_ORTHO)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        config.COLLECTION_ORTHO, metadata={"hnsw:space": "cosine"}
    )

    print(f"Embedding {len(texts)} chunks locally with {config.EMBED_MODEL} ...")
    embeddings = embed_documents(texts)   # GPU, no API cost

    # Chroma caps a single upsert (~5,461 items), so write in batches.
    B = 5000
    for i in range(0, len(ids), B):
        collection.upsert(
            ids=ids[i:i + B], documents=texts[i:i + B],
            embeddings=embeddings[i:i + B], metadatas=metadatas[i:i + B],
        )
    print(f"Done. '{config.COLLECTION_ORTHO}' now holds {collection.count()} chunks at {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
