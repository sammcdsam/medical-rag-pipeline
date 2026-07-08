"""Ingest PMC full-text articles: section-aware chunking -> embed -> Chroma.

    python pmc.py --target 500          # step 1: cache OA full text (once)
    python ingest_fulltext.py           # step 2: chunk + embed into COLLECTION_FULLTEXT
    python ingest_fulltext.py --rebuild # wipe and rebuild

SECTION-AWARE chunking: each JATS section is chunked independently (a chunk never
crosses a section boundary), and every chunk carries its section heading as
metadata — so a retrieved chunk knows whether it came from Methods, Results, etc.,
which matters a lot more for long full text than for a short abstract.

Access labels are stamped at ingest: `classification` is the same deterministic
hash of the PMID used for the abstract corpus (so a paper's full text inherits its
abstract's clearance), and `compartment` is the subtopic — so this corpus is
access-controllable with the exact same pre-filter as the abstracts.
"""
import argparse
import os

import chromadb

import access
import config
import pmc
from embedder import embed_documents
from ingest import chunk_text   # same sentence-aware, token-bounded chunker


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PMC full-text articles into Chroma.")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N cached articles.")
    parser.add_argument("--rebuild", action="store_true", help="Delete the collection and rebuild.")
    args = parser.parse_args()

    if not os.path.exists(config.FULLTEXT_CACHE):
        raise SystemExit("No full-text cache found. Run `python pmc.py --target N` first.")

    docs = pmc.load_fulltext(limit=args.limit)
    print(f"Loaded {len(docs)} full-text articles from {config.FULLTEXT_CACHE}")

    ids, texts, metadatas = [], [], []
    for d in docs:
        pmid = d["pmid"]
        chunk_i = 0
        for sec_i, sec in enumerate(d["sections"]):
            for chunk in chunk_text(sec["text"]):
                ids.append(f"{pmid}-{chunk_i}")
                texts.append(chunk)
                metadatas.append({
                    "pmid": pmid, "pmcid": d.get("pmcid", ""),
                    "title": d.get("title", ""), "journal": d.get("journal", ""),
                    "year": d.get("year", ""),
                    "section": sec["heading"], "section_index": sec_i, "chunk_index": chunk_i,
                    "compartment": d.get("subtopic") or "(untagged)",
                    "classification": access.classify(pmid),   # inherits the abstract's clearance
                })
                chunk_i += 1
    print(f"{len(docs)} articles -> {len(texts)} chunks "
          f"({len(texts) / max(len(docs), 1):.0f} chunks/article)")

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    if args.rebuild:
        try:
            client.delete_collection(config.COLLECTION_FULLTEXT)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        config.COLLECTION_FULLTEXT, metadata={"hnsw:space": "cosine"}
    )

    print(f"Embedding {len(texts)} chunks locally with {config.EMBED_MODEL} (GPU) ...")
    embeddings = embed_documents(texts)

    B = 5000   # Chroma caps a single upsert (~5,461 items)
    for i in range(0, len(ids), B):
        collection.upsert(
            ids=ids[i:i + B], documents=texts[i:i + B],
            embeddings=embeddings[i:i + B], metadatas=metadatas[i:i + B],
        )
    print(f"Done. '{config.COLLECTION_FULLTEXT}' now holds {collection.count()} chunks.")


if __name__ == "__main__":
    main()
