"""Ingest the Wikipedia BACKGROUND corpus into the main index, tagged as reference.

    python wiki.py                  # step 1: cache the articles (once, needs network)
    python ingest_reference.py      # step 2: chunk -> embed -> Chroma (offline)
    python ingest_reference.py --stamp-research   # also tag existing PubMed chunks

Why the SAME collection as the papers rather than a separate one? Because the
whole point is that one question should reach whichever source actually answers
it: "what is a total knee arthroplasty?" should surface the background article,
"what's the DVT risk after one?" should surface the papers. Two collections would
force the caller to decide which to search BEFORE knowing the answer — exactly
the guess we want retrieval to make for us. The `source_type` tag keeps them
distinguishable ("reference" vs "research") for labeling and filtering.

Access labels are stamped at ingest (same scheme as the papers, keyed on the
Wikipedia pageid instead of a PMID), so background chunks are access-controlled
identically — a background article can't sneak past a clearance filter.
"""
import argparse

import chromadb

import access
import config
import wiki
from embedder import embed_documents
from ingest import chunk_text


def stamp_research(col) -> None:
    """Tag the EXISTING PubMed chunks source_type='research' (metadata only).

    Without this the papers have no source_type at all, and a `where` filter on
    it would silently exclude them — worse than not filtering. Chroma has no
    "update all", so we page through and rewrite in batches; nothing re-embeds.
    """
    total = col.count()
    print(f"Stamping source_type='research' on existing chunks in '{col.name}' …")
    done, B = 0, 5000
    while done < total:
        got = col.get(limit=B, offset=done, include=["metadatas"])
        ids = got["ids"]
        if not ids:
            break
        metas = []
        for m in got["metadatas"]:
            m = dict(m or {})
            # Don't clobber the reference chunks if this is re-run after ingest.
            m["source_type"] = m.get("source_type") or "research"
            metas.append(m)
        col.update(ids=ids, metadatas=metas)
        done += len(ids)
        print(f"  {done}/{total}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest the Wikipedia background corpus.")
    ap.add_argument("--limit", type=int, default=None, help="Use only the first N articles.")
    ap.add_argument("--stamp-research", action="store_true",
                    help="Also tag existing PubMed chunks source_type='research'.")
    args = ap.parse_args()

    docs = wiki.load_reference(limit=args.limit)
    if not docs:
        raise SystemExit("No cached articles — run `python wiki.py` first.")
    print(f"Loaded {len(docs)} background articles from {config.REFERENCE_CACHE}")

    ids, texts, metadatas = [], [], []
    for d in docs:
        # Classification keys on a "wiki-<pageid>" string rather than the bare
        # pageid: the id space is disjoint from PMIDs, so a page can't inherit a
        # paper's level by numeric collision.
        level = access.classify(f"wiki-{d['pageid']}")
        for i, chunk in enumerate(chunk_text(d["text"])):
            ids.append(f'wiki{d["pageid"]}-{i}')
            texts.append(chunk)
            metadatas.append({
                "pageid": d["pageid"], "title": d["title"], "url": d["url"],
                "chunk_index": i,
                "source_type": "reference",          # vs "research" for PubMed
                "source": "Wikipedia (CC BY-SA)",    # attribution the licence requires
                "classification": level,
                "compartment": d.get("compartment", "general"),
            })
    print(f"{len(docs)} articles -> {len(texts)} chunks")

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    col = client.get_or_create_collection(config.COLLECTION_ORTHO,
                                          metadata={"hnsw:space": "cosine"})

    if args.stamp_research:
        stamp_research(col)

    print(f"Embedding {len(texts)} chunks locally with {config.EMBED_MODEL} …")
    embeddings = embed_documents(texts)

    B = 5000
    for i in range(0, len(ids), B):
        col.upsert(ids=ids[i:i + B], documents=texts[i:i + B],
                   embeddings=embeddings[i:i + B], metadatas=metadatas[i:i + B])
    print(f"Done. '{config.COLLECTION_ORTHO}' now holds {col.count()} chunks.")


if __name__ == "__main__":
    main()
