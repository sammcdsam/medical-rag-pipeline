"""Ingest the BACKGROUND corpora into the main index, tagged source_type=reference.

Two background sources, both tagged "reference" but distinguishable by `source`:

  - StatPearls  — PEER-REVIEWED clinical reference (statpearls.py). The good one:
                  section-structured as Indications / Technique / Complications.
  - Wikipedia   — encyclopedic fallback (wiki.py). Decent coverage, NOT peer
                  reviewed; kept for breadth where StatPearls has no chapter.

    python statpearls.py            # step 1a: extract chapters from the tarball
    python wiki.py                  # step 1b: cache the Wikipedia articles
    python ingest_reference.py      # step 2: chunk -> embed -> Chroma (offline)
    python ingest_reference.py --stamp-research   # also tag existing PubMed chunks
    python ingest_reference.py --only statpearls  # just one source

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
import statpearls
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


def wiki_chunks(limit=None):
    """Wikipedia articles -> (ids, texts, metadatas)."""
    ids, texts, metas = [], [], []
    for d in wiki.load_reference(limit=limit):
        # Classification keys on a "wiki-<pageid>" string rather than the bare
        # pageid: the id space is disjoint from PMIDs, so a page can't inherit a
        # paper's level by numeric collision.
        level = access.classify(f"wiki-{d['pageid']}")
        for i, chunk in enumerate(chunk_text(d["text"])):
            ids.append(f'wiki{d["pageid"]}-{i}')
            texts.append(chunk)
            metas.append({
                "pageid": d["pageid"], "title": d["title"], "url": d["url"],
                "chunk_index": i,
                "source_type": "reference",          # vs "research" for PubMed
                "source": "Wikipedia (CC BY-SA)",    # attribution the licence requires
                "classification": level,
                "compartment": d.get("compartment", "general"),
            })
    return ids, texts, metas


def statpearls_chunks(limit=None):
    """StatPearls chapters -> (ids, texts, metadatas), chunked section-aware.

    A chunk never crosses a section boundary (same rule as ingest_fulltext.py):
    "Indications" and "Complications" answer different questions, and a chunk
    spanning both retrieves badly for either. The heading rides along as metadata
    so an answer can say WHICH part of the chapter it came from.
    """
    ids, texts, metas = [], [], []
    for d in statpearls.load(limit=limit):
        cid = d["chapter_id"]
        level = access.classify(f"statpearls-{cid}")
        for s_i, sec in enumerate(d["sections"]):
            for c_i, chunk in enumerate(chunk_text(sec["text"])):
                ids.append(f"sp{cid}-{s_i}-{c_i}")
                texts.append(chunk)
                metas.append({
                    "chapter_id": cid, "title": d["title"],
                    "section": sec["heading"], "section_index": s_i, "chunk_index": c_i,
                    "source_type": "reference",
                    "source": d["source"],      # CC BY-NC-ND requires the credit line
                    "url": f"https://www.ncbi.nlm.nih.gov/books/NBK430685/",
                    "classification": level,
                    "compartment": d.get("compartment", "general"),
                })
    return ids, texts, metas


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest the background reference corpora.")
    ap.add_argument("--limit", type=int, default=None, help="Use only the first N documents per source.")
    ap.add_argument("--only", choices=["statpearls", "wiki"], default=None,
                    help="Ingest just one source (default: both).")
    ap.add_argument("--stamp-research", action="store_true",
                    help="Also tag existing PubMed chunks source_type='research'.")
    args = ap.parse_args()

    ids, texts, metadatas = [], [], []
    if args.only in (None, "statpearls"):
        i, t, m = statpearls_chunks(args.limit)
        print(f"StatPearls  -> {len(t):>6,} chunks")
        ids += i; texts += t; metadatas += m
    if args.only in (None, "wiki"):
        i, t, m = wiki_chunks(args.limit)
        print(f"Wikipedia   -> {len(t):>6,} chunks")
        ids += i; texts += t; metadatas += m
    if not texts:
        raise SystemExit("Nothing to ingest — run `python statpearls.py` / `python wiki.py` first.")
    print(f"total       -> {len(texts):>6,} chunks")

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
