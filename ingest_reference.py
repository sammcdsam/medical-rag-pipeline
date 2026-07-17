"""Ingest the StatPearls BACKGROUND corpus, tagged source_type=reference.

The PubMed corpus is all RESEARCH: it reports what studies found, but nothing in
it explains what a procedure IS. StatPearls is the missing layer — a
peer-reviewed clinical reference, section-structured as exactly the background a
clinician wants: Indications / Technique or Treatment / Complications.

    python statpearls.py            # step 1: extract chapters from the tarball
    python ingest_reference.py      # step 2: chunk -> embed -> Chroma (offline)
    python ingest_reference.py --stamp-research   # also tag existing PubMed chunks

Wikipedia briefly filled this role and was dropped once StatPearls landed: every
source in the corpus is now peer-reviewed, which is the whole point on a medical
corpus. (See git history for the wiki.py fetcher if that layer is ever wanted
back for breadth.)

Why the SAME collection as the papers rather than a separate one? Because the
whole point is that one question should reach whichever source actually answers
it: "what is a total knee arthroplasty?" should surface the reference chapter,
"what's the DVT risk after one?" should surface the papers. Two collections would
force the caller to decide which to search BEFORE knowing the answer — exactly
the guess we want retrieval to make for us. The `source_type` tag keeps them
distinguishable ("reference" vs "research") for labeling and filtering.

Access labels are stamped at ingest (same scheme as the papers, keyed on the
chapter id instead of a PMID), so background chunks are access-controlled
identically — a reference chapter can't sneak past a clearance filter.
"""
import argparse

import chromadb

import access
import config
import statpearls
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
        # Classification keys on a "statpearls-<id>" string rather than the bare
        # id: that id space is disjoint from PMIDs, so a chapter can't inherit a
        # paper's level by numeric collision.
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
    ap = argparse.ArgumentParser(description="Ingest the StatPearls background corpus.")
    ap.add_argument("--limit", type=int, default=None, help="Use only the first N chapters.")
    ap.add_argument("--stamp-research", action="store_true",
                    help="Also tag existing PubMed chunks source_type='research'.")
    args = ap.parse_args()

    ids, texts, metadatas = statpearls_chunks(args.limit)
    if not texts:
        raise SystemExit("Nothing to ingest — run `python statpearls.py` first.")
    print(f"StatPearls -> {len(texts):,} chunks from {len(statpearls.load(args.limit)):,} chapters")

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
