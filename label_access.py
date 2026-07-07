"""One-time migration: stamp synthetic access labels onto every chunk.

Adds two metadata fields to each chunk already in the index:
  - `classification` (0-3)  — from access.classify(pmid)
  - `compartment` (subtopic) — from the corpus cache, keyed by PMID

It uses Chroma's update() (metadata only), so nothing is re-embedded — this runs
in seconds, not the minutes a full re-ingest would take. Idempotent: re-running
rewrites the same deterministic labels.

    python label_access.py

Why a separate script? The access labels are a concern layered ON TOP of the
corpus, not part of building it — keeping them here means ingest stays about
"get the knowledge in" and this stays about "who may see what".
"""
import chromadb

import access
import config
import pubmed


def pmid_to_compartment() -> dict:
    """Map PMID -> orthopedic subtopic (the compartment) from the local cache."""
    mapping = {}
    for d in pubmed.load_corpus():
        if d.get("pmid"):
            mapping[d["pmid"]] = d.get("subtopic") or "(untagged)"
    return mapping


def main() -> None:
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    col = client.get_collection(config.COLLECTION_ORTHO)
    comp = pmid_to_compartment()

    total = col.count()
    print(f"Labeling {total} chunks in '{config.COLLECTION_ORTHO}' (metadata-only, no re-embed)…")

    BATCH = 2000
    done, offset = 0, 0
    dist = {0: 0, 1: 0, 2: 0, 3: 0}
    while offset < total:
        # Page through by offset — update() doesn't add/remove ids, so paging is stable.
        got = col.get(limit=BATCH, offset=offset, include=["metadatas"])
        ids = got["ids"]
        if not ids:
            break
        new_metas = []
        for meta in got["metadatas"]:
            pmid = meta.get("pmid", "")
            level = access.classify(pmid)
            merged = dict(meta)  # keep pmid/title/journal/year; add access fields
            merged["classification"] = level
            merged["compartment"] = comp.get(pmid, "(untagged)")
            new_metas.append(merged)
            dist[level] += 1
        col.update(ids=ids, metadatas=new_metas)
        offset += len(ids)
        done += len(ids)
        print(f"  {done}/{total}")

    print("\nDone. Synthetic classification distribution:")
    for lvl in sorted(dist):
        pct = 100 * dist[lvl] / done if done else 0
        print(f"  {access.LEVEL_NAME[lvl]:>13}: {dist[lvl]:>6}  ({pct:.0f}%)")


if __name__ == "__main__":
    main()
