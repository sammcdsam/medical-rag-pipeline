"""Partition the single orthopedic index into independent per-silo collections.

Copies each chunk (vector + document + metadata) from COLLECTION_ORTHO into its
assigned per-silo Chroma collection, tagging it with `silo`. Embeddings are copied
straight across — NO re-embedding. Drops and recreates the silo collections each
run, so it's safe to re-run after re-labeling.

    python build_silos.py

Prereq: the base index exists (ingest_pubmed.py) and is access-labeled
(label_access.py), so the copied metadata already carries classification +
compartment for the document-level pre-filter.
"""
import chromadb

import config
import federated


def main() -> None:
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    src = client.get_collection(config.COLLECTION_ORTHO)
    total = src.count()
    print(f"Partitioning {total} chunks from '{config.COLLECTION_ORTHO}' into "
          f"{len(federated.SILOS)} silos (vectors copied, no re-embed)…")

    # Fresh silo collections, cosine to match the source.
    silos = {}
    for s in federated.SILOS:
        name = federated.collection_name(s)
        try:
            client.delete_collection(name)
        except Exception:
            pass
        silos[s] = client.create_collection(name, metadata={"hnsw:space": "cosine"})

    buffers = {s: {"ids": [], "embeddings": [], "documents": [], "metadatas": []} for s in federated.SILOS}

    def flush(s):
        b = buffers[s]
        if b["ids"]:
            silos[s].add(ids=b["ids"], embeddings=b["embeddings"],
                         documents=b["documents"], metadatas=b["metadatas"])
            for key in b:
                b[key] = []

    BATCH, offset = 2000, 0
    while offset < total:
        got = src.get(limit=BATCH, offset=offset, include=["embeddings", "documents", "metadatas"])
        ids = got["ids"]
        if not ids:
            break
        for cid, emb, doc, meta in zip(ids, got["embeddings"], got["documents"], got["metadatas"]):
            s = federated.assign_silo(meta.get("pmid", ""))
            meta = dict(meta)
            meta["silo"] = s
            b = buffers[s]
            b["ids"].append(cid)
            b["embeddings"].append(emb.tolist() if hasattr(emb, "tolist") else emb)
            b["documents"].append(doc)
            b["metadatas"].append(meta)
            if len(b["ids"]) >= 1000:
                flush(s)
        offset += len(ids)
        print(f"  scanned {offset}/{total}")

    for s in federated.SILOS:
        flush(s)

    print("\nSilo sizes:")
    for s in federated.SILOS:
        meta = federated.SILOS[s]
        print(f"  {meta['label']:<28} min={access_name(meta['min_clearance']):<13} {silos[s].count():>7} chunks")


def access_name(level: int) -> str:
    import access
    return access.LEVEL_NAME[level]


if __name__ == "__main__":
    main()
