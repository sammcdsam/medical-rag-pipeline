"""Federated retrieval across independent, access-gated silos.

Adelphi-style "federated discovery across silos": the corpus is partitioned into
N independent Chroma collections, each standing in for a separate source/org that
doesn't share its raw index. A query fans out across the silos the user is CLEARED
to query, each silo is searched independently (with the same per-document access
pre-filter as single-index retrieval), and the ranked lists are MERGED into one
result tagged with provenance.

Two levels of access:
  - SILO-level authorization: each silo has a minimum clearance to query it at all;
    an under-cleared user's query never touches it (skipped, and reported).
  - DOCUMENT-level pre-filter: within a queried silo, access.build_where(user)
    still gates individual chunks (classification + need-to-know).

Merge note: every silo uses the SAME embedder here, so cosine distances are
directly comparable and we merge by distance. A real heterogeneous federation
(a different embedder per silo) would need score normalisation or a cross-encoder
rerank to merge — a natural extension, and a good thing to be able to name.
"""
import hashlib

import chromadb

import access
import config
from embedder import embed_query

# Independent silos. Each becomes its own Chroma collection; `min_clearance` is the
# silo-level authorization gate — you must be cleared at least this high to query it.
SILOS = {
    "mercy_general": {"label": "Mercy General Hospital",  "min_clearance": access.CLEARANCE["UNCLASSIFIED"]},
    "univ_biomech":  {"label": "University Biomech Lab",   "min_clearance": access.CLEARANCE["UNCLASSIFIED"]},
    "va_network":    {"label": "Veterans Health Network",  "min_clearance": access.CLEARANCE["CONFIDENTIAL"]},
    "dod_research":  {"label": "DoD Orthopedic Research",  "min_clearance": access.CLEARANCE["SECRET"]},
}


def collection_name(silo: str) -> str:
    return f"{config.COLLECTION_ORTHO}__{silo}"


def assign_silo(pmid: str) -> str:
    """Deterministically route a PMID to a silo (stable md5, so build and query agree)."""
    names = list(SILOS)
    idx = int(hashlib.md5(("silo:" + str(pmid)).encode()).hexdigest(), 16) % len(names)
    return names[idx]


def authorized_silos(user: "access.User") -> list[str]:
    """The silos this user is cleared to QUERY (silo-level authorization)."""
    return [s for s, meta in SILOS.items() if user.clearance >= meta["min_clearance"]]


_client = None


def _collection(silo: str):
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return _client.get_collection(collection_name(silo))


def federated_retrieve(question: str, user: "access.User", k: int = config.TOP_K):
    """Fan out across authorized silos, apply the per-doc pre-filter, merge by distance.

    Returns (hits, report):
      hits   — up to k (text, meta, distance) tuples, meta['silo'] set, nearest first
      report — {"queried": [...], "skipped": [...]} describing silo-level authorization
    """
    q_emb = embed_query(question)
    where = access.build_where(user)          # document-level pre-filter (same as single-index)
    allowed = authorized_silos(user)
    skipped = [s for s in SILOS if s not in allowed]

    all_hits, per_silo_count = [], {}
    for silo in allowed:
        res = _collection(silo).query(
            query_embeddings=[q_emb], n_results=k, where=where,
            include=["documents", "metadatas", "distances"],
        )
        docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
        per_silo_count[silo] = len(docs)
        for t, m, d in zip(docs, metas, dists):
            m = dict(m)
            m["silo"] = silo
            all_hits.append((t, m, d))

    all_hits.sort(key=lambda h: h[2])          # merge: nearest across all silos
    merged = all_hits[:k]
    report = {
        "queried": [
            {"silo": s, "label": SILOS[s]["label"], "returned": per_silo_count.get(s, 0)}
            for s in allowed
        ],
        "skipped": [
            {"silo": s, "label": SILOS[s]["label"],
             "min_clearance": access.LEVEL_NAME[SILOS[s]["min_clearance"]]}
            for s in skipped
        ],
    }
    return merged, report
