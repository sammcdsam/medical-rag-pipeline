"""Demonstrate access-control-aware retrieval — the security property, made visible.

Runs the SAME query as three principals with different clearance + need-to-know,
and shows retrieval is correctly scoped to each. Enforcement is a PRE-FILTER in
the vector store, so unauthorized chunks never reach the LLM.

    python access_demo.py
    python access_demo.py "surgical management of bone sarcoma" -k 5

Prereq: run `python label_access.py` once to stamp the (synthetic) access labels.
"""
import argparse
from collections import Counter

import access
import audit
import config
from query import get_collection, retrieve, _provenance

DEFAULT_Q = "surgical management of bone sarcoma and musculoskeletal oncology"


def main() -> None:
    parser = argparse.ArgumentParser(description="Show access-controlled retrieval across principals.")
    parser.add_argument("question", nargs="?", default=DEFAULT_Q)
    parser.add_argument("-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    collection = get_collection(config.COLLECTION_ORTHO)
    print(f"Query: {args.question!r}   (top-{args.k}, corpus '{config.COLLECTION_ORTHO}')\n")

    seen_by: dict[str, set] = {}   # pmid -> which users could retrieve it
    for name, user in access.USERS.items():
        where = access.build_where(user)
        hits = retrieve(collection, args.question, k=args.k, where=where)
        audit.record_retrieval(user, args.question, hits, where)  # log every access

        print(f"── {user.describe()}")
        print(f"   where-filter: {where}")
        levels = Counter()
        for i, (text, meta, dist) in enumerate(hits):
            _, pmid = _provenance(meta)
            level = access.LEVEL_NAME.get(meta.get("classification"), "?")
            comp = meta.get("compartment", "?")
            levels[level] += 1
            seen_by.setdefault(pmid, set()).add(name)
            print(f"     [{i}] PMID={pmid}  {level}/{comp}  d={dist:.3f}  {text[:64].strip()}…")
        print(f"   classification mix: {dict(levels)}\n")

    # The security check: documents only the fully-cleared director could reach.
    director_only = sorted(p for p, users in seen_by.items() if users == {"director"})
    print("── Security check " + "─" * 50)
    print(f"Documents surfaced for the director but BLOCKED for public & clinician "
          f"on the same query: {len(director_only)}")
    for p in director_only:
        print(f"   • PMID {p}")
    print(
        "\nThese never entered the lower principals' candidate set — the filter runs\n"
        "DURING the vector search (pre-filter), so blocked chunks cannot appear in\n"
        "their retrieved context, citations, or generated answer. Every retrieval\n"
        f"above was written to {audit.AUDIT_PATH.name} (who saw what, when)."
    )


if __name__ == "__main__":
    main()
