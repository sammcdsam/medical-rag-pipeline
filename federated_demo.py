"""Demonstrate FEDERATED retrieval across siloed sources with two-level access.

Same query, three principals. Shows silo-level authorization (an under-cleared
user's query never touches a restricted silo), the document-level pre-filter
still applying within each silo, and the merged, provenance-tagged result.

    python federated_demo.py
    python federated_demo.py "your query" -k 6

Prereq: python build_silos.py  (partitions the index into per-silo collections).
"""
import argparse
from collections import Counter

import access
import config
import federated

DEFAULT_Q = "surgical management of bone sarcoma and musculoskeletal oncology"


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated, access-gated retrieval across silos.")
    parser.add_argument("question", nargs="?", default=DEFAULT_Q)
    parser.add_argument("-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    print(f"Query: {args.question!r}   (federated top-{args.k})\n")

    for user in access.USERS.values():
        hits, report = federated.federated_retrieve(args.question, user, k=args.k)
        print(f"── {user.describe()}")

        queried = ", ".join(f"{r['label']} ({r['returned']} hits)" for r in report["queried"]) or "(none)"
        skipped = ", ".join(f"{r['label']} [needs {r['min_clearance']}]" for r in report["skipped"]) or "(none)"
        print(f"   silos queried : {queried}")
        print(f"   silos SKIPPED : {skipped}")

        for i, (text, meta, dist) in enumerate(hits):
            silo = federated.SILOS[meta["silo"]]["label"]
            level = access.LEVEL_NAME.get(meta.get("classification"), "?")
            print(f"     [{i}] {silo:<24} PMID={meta.get('pmid')}  {level}/{meta.get('compartment')}  d={dist:.3f}")

        mix = Counter(federated.SILOS[m["silo"]]["label"] for _t, m, _d in hits)
        print(f"   merged result drew from: {dict(mix)}\n")

    print("── Note " + "─" * 50)
    print("public (UNCLASSIFIED) can't even query the CONFIDENTIAL/SECRET silos — those")
    print("are skipped at the silo level. Within the silos it CAN query, the same")
    print("document pre-filter still applies. Results are merged by cosine distance")
    print("(all silos share one embedder) and tagged with their source silo.")


if __name__ == "__main__":
    main()
