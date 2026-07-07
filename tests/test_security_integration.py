"""Integration tests: the access pre-filter actually blocks against the real index.

These need the labeled corpus (label_access.py) and, for the federated ones, the
silo collections (build_silos.py). If those aren't built, the tests SKIP rather
than fail — so the suite still runs on a fresh clone.
"""
import pytest

import access
import config
import federated
from query import get_collection, retrieve

ONCOLOGY_QUERY = "surgical management of bone sarcoma and musculoskeletal oncology"


def _labeled_collection():
    """The ortho collection, but only if it exists AND has access labels."""
    try:
        col = get_collection(config.COLLECTION_ORTHO)
    except SystemExit:
        return None
    got = col.get(limit=1, include=["metadatas"])
    if not got["ids"] or "classification" not in (got["metadatas"][0] or {}):
        return None
    return col


def _silos_built():
    try:
        return federated._collection("mercy_general").count() > 0
    except Exception:
        return False


_COL = _labeled_collection()
needs_labels = pytest.mark.skipif(_COL is None, reason="labeled corpus not built (run label_access.py)")
needs_silos = pytest.mark.skipif(not _silos_built(), reason="silos not built (run build_silos.py)")


@needs_labels
def test_public_retrieval_returns_only_unclassified():
    hits = retrieve(_COL, ONCOLOGY_QUERY, k=8, where=access.build_where(access.USERS["public"]))
    assert hits, "expected some UNCLASSIFIED results"
    for _t, meta, _d in hits:
        assert meta.get("classification") == 0   # UNCLASSIFIED only, no leak


@needs_labels
def test_clinician_never_sees_oncology_or_above_clearance():
    # Clinician: SECRET clearance, but no oncology/infection need-to-know.
    hits = retrieve(_COL, ONCOLOGY_QUERY, k=8, where=access.build_where(access.USERS["clinician"]))
    for _t, meta, _d in hits:
        assert meta.get("compartment") not in ("oncology", "infection")
        assert meta.get("classification", 0) <= access.CLEARANCE["SECRET"]


@needs_silos
def test_federated_public_skips_classified_silos_and_leaks_nothing():
    hits, report = federated.federated_retrieve(ONCOLOGY_QUERY, access.USERS["public"], k=6)
    skipped = {r["silo"] for r in report["skipped"]}
    assert {"va_network", "dod_research"} <= skipped     # silo-level authorization
    for _t, meta, _d in hits:
        assert meta.get("classification") == 0           # doc-level pre-filter still holds
