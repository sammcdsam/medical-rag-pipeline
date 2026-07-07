"""Unit tests for federated silo logic (federated.py) — no DB, no network."""
import access
import federated


def test_assign_silo_is_deterministic_and_valid():
    for pmid in ["1", "2", "xyz", "42024037"]:
        assert federated.assign_silo(pmid) == federated.assign_silo(pmid)
        assert federated.assign_silo(pmid) in federated.SILOS


def test_public_can_only_query_unclassified_silos():
    # Silo-level authorization: UNCLASSIFIED clearance can't touch CONFIDENTIAL/SECRET silos.
    allowed = set(federated.authorized_silos(access.USERS["public"]))
    assert allowed == {"mercy_general", "univ_biomech"}


def test_secret_clearance_can_query_all_silos():
    # The most restrictive silo requires SECRET; clinician (SECRET) clears all four.
    for role in ("clinician", "director"):
        assert set(federated.authorized_silos(access.USERS[role])) == set(federated.SILOS)


def test_collection_name_is_namespaced():
    assert federated.collection_name("dod_research").endswith("__dod_research")
