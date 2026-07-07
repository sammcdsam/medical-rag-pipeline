"""Unit tests for the access-control model (access.py) — no DB, no network."""
import access


def test_classify_is_deterministic_and_in_range():
    for pmid in ["12345", "999", "abc", "42024037"]:
        assert access.classify(pmid) == access.classify(pmid)   # stable
        assert 0 <= access.classify(pmid) <= access.MAX_LEVEL


def test_classify_distribution_skews_unclassified():
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for i in range(5000):
        counts[access.classify(str(i))] += 1
    # Designed ~60/25/12/3 — assert the monotonic shape and a loose top-bucket band.
    assert counts[0] > counts[1] > counts[2] > counts[3]
    assert 0.50 < counts[0] / 5000 < 0.70


def test_build_where_public_is_clearance_ceiling_only():
    assert access.build_where(access.USERS["public"]) == {"classification": {"$lte": 0}}


def test_build_where_director_is_none():
    # Fully cleared for everything -> no predicate needed.
    assert access.build_where(access.USERS["director"]) is None


def test_build_where_clinician_combines_clearance_and_need_to_know():
    where = access.build_where(access.USERS["clinician"])
    assert "$and" in where
    clauses = where["$and"]
    assert {"classification": {"$lte": access.CLEARANCE["SECRET"]}} in clauses
    comp = next(c for c in clauses if "compartment" in c)["compartment"]["$in"]
    # Clinician lacks oncology + infection need-to-know; has arthroplasty.
    assert "oncology" not in comp and "infection" not in comp
    assert "arthroplasty" in comp


def test_clearance_name_roundtrip():
    assert access.USERS["director"].clearance_name == "TOP_SECRET"
    assert access.USERS["public"].clearance_name == "UNCLASSIFIED"
