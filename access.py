"""Access-control model for retrieval — clearance + need-to-know compartments.

This mirrors the core problem of a secure data-sharing system (discover and
retrieve across silos WITHOUT letting a user see what they're not authorized to):
gate retrieval by *who is asking*. Two orthogonal axes, the classic
Bell-LaPadula shape:

  1. CLEARANCE (hierarchical): UNCLASSIFIED < CONFIDENTIAL < SECRET < TOP_SECRET.
     A user may see a document only if its classification <= the user's clearance.
  2. COMPARTMENTS (need-to-know, non-hierarchical): a set of tags. A user may see
     a document only if the document's compartment is one the user holds — even
     with high clearance, no need-to-know means no access.

Enforcement is a PRE-FILTER: build_where() turns a user into a vector-store
`where` predicate that is applied DURING search, so unauthorized chunks never
enter the candidate set. They cannot leak into the retrieved context, the
citations, or the generated answer. (Post-filtering after retrieval is both a
leak risk — the model already saw the text — and silently shrinks k.)

NOTE: PubMed abstracts are public. The labels here are SYNTHETIC: a stable,
deterministic stand-in for real classification so the mechanism is demonstrable.
`classification` is assigned by hashing the PMID; `compartment` is the abstract's
real orthopedic subtopic.
"""
import hashlib

import config

# Hierarchical clearance levels (higher int = more restricted).
CLEARANCE = {"UNCLASSIFIED": 0, "CONFIDENTIAL": 1, "SECRET": 2, "TOP_SECRET": 3}
LEVEL_NAME = {v: k for k, v in CLEARANCE.items()}
MAX_LEVEL = max(CLEARANCE.values())

# The orthopedic subtopics double as need-to-know compartments.
COMPARTMENTS = list(config.PUBMED_SUBTOPICS.keys())
ALL_COMPARTMENTS = set(COMPARTMENTS)


def classify(pmid: str) -> int:
    """Deterministically assign a synthetic classification level to a PMID.

    Uses md5 (stable across processes — unlike Python's salted hash()) so the
    same PMID always lands at the same level. Skewed toward UNCLASSIFIED like a
    real corpus: ~60% U / 25% C / 12% S / 3% TS.
    """
    bucket = int(hashlib.md5(str(pmid).encode()).hexdigest(), 16) % 100
    if bucket < 60:
        return 0
    if bucket < 85:
        return 1
    if bucket < 97:
        return 2
    return 3


class User:
    """A retrieval principal: a clearance level and a set of held compartments."""

    def __init__(self, name: str, clearance: int, compartments):
        self.name = name
        self.clearance = clearance
        self.compartments = set(compartments)

    @property
    def clearance_name(self) -> str:
        return LEVEL_NAME[self.clearance]

    def describe(self) -> str:
        comps = "ALL" if self.compartments == ALL_COMPARTMENTS else ", ".join(sorted(self.compartments))
        return f"{self.name}: clearance={self.clearance_name}, need-to-know={comps}"


# Demo principals — three roles chosen to make the security property visible.
USERS = {
    # Only unclassified material, but no topic restriction.
    "public":    User("public", CLEARANCE["UNCLASSIFIED"], ALL_COMPARTMENTS),
    # High clearance, but NO need-to-know for oncology or infection.
    "clinician": User("clinician", CLEARANCE["SECRET"], ALL_COMPARTMENTS - {"oncology", "infection"}),
    # Fully cleared for everything.
    "director":  User("director", CLEARANCE["TOP_SECRET"], ALL_COMPARTMENTS),
}


def build_where(user: User):
    """Translate a user's clearance + compartments into a Chroma `where` filter.

    Returns None when the user is cleared for everything (no predicate needed).
    Otherwise an `$and` of a clearance ceiling and, if restricted, a
    need-to-know membership test.
    """
    clauses = [{"classification": {"$lte": user.clearance}}]
    if user.compartments != ALL_COMPARTMENTS:
        clauses.append({"compartment": {"$in": sorted(user.compartments)}})

    fully_cleared = user.clearance >= MAX_LEVEL and user.compartments == ALL_COMPARTMENTS
    if fully_cleared:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
