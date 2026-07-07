"""Append-only audit log of access-controlled retrievals.

The compliance/observability half of secure retrieval: every authorized query
writes one JSON line recording WHO asked, WHEN, the query, the access predicate
that was applied, and WHICH documents (PMID + classification) were returned. You
can then prove after the fact exactly what each principal was shown — the kind of
accountability an air-gapped / classified deployment requires.

Append-only JSONL (never rewritten) so the trail is tamper-evident by
construction. Git-ignored — it's runtime data, not source.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import access

AUDIT_PATH = Path(__file__).parent / "audit_log.jsonl"


def record_retrieval(user: "access.User", question: str, hits, where, *, backend: str | None = None) -> dict:
    """Append one audit entry for an access-controlled retrieval; return it too."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user": user.name,
        "clearance": user.clearance_name,
        "query": question,
        "where": where,
        "backend": backend,
        "returned": [
            {
                "pmid": m.get("pmid"),
                "classification": access.LEVEL_NAME.get(m.get("classification"), "?"),
                "compartment": m.get("compartment"),
            }
            for _t, m, _d in hits
        ],
    }
    with AUDIT_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def tail(n: int = 10) -> list[dict]:
    """Return the last n audit entries (for display on the /audit page or CLI)."""
    if not AUDIT_PATH.exists():
        return []
    lines = [ln for ln in AUDIT_PATH.read_text().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines[-n:]]
