"""Agentic RAG — Claude decides when and how to retrieve, under access control.

Where query.py runs a FIXED pipeline (embed -> retrieve -> answer, once), this
gives Claude a set of tools and a goal and lets it drive its own loop: search,
read the results, refine or search a different sub-topic, then answer. That
enables multi-hop questions, query reformulation, and honest abstention — none of
which a single-shot pipeline can do.

THE SECURITY DESIGN (the important part): the user's clearance is bound HERE, in
the harness, when the agent is constructed — it is NOT a tool parameter Claude
controls. The tool schema Claude sees is just `search(query, k)`; every call is
executed as `retrieve(query, user=<the bound principal>)` with the access
pre-filter applied. So the model physically cannot escalate its own privileges,
even if a retrieved document contains a prompt-injection telling it to. The
access boundary lives in the code, not in the model's reasoning. Every tool call
is also written to the audit log.

    python agent.py "Compare infection risk after knee vs hip replacement" --user clinician
    python agent.py "..." --user director
"""
import argparse

from anthropic import Anthropic

import access
import audit
import config
import federated
from query import get_collection, retrieve, _provenance

SYSTEM = (
    "You are a careful clinical research assistant answering questions from an "
    "orthopedic literature corpus. You can only see what your search tools return, "
    "and those results are already restricted to what the current user is authorized "
    "to access — do not speculate about material outside them.\n\n"
    "Work by searching for evidence, reading it, and searching again with a refined "
    "query or a different sub-topic if needed (e.g. compare two procedures by searching "
    "each). When you have enough grounded evidence, answer concisely and cite the "
    "supporting PMIDs inline like (PMID 12345). If the authorized results do not contain "
    "the answer, say so plainly rather than guessing."
)

# The tool SCHEMAS Claude sees. Note there is NO `user`/clearance field — that is
# bound in the harness (see _run_tool), so the model can't set its own access level.
TOOLS = [
    {
        "name": "search_corpus",
        "description": (
            "Search the orthopedic literature for passages relevant to a query and return the "
            "best-matching abstract chunks (each with its PMID). Results are automatically "
            "restricted to what the current user may access. Call again with a refined query if "
            "the first results are off-target or you need a different sub-topic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused search query."},
                "k": {"type": "integer", "description": "How many passages to return (default 5, max 10)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "federated_search",
        "description": (
            "Like search_corpus, but fans the query out across every independent data silo the "
            "current user is cleared to query, then merges the results. Use for the broadest "
            "coverage across sources. Each result is tagged with the silo it came from."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused search query."},
                "k": {"type": "integer", "description": "How many merged passages to return (default 5, max 10)."},
            },
            "required": ["query"],
        },
    },
]


def _format_hits(hits) -> str:
    """Render retrieved chunks as compact, citeable text for the tool result."""
    if not hits:
        return "(no authorized results for this query)"
    lines = []
    for i, (text, meta, dist) in enumerate(hits):
        _kind, pmid = _provenance(meta)
        tags = []
        if "classification" in meta:
            tags.append(access.LEVEL_NAME.get(meta.get("classification"), "?"))
        if meta.get("compartment"):
            tags.append(meta["compartment"])
        if meta.get("silo"):
            tags.append(f"silo:{federated.SILOS[meta['silo']]['label']}")
        tag = f" [{', '.join(tags)}]" if tags else ""
        snippet = text[:500].replace("\n", " ")
        lines.append(f"Result {i + 1}: PMID {pmid}{tag} (dist {dist:.3f})\n{snippet}")
    return "\n\n".join(lines)


def _run_tool(name: str, tool_input: dict, user, collection):
    """Execute a tool with the user's access bound HERE — not passed by the model."""
    query = tool_input.get("query", "")
    k = min(int(tool_input.get("k", config.TOP_K)), 10)
    try:
        if name == "federated_search":
            hits, _report = federated.federated_retrieve(query, user, k=k)
        else:  # search_corpus (and anything unexpected → fail closed to a scoped search)
            hits = retrieve(collection, query, k=k, where=access.build_where(user))
    except Exception as e:
        return f"(search error: {e})", []
    audit.record_retrieval(user, query, hits, {"tool": name}, backend="agent")
    return _format_hits(hits), hits


def run(question: str, user, model: str = config.CLAUDE_MODEL, max_steps: int = 6, verbose: bool = True):
    """Drive the agentic loop until Claude answers or we hit max_steps."""
    client = Anthropic()
    collection = get_collection(config.COLLECTION_ORTHO)
    messages = [{"role": "user", "content": question}]
    all_hits = []

    for step in range(1, max_steps + 1):
        resp = client.messages.create(
            model=model, max_tokens=config.MAX_TOKENS,
            system=SYSTEM, tools=TOOLS, messages=messages,
        )

        # Show any narration the model wrote alongside its tool calls.
        if verbose:
            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    print(f"  · {block.text.strip()}")

        if resp.stop_reason in ("tool_use", "pause_turn"):
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if verbose:
                    print(f"  → {block.name}({block.input.get('query', '')!r})")
                text, hits = _run_tool(block.name, block.input, user, collection)
                all_hits.extend(hits)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": text})
            if results:
                messages.append({"role": "user", "content": results})
            continue

        # end_turn — the agent is done.
        answer = "".join(b.text for b in resp.content if b.type == "text")
        return {"answer": answer, "hits": all_hits, "steps": step}

    return {"answer": "(stopped: reached max_steps without a final answer)",
            "hits": all_hits, "steps": max_steps}


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic, access-controlled RAG over the orthopedic corpus.")
    parser.add_argument("question")
    parser.add_argument("--user", choices=tuple(access.USERS), default="director",
                        help="Run AS this principal — its clearance is bound to the tools (default: director).")
    parser.add_argument("--model", default=config.CLAUDE_MODEL, help="Claude model for the agent loop.")
    parser.add_argument("--max-steps", type=int, default=6, help="Max tool-use rounds before giving up.")
    args = parser.parse_args()

    user = access.USERS[args.user]
    print(f"=== AGENT ({args.model}) ===\n{user.describe()}\n")
    print(f"Q: {args.question}\n")
    print("--- trace ---")
    result = run(args.question, user, model=args.model, max_steps=args.max_steps)

    print(f"\n--- answer ({result['steps']} step{'s' if result['steps'] != 1 else ''}) ---")
    print(result["answer"])

    # Provenance: the distinct documents the agent actually pulled (deduped by PMID).
    seen, prov = set(), []
    for _text, meta, _dist in result["hits"]:
        _kind, pmid = _provenance(meta)
        if pmid in seen:
            continue
        seen.add(pmid)
        lvl = access.LEVEL_NAME.get(meta.get("classification"), "?") if "classification" in meta else "-"
        prov.append(f"  PMID {pmid}  [{lvl}/{meta.get('compartment', '?')}]")
    print(f"\n--- evidence pulled ({len(prov)} docs, all access-authorized) ---")
    print("\n".join(prov) if prov else "  (none)")


if __name__ == "__main__":
    main()
