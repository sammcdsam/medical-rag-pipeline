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
    "orthopedic literature corpus. Answer using ONLY what your search tools return — "
    "those results are already restricted to what the current user may access; do not "
    "speculate beyond them.\n\n"
    "Your tools:\n"
    "- search_corpus: broad search over ABSTRACTS (widest coverage of the literature).\n"
    "- search_fulltext: deep search into the FULL TEXT of papers (methods, results, "
    "effect sizes, specifics an abstract omits) — use it to read a promising paper closely.\n"
    "- federated_search: search across access-gated data silos and merge.\n"
    "- find_influential_papers: the most-cited (foundational) papers in the citation graph — "
    "ground a review in seminal works.\n"
    "- trace_citations: for a PMID, what it cites and which papers cite it — trace a finding's lineage.\n\n"
    "Work like a researcher: search broadly first, then deep-read the most relevant papers, "
    "reformulate or search a different sub-topic as needed, and synthesize across sources. "
    "When comparing or reviewing evidence, note where studies agree and where they disagree. "
    "Cite supporting PMIDs inline like (PMID 12345). If the authorized evidence does not "
    "answer the question, say so plainly rather than guessing."
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
        "name": "search_fulltext",
        "description": (
            "Search the FULL TEXT of papers (not just abstracts) for a query. Returns passages "
            "tagged with the section they came from (Introduction, Methods, Results, Discussion). "
            "Use this to read a paper closely — for methods, sample sizes, effect sizes, and other "
            "specifics abstracts leave out. Access-restricted like the other tools."
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
    {
        "name": "find_influential_papers",
        "description": (
            "Return the most-cited (foundational) papers in the corpus's CITATION GRAPH — the works "
            "the literature is built on, ranked by how many corpus papers cite them. Use to ground a "
            "review in seminal sources. This is graph structure, not semantic search; some results "
            "are foundational works outside the corpus (search_fulltext to read the ones inside it)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"k": {"type": "integer", "description": "How many top papers (default 12)."}},
        },
    },
    {
        "name": "trace_citations",
        "description": (
            "For a given PMID, return what it CITES and which corpus papers CITE it — the paper's "
            "place in the citation graph. Use to trace a finding's lineage or find related work."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"pmid": {"type": "string", "description": "The PMID to look up."}},
            "required": ["pmid"],
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
        if meta.get("section"):
            tags.append(f"§{meta['section']}")   # full-text: which section this came from
        tag = f" [{', '.join(tags)}]" if tags else ""
        snippet = text[:500].replace("\n", " ")
        lines.append(f"Result {i + 1}: PMID {pmid}{tag} (dist {dist:.3f})\n{snippet}")
    return "\n\n".join(lines)


_ft_collection = None


def _fulltext_collection():
    """Lazily open the full-text collection (only if the agent actually deep-reads)."""
    global _ft_collection
    if _ft_collection is None:
        _ft_collection = get_collection(config.COLLECTION_FULLTEXT)
    return _ft_collection


def _format_influential(papers) -> str:
    if not papers:
        return "(citation graph unavailable — is the full-text corpus ingested?)"
    lines = [f"  {p['citations']}x  PMID {p['pmid']} "
             f"{'[in corpus]' if p['in_corpus'] else '[external]'}  {p['title'][:80]}" for p in papers]
    return "Most-cited (foundational) papers in the citation graph:\n" + "\n".join(lines)


def _format_lineage(pmid: str, cites, cited_by) -> str:
    if not cites and not cited_by:
        return f"(PMID {pmid} is not in the citation graph.)"
    out = [f"Citation lineage for PMID {pmid}:", f"  CITES {len(cites)} papers:"]
    out += [f"    -> PMID {c['pmid']}  {c['title'][:70]}" for c in cites[:12]]
    out.append(f"  CITED BY {len(cited_by)} corpus papers:")
    out += [f"    <- PMID {c['pmid']}  {c['title'][:70]}" for c in cited_by[:12]]
    return "\n".join(out)


def _run_tool(name: str, tool_input: dict, user, collection):
    """Execute a tool with the user's access bound HERE — not passed by the model."""
    # Citation-graph tools return bibliographic STRUCTURE (PMIDs/titles/counts), not
    # document content — so no access pre-filter and no document hits here. The
    # sensitive full text still requires the access-filtered search tools to read.
    if name == "find_influential_papers":
        import citation_graph
        return _format_influential(citation_graph.influential(min(int(tool_input.get("k", 12)), 25))), []
    if name == "trace_citations":
        import citation_graph
        pmid = str(tool_input.get("pmid", "")).strip()
        return _format_lineage(pmid, citation_graph.cited_by(pmid), citation_graph.who_cites(pmid)), []

    query = tool_input.get("query", "")
    k = min(int(tool_input.get("k", config.TOP_K)), 10)
    try:
        if name == "federated_search":
            hits, _report = federated.federated_retrieve(query, user, k=k)
        elif name == "search_fulltext":
            hits = retrieve(_fulltext_collection(), query, k=k, where=access.build_where(user))
        else:  # search_corpus (and anything unexpected → fail closed to a scoped abstract search)
            hits = retrieve(collection, query, k=k, where=access.build_where(user))
    except Exception as e:
        return f"(search error: {e})", []
    audit.record_retrieval(user, query, hits, {"tool": name}, backend="agent")
    return _format_hits(hits), hits


def _level_mix(hits) -> dict:
    """Count of classification levels among a set of hits — for the trace summary."""
    mix = {}
    for _t, m, _d in hits:
        lvl = access.LEVEL_NAME.get(m.get("classification"), "?") if "classification" in m else "-"
        mix[lvl] = mix.get(lvl, 0) + 1
    return mix


AGENT_MAX_TOKENS = 2048   # roomy enough for a synthesized, multi-paper answer


def run(question: str, user, model: str = config.CLAUDE_MODEL, max_steps: int = 8, verbose: bool = True):
    """Drive the agentic loop until Claude answers or we hit max_steps.

    Returns {answer, hits, steps, trace}. `trace` is a structured list of what the
    agent did (its narration and each tool call), so a UI can render the loop.
    """
    client = Anthropic()
    collection = get_collection(config.COLLECTION_ORTHO)
    messages = [{"role": "user", "content": question}]
    all_hits, trace = [], []

    for step in range(1, max_steps + 1):
        resp = client.messages.create(
            model=model, max_tokens=AGENT_MAX_TOKENS,
            system=SYSTEM, tools=TOOLS, messages=messages,
        )

        for block in resp.content:
            if block.type == "text" and block.text.strip():
                trace.append({"type": "thought", "step": step, "text": block.text.strip()})
                if verbose:
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
                trace.append({"type": "tool", "step": step, "name": block.name,
                              "query": block.input.get("query", ""),
                              "returned": len(hits), "levels": _level_mix(hits)})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": text})
            if results:
                messages.append({"role": "user", "content": results})
            continue

        # end_turn — the agent is done.
        answer = "".join(b.text for b in resp.content if b.type == "text")
        return {"answer": answer, "hits": all_hits, "steps": step, "trace": trace}

    # Ran out of steps while still searching — force a final synthesis turn with no
    # tools available, so a thorough researcher still gets a written answer.
    if verbose:
        print("  · (step limit reached — forcing synthesis)")
    trace.append({"type": "thought", "step": max_steps, "text": "(step limit reached — synthesizing)"})
    messages.append({"role": "user", "content":
        "You have gathered enough evidence. Do not search further. Write your final answer "
        "now: synthesize what you found, note where studies agree or disagree, and cite PMIDs."})
    final = client.messages.create(model=model, max_tokens=AGENT_MAX_TOKENS,
                                   system=SYSTEM, messages=messages)   # no tools -> must answer
    answer = "".join(b.text for b in final.content if b.type == "text")
    return {"answer": answer, "hits": all_hits, "steps": max_steps, "trace": trace}


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic, access-controlled RAG over the orthopedic corpus.")
    parser.add_argument("question")
    parser.add_argument("--user", choices=tuple(access.USERS), default="director",
                        help="Run AS this principal — its clearance is bound to the tools (default: director).")
    parser.add_argument("--model", default=config.CLAUDE_MODEL, help="Claude model for the agent loop.")
    parser.add_argument("--max-steps", type=int, default=8, help="Max tool-use rounds before forced synthesis.")
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
