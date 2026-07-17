"""Step 2 — answer a question with retrieval + Claude citations.

    python query.py "Does aspirin reduce the risk of a second heart attack?"

Flow: embed the question -> retrieve top-5 chunks from Chroma -> hand those
chunks to Claude Haiku as citeable `document` blocks -> print the answer plus
the exact spans the model cited.

Requires ANTHROPIC_API_KEY in the environment (the SDK reads it automatically).
"""
import argparse

import chromadb
from anthropic import Anthropic

import access
import config
from embedder import embed_query


def get_collection(name: str = config.COLLECTION_NAME):
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    # get_collection raises if it doesn't exist — a clear signal to run an ingest.
    try:
        return client.get_collection(name)
    except Exception:
        raise SystemExit(f"No collection '{name}'. Run an ingest first (e.g. `python ingest_pubmed.py`).")


def _provenance(meta: dict) -> tuple[str, str]:
    """Return (kind, id) for a chunk — PMID for the live-PubMed corpus, StatPearls
    for background chapters, pubid for the PubMedQA baseline. Keeps
    query/eval/server corpus-agnostic."""
    if meta.get("pmid"):
        return "PMID", meta["pmid"]
    if meta.get("chapter_id"):
        return "StatPearls", meta["chapter_id"]
    return "pubid", str(meta.get("pubid", "?"))


def _access_tag(meta: dict) -> str:
    """Human-readable access/provenance label for a chunk, if present (see access.py / federated.py)."""
    parts = []
    if "classification" in meta:
        parts.append(f"{access.LEVEL_NAME.get(meta.get('classification'), '?')}/{meta.get('compartment', '?')}")
    if meta.get("silo"):
        parts.append(f"silo={meta['silo']}")
    return "[" + " ".join(parts) + "]" if parts else ""


def retrieve(collection, question: str, k: int = config.TOP_K,
             rerank: bool = False, candidates: int = config.RERANK_CANDIDATES,
             where: dict | None = None):
    """Return a list of (chunk_text, metadata, distance), best first.

    With rerank=False this is a plain bi-encoder search for the top-k.
    With rerank=True we first pull a wider pool of `candidates` (default 30),
    then let the cross-encoder narrow it to the final k — see reranker.py. The
    single entry point means the CLI, server, and eval all get reranking for
    free by passing one flag.

    `where` is an access-control PRE-FILTER (see access.py): the vector store
    applies it DURING search, so chunks the user isn't authorized for never
    enter the candidate set — they can't leak into the context, citations, or
    answer, and the reranker never sees them either.
    """
    n = candidates if rerank else k
    q_emb = embed_query(question)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma returns a list-per-query; we only sent one query, so unwrap [0].
    hits = list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))
    if rerank:
        import reranker  # local import: only load the cross-encoder when asked
        hits = reranker.rerank(question, hits, top_k=k)
    return hits


def build_document_blocks(hits) -> list[dict]:
    """Turn retrieved chunks into Claude `document` content blocks.

    citations.enabled=true tells Claude it may cite these — the response then
    comes back split into text blocks, and any block grounded in a source
    carries a `.citations` list pointing at the exact chunk + character span.
    """
    blocks = []
    for i, (text, meta, _dist) in enumerate(hits):
        kind, ident = _provenance(meta)
        title = meta.get("title")  # real article title on the PubMed corpus
        heading = f"{title} ({kind} {ident})" if title else f"chunk {i} ({kind} {ident})"
        blocks.append(
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": text},
                "title": heading,
                "citations": {"enabled": True},
            }
        )
    return blocks


def answer(question: str, hits, model: str | None = None):
    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    # Content order: all the source documents first, then the question/instruction.
    content = build_document_blocks(hits) + [
        {
            "type": "text",
            "text": (
                "Answer the question using ONLY the documents above. "
                "Cite the document(s) supporting each claim. If the documents do "
                "not contain the answer, say so.\n\n"
                f"Question: {question}"
            ),
        }
    ]

    return client.messages.create(
        model=model or config.CLAUDE_MODEL,
        max_tokens=config.MAX_TOKENS,
        messages=[{"role": "user", "content": content}],
    )


def print_response(resp, hits) -> None:
    print("\n=== RETRIEVED CHUNKS ===")
    for i, (text, meta, dist) in enumerate(hits):
        preview = text[:120].replace("\n", " ")
        kind, ident = _provenance(meta)
        print(f"[{i}] {kind}={ident}  cos_dist={dist:.3f}  {preview}...")

    print("\n=== ANSWER ===")
    cited = []
    for block in resp.content:
        if block.type != "text":
            continue
        print(block.text, end="")
        # Blocks backed by a source carry citations; plain prose blocks don't.
        for c in block.citations or []:
            cited.append(c)
    print()

    print("\n=== CITED SOURCES ===")
    if not cited:
        print("(the model did not attach any citations)")
    for c in cited:
        snippet = c.cited_text.strip().replace("\n", " ")
        print(f'- {c.document_title}: "{snippet}"')


def print_result(result: dict, hits) -> None:
    """Backend-agnostic printing (Claude or local) for a normalized llm result."""
    print("\n=== RETRIEVED CHUNKS ===")
    for i, (text, meta, dist) in enumerate(hits):
        kind, ident = _provenance(meta)
        preview = text[:100].replace("\n", " ")
        tag = _access_tag(meta)
        sec = f" §{meta['section']}" if meta.get("section") else ""   # full-text: which section
        print(f"[{i}] {kind}={ident}  cos_dist={dist:.3f}  {tag}{sec}  {preview}...")

    print(f"\n=== ANSWER ({result['backend']} · {result['model']}) ===")
    print(result["answer"])

    print("\n=== CITED SOURCES ===")
    if result["citations"]:
        for c in result["citations"]:
            snippet = c["text"].strip().replace("\n", " ")
            print(f'- {c["title"]}: "{snippet}"')
    else:
        print("(no native citations — local model cites [n] inline; see chunks above for PMIDs)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the ingested corpus.")
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument("-k", type=int, default=config.TOP_K, help="Number of chunks to retrieve.")
    parser.add_argument("--model", choices=("claude", "local"), default="claude",
                        help="Generation backend: Claude API (native citations) or local Ollama model.")
    parser.add_argument("--rerank", action="store_true",
                        help="Add the cross-encoder rerank stage (retrieve a wider pool, then rerank to top-k).")
    parser.add_argument("--user", choices=tuple(access.USERS), default=None,
                        help="Retrieve AS this principal — access-control pre-filter by clearance + need-to-know.")
    parser.add_argument("--federated", action="store_true",
                        help="Fan the query out across access-gated silos and merge (see federated.py). Implies --user.")
    parser.add_argument("--fulltext", action="store_true",
                        help="Retrieve from the PMC full-text corpus instead of abstracts (section-tagged chunks).")
    args = parser.parse_args()

    import audit
    import llm  # local import avoids a circular import (llm imports query)

    user = access.USERS[args.user] if args.user else None

    if args.federated:
        import federated
        principal = user or access.USERS["director"]  # federation needs a principal; default fully-cleared
        print(f"\n=== FEDERATED RETRIEVAL ({principal.name}) ===")
        hits, report = federated.federated_retrieve(args.question, principal, k=args.k)
        for r in report["queried"]:
            print(f"  queried  {r['label']}: {r['returned']} hits")
        for r in report["skipped"]:
            print(f"  SKIPPED  {r['label']} (needs {r['min_clearance']})")
        audit.record_retrieval(principal, args.question, hits, {"federated": True}, backend=args.model)
    else:
        where = None
        if user:
            where = access.build_where(user)
            print(f"\n=== ACCESS CONTEXT ===\n{user.describe()}\nwhere-filter: {where}")
        collection = get_collection(config.COLLECTION_FULLTEXT if args.fulltext else config.COLLECTION_NAME)
        hits = retrieve(collection, args.question, k=args.k, rerank=args.rerank, where=where)
        if user:
            audit.record_retrieval(user, args.question, hits, where, backend=args.model)

    result = llm.generate(args.question, hits, backend=args.model)
    print_result(result, hits)


if __name__ == "__main__":
    main()
