# Medical RAG pipeline — live PubMed, two-stage retrieval, air-gap ready

A Retrieval-Augmented Generation system over a **real, live corpus of orthopedic
surgery abstracts pulled from PubMed** — built to be read end to end. No
LangChain, no cloud vector DB, no hidden magic: every moving part is a small
Python file you can open.

It is deliberately shaped toward **forward-deployed / offline** use: the same
pipeline runs against the frontier Claude API *or* a fully local model with no
internet, so it can operate in an air-gapped environment.

```
                    ┌── stage 1: bi-encoder (bge-small) ──▶ top-30 candidates
question ──▶ embed ─┤                                              │
                    └── stage 2: cross-encoder rerank ────────────▶ top-5
                                                                   │
                                        ┌──────────────────────────┘
                                        ▼
                    Claude Haiku (native citations)  ── or ──  local Llama (Ollama, $0, offline)
                                        │
                                        ▼
                              answer + cited sources (real PMIDs)
```

## What this demonstrates

A compact but complete picture of production-minded, security-aware RAG — each
capability is a small, readable module, not a framework call:

| Capability | What it shows | Where |
|---|---|---|
| **Research agent** | Claude drives a multi-hop loop over five tools (abstract search, **full-text deep-read**, federated, + the two graph tools), synthesizing a cited review; clearance bound at the **tool boundary** so it can't escalate — even under prompt injection | `agent.py` |
| **Citation graph** | Directed citation network from the full-text references — finds **foundational works** and traces lineage; the graph-structure query vector search can't do | `citation_graph.py` |
| **MCP server** | The access-bound retriever exposed over the Model Context Protocol — any MCP client (Claude Desktop/Code, Cursor) can call it; principal bound at launch | `mcp_server.py` |
| **Access-controlled retrieval** | Clearance + need-to-know enforced as a vector-store **pre-filter** — unauthorized docs never reach the model | `access.py`, `query.py` |
| **Federated multi-silo retrieval** | Fan-out across independent, separately-governed silos with **two-level access** + a distance-merged, provenance-tagged result | `federated.py`, `build_silos.py` |
| **Air-gap / offline capable** | One interface, frontier Claude API **or** a fully-local model — identical pipeline, no internet | `llm.py` |
| **Rigorous evaluation** | Label-free synthetic eval, **paired A/B**, Hit@k · MRR · Precision@k · nDCG@k · LLM-judge faithfulness — used to *reject* a reranker that didn't earn its place | `eval_ortho.py` |
| **Real provenance & citations** | Native character-span citations pointing at genuine PMIDs | `query.py` |
| **Auditability** | Append-only log of who retrieved what, when, under which filter | `audit.py` |
| **Two-stage retrieval** | Bi-encoder recall → cross-encoder rerank (measured, optional) | `reranker.py` |
| **Full-text ingestion** | PMC open-access JATS parsed into **section-aware chunks** with references captured (citation-graph-ready); ~50× more text/paper than abstracts | `pmc.py`, `ingest_fulltext.py` |
| **Live data pipeline** | 27K abstracts pulled from NCBI PubMed, cached for offline/reproducible ingest | `pubmed.py`, `download_corpus.py` |

Everything runs locally on a single GPU; the only external call is the optional
Claude API for generation. See the sections below for the how and why of each.

## What's in the corpus

Live-fetched from NCBI PubMed via the Entrez API, cached locally so ingest is
offline and reproducible:

- **27,445 unique abstracts → 34,469 chunks**, spanning 2021–2026, 2,200+ journals.
- Balanced across **10 orthopedic subtopics** (arthroplasty, spine, sports/knee,
  trauma, shoulder/elbow, hand/wrist, foot/ankle, pediatric, oncology, infection).
  PubMed's `efetch` caps a single query at ~10k records, so the corpus is built by
  partitioning into subtopic queries under the cap and unioning them (deduped by PMID).
- Every chunk keeps its real **PMID + article title** as provenance, so citations
  point at genuine papers.

## Full text (PMC open access)

Abstracts are the default corpus; `pmc.py` additionally pulls **full text** for the
open-access subset from PubMed Central: `PMID → PMCID` (NCBI ID Converter) →
`efetch db=pmc` → JATS XML → **section-aware chunks** (a chunk never crosses a
section boundary; each carries its heading) plus the article's **reference list
with PMIDs** — the raw material for a citation graph. Only OA articles are fetched
(non-OA return no `<body>`), and the text cache is git-ignored.

Access labels are inherited from the PMID hash, so a paper's full text is
access-controlled identically to its abstract.

```bash
python pmc.py --target 500      # cache OA full text (PMID→PMCID→JATS)
python ingest_fulltext.py       # section-aware chunk → embed → orthopedic_fulltext
python query.py "..." --fulltext --user clinician    # retrieve from full text
```

A 500-article pull yields ~9.8k section-tagged chunks; ~93% of their ~17.7k
references carry a PMID (the citation graph is a natural next step).

## The pieces

| File | Role |
|------|------|
| `config.py` | Every knob: models, chunk size, top-k, rerank pool, PubMed queries, paths. |
| `embedder.py` | Loads `BAAI/bge-small-en-v1.5` once; `embed_documents` / `embed_query`. |
| `reranker.py` | **Stage 2.** Cross-encoder (`BAAI/bge-reranker-base`) reranks the candidate pool. |
| `pubmed.py` / `download_corpus.py` | Fetch orthopedic abstracts from NCBI Entrez → cache to JSONL. |
| `ingest_pubmed.py` | Chunk → embed locally (GPU) → store in Chroma. Reads the local cache offline. |
| `pmc.py` | Fetch PMC **open-access full text** (JATS) → sections + references; PMID→PMCID via idconv. |
| `ingest_fulltext.py` | Section-aware chunk full text → embed → `orthopedic_fulltext` (access labels inherited). |
| `citation_graph.py` | Citation network (NetworkX) from the full-text references — foundational-paper ranking + lineage. |
| `query.py` | Question → retrieve (± rerank, ± access filter) → generate with citations → print answer + sources. |
| `access.py` | Access-control model — clearance + need-to-know compartments; builds the retrieval pre-filter. |
| `label_access.py` | One-time migration: stamp (synthetic) `classification` + `compartment` onto every chunk. |
| `access_demo.py` | Runs one query as three principals — shows correctly-scoped retrieval + leak prevention. |
| `audit.py` | Append-only JSONL log: who retrieved what, when, under which access filter. |
| `federated.py` | Federated retrieval across access-gated silos — silo-level authz, fan-out, merge. |
| `build_silos.py` | Partition the index into independent per-silo collections (vectors copied, no re-embed). |
| `federated_demo.py` | Same query, three principals — shows silo skipping + merged, provenance-tagged results. |
| `agent.py` | **Agentic RAG** — Claude drives its own retrieval loop; clearance bound at the tool boundary. |
| `mcp_server.py` | **MCP server** — exposes the access-bound retriever as tools any MCP client can call. |
| `llm.py` | Pluggable generation: Claude API (native span citations) **or** local Ollama model. |
| `eval_ortho.py` | Synthetic, label-free retrieval eval — Hit@k, MRR, faithfulness, paired A/B. |
| `server.py` | FastAPI web demo: query UI (model / rerank / role / federated), live agent runner, and `/agent` `/security` `/corpus` `/eval` explainer pages. |
| `tests/` | pytest suite — access/federated/eval unit tests + security-invariant integration tests. |
| `ingest.py` / `eval.py` | The original PubMedQA baseline (kept for reference). |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env   # git-ignored; loaded automatically
# Optional: NCBI_API_KEY=... in .env raises the PubMed rate limit 3→10 req/s.
```

## Run

```bash
# 1. Build the corpus cache from PubMed (once), then ingest into Chroma.
python download_corpus.py          # live PubMed pull → ortho_corpus.jsonl
python ingest_pubmed.py            # chunk → embed (local GPU) → Chroma

# 2. Ask a question. Add --rerank for the two-stage retriever; --model local for offline.
python query.py "What are the risk factors for infection after a knee replacement?" --rerank
python query.py "..." --model local      # $0, no internet — the air-gap path

# 3. (Optional) enable the security features on the index.
python label_access.py             # stamp synthetic classification + compartment
python build_silos.py              # partition into independent per-silo collections

# 4. Ask a question — as a plain pipeline, a specific principal, or an agent.
python query.py "..." --user clinician             # access-controlled retrieval
python query.py "..." --user public --federated    # federated across silos
python agent.py "Compare infection risk after knee vs spinal fusion" --user director

# 5. Evaluate the retriever (no gold labels needed — see below).
python eval_ortho.py --n 60 --hard --compare --relevance --faithfulness

# 6. Web demo (query UI + agent runner + explainer pages), and the MCP server.
python server.py                                   # http://localhost:8022
RAG_MCP_USER=clinician python mcp_server.py        # expose the retriever over MCP
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest                             # 23 tests, ~7s
```

The suite is unit tests for the access / federated / eval logic plus
**security-invariant** checks — the ones worth reading:

- **The pre-filter actually blocks** (integration, against the real index): a
  `public` retrieval returns *only* UNCLASSIFIED docs; a `clinician` never gets an
  oncology or above-clearance doc; a `public` federated query skips the classified
  silos entirely. (These skip automatically if the corpus/silos aren't built.)
- **The model can't set its own access**: the agent and MCP tool schemas expose
  *only* `query` + `k` — no `user`/`clearance`/`classification` — and the MCP
  server fails closed to the least-privileged `public` when no principal is set.
- **nDCG sees ordering, Precision doesn't**: the same relevant set in a worse order
  scores identical Precision@k but lower nDCG@k — the property behind the reranker
  A/B.

## Evaluation — and an honest result

There are no gold labels on a live PubMed pull, so the eval **manufactures** a
signal: sample an abstract → have Claude write a question it answers → retrieve
against the whole corpus → check whether the source abstract's PMID comes back
(Hit@k, MRR). `--faithfulness` additionally runs the full RAG answer and has an
LLM judge score how well-grounded it is.

Three design choices make this more than a vanity metric:

- **`--hard` mode** generates *paraphrased* questions that strip the abstract's
  exact device names, cohort sizes, and numbers — removing the verbatim-overlap
  crutch. Hit@5 drops from ~90% (easy) to ~65–70% (hard): a real difficulty test.
- **`--compare` runs a *paired* A/B** — the same questions retrieved with and
  without the reranker — so any difference is the reranker's true effect, not
  question-sampling noise.
- **`--relevance` grades *every* retrieved chunk** (LLM-judge, 0/1/2) → Precision@k
  and **nDCG@k**, a multi-relevant metric — because on a paraphrased question many
  abstracts are relevant, not just the source PMID.

The honest finding (paired, n=60, hard questions): the cross-encoder reranker
does **not** beat the bge-small bi-encoder on this corpus. Across every metric
family it's roughly neutral, and marginally *negative* on most — Hit@5 65→68,
MRR 0.57→0.51, Precision@5 75→73, nDCG@5 0.89→0.86, faithfulness 89→89. The
bi-encoder already reaches nDCG@5 ≈ 0.89, so there's little headroom, and a
general-purpose reranker adds noise on short, self-contained abstracts.

**The value here isn't the reranker — it's the eval rigorous enough to prove the
reranker isn't worth shipping on this corpus.** (An earlier n=8 spot-check looked
like a big nDCG win; it was small-sample noise, which is exactly why the real run
is n=60 with a paired design. Knowing when *not* to add a component is as much
the job as adding one.)

## Access-controlled retrieval

Retrieval can be gated by *who is asking* — the core of any secure data-sharing
system. Two orthogonal axes (the classic Bell–LaPadula shape):

- **Clearance** (hierarchical): `UNCLASSIFIED < CONFIDENTIAL < SECRET < TOP_SECRET`.
  You see a doc only if its `classification ≤ your clearance`.
- **Compartments** (need-to-know, non-hierarchical): the orthopedic subtopics act
  as compartments. Even with high clearance, no need-to-know for a topic means no
  access to it.

The labels are **synthetic** (PubMed is public): `classification` is a stable
hash of the PMID, `compartment` is the abstract's real subtopic — a stand-in so
the mechanism is demonstrable. `label_access.py` stamps them onto the index once.

The key property is that enforcement is a **pre-filter**: `access.build_where(user)`
becomes a vector-store `where` predicate applied *during* the search, so
unauthorized chunks never enter the candidate set — they can't leak into the
retrieved context, the citations, or the generated answer, and the reranker never
sees them either. (Post-filtering *after* retrieval is both a leak risk — the
model already read the text — and silently shrinks k.)

```bash
python label_access.py     # once: stamp synthetic classification + compartment
python access_demo.py      # same query as public / clinician / director
python query.py "..." --user clinician    # retrieve as a specific principal
```

`access_demo.py` runs one oncology query as three principals and shows the
outcome: the **clinician** (SECRET clearance but no oncology need-to-know) gets
**zero** oncology documents, while the **director** retrieves `SECRET`- and
`TOP_SECRET`-classified papers that the others provably cannot. Every retrieval
is written to `audit_log.jsonl` (who saw what, when, under which filter).

## Federated retrieval across silos

Real secure-sharing systems don't hold one index — they discover across many
independent, separately-governed sources. `build_silos.py` partitions the corpus
into **N independent Chroma collections** (genuinely separate indexes, as if each
lived at a different org), each with its own **minimum clearance to query it**:

| Silo | Min clearance to query |
|---|---|
| Mercy General Hospital | UNCLASSIFIED |
| University Biomech Lab | UNCLASSIFIED |
| Veterans Health Network | CONFIDENTIAL |
| DoD Orthopedic Research | SECRET |

`federated.federated_retrieve(question, user)` then does the distributed-retrieval
dance — **two levels of access**:

1. **Silo-level authorization** — skip silos the user isn't cleared to query at all
   (an UNCLASSIFIED user's query never even touches the SECRET silo).
2. **Document-level pre-filter** — within each queried silo, the same
   clearance/need-to-know `where` filter still gates individual chunks.
3. **Merge** — the per-silo ranked lists are merged by cosine distance into one
   result, each hit tagged with its **source silo** for provenance.

```bash
python build_silos.py       # once: partition into per-silo collections
python federated_demo.py    # same query as public / clinician / director
python query.py "..." --user public --federated
```

Run `federated_demo.py`: **public** (UNCLASSIFIED) can only query 2 of the 4
silos — the CONFIDENTIAL and SECRET silos are skipped at the silo level — while
**director** fans out across all four and the merged result pulls a `SECRET` paper
from the DoD silo that no one else can reach.

> **Merge caveat (worth naming):** every silo here shares one embedder, so cosine
> distances are directly comparable and a distance-sort merge is valid. A real
> heterogeneous federation (a different embedder per silo) would need score
> normalisation or a cross-encoder rerank to merge fairly.

## Agentic retrieval — a research agent

`query.py` runs a fixed pipeline (embed → retrieve → answer, once). `agent.py`
hands Claude the retrieval tools and lets it drive its own loop — search, read,
refine, search again, then answer. It has three tools:

- **`search_corpus`** — broad search over abstracts (widest coverage)
- **`search_fulltext`** — deep-read the **full text** of papers (methods, results,
  effect sizes — the specifics abstracts omit)
- **`federated_search`** — search across the access-gated silos and merge
- **`find_influential_papers`** / **`trace_citations`** — the **citation graph**:
  foundational works and a paper's lineage (see below)

That makes it a small **research agent**: it searches broadly, deep-reads the most
relevant papers, reformulates, synthesizes across sources (noting where studies
agree or disagree), and writes a **cited literature review**. It runs a multi-step
loop (bounded, with a forced-synthesis final turn) — so a question like *"what's
the evidence that tranexamic acid reduces blood loss in total hip arthroplasty,
with effect sizes?"* comes back as a structured review with real numbers
(e.g. "~150 mL reduction", "10 RCTs, 1,295 patients") deep-read from the full text
and cited by PMID. It also handles multi-hop comparisons, query reformulation,
and honest abstention.

The security-critical design: **the user's clearance is bound in the harness, not
exposed as a tool parameter.** Claude only ever sees `search(query, k)`; every
call is executed as `retrieve(query, user=<the bound principal>)` with the access
pre-filter applied. So the model **cannot escalate its own privileges** — even a
prompt-injection hidden in a retrieved document can't change the access level,
because the boundary lives in the code, not in the model's reasoning. Every tool
call is written to the audit log.

```bash
python agent.py "Compare infection risk after knee replacement vs spinal fusion" --user clinician
python agent.py "..." --user director
```

Run it as different principals and inspect the "evidence pulled" list: as
`public`, *every* document the agent touches across all its hops is UNCLASSIFIED;
as `director`, it can reach the classified silos. The access ceiling holds through
the entire multi-step loop, not just the first call.

## Citation graph — hybrid retrieval

The vector store answers *"what's semantically similar?"*. A **citation graph**
answers a different question — *"how are papers connected, and what is the field
built on?"* — which similarity search structurally cannot. `citation_graph.py`
builds a directed network (NetworkX) from the reference lists captured in the full
text: **500 corpus papers → ~14.8k nodes, ~16.4k citation edges**.

Because a corpus's references point mostly at papers *outside* it, ranking by
in-degree surfaces the field's **foundational works** even when they aren't in the
corpus — e.g. it correctly ranks *"The operation of the century: total hip
replacement"* and *"The 2018 definition of periprosthetic hip and knee infection"*
at the top. The agent uses this to ground a review in seminal sources
(`find_influential_papers`) and to trace a finding's lineage (`trace_citations`).

```bash
python citation_graph.py       # build the graph, print the most-cited papers
```

> Graph tools return **bibliographic structure** (PMIDs, titles, citation counts),
> not document content — a deliberate metadata-vs-content boundary. Reading a
> paper's text still goes through the access-filtered search tools.

## MCP server — the retriever as a reusable tool

`agent.py` wires the retrieval tools into *our own* loop. `mcp_server.py` exposes
the same tools over the **Model Context Protocol** (the open standard for
connecting AI apps to tools), so *any* MCP client — Claude Desktop, Claude Code,
Cursor — can call our access-controlled orthopedic retriever without a bespoke
integration. Write the tool once; every MCP-compatible agent can use it.

It advertises three tools — `search_corpus`, `federated_search`, and
`access_context` — and runs over stdio:

```bash
RAG_MCP_USER=clinician python mcp_server.py
```

Register it with an MCP client (e.g. Claude Desktop's `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "orthopedic-rag": {
      "command": "/abs/path/.venv/bin/python",
      "args": ["/abs/path/mcp_server.py"],
      "env": { "RAG_MCP_USER": "clinician" }
    }
  }
}
```

**Same security boundary as the agent, one layer out:** the principal is bound at
*server launch* via `RAG_MCP_USER` — it is **not** a tool parameter. The client's
model only sees `search_corpus(query, k)`, so it cannot set or raise its own
clearance; it defaults to the least-privileged `public` if unset (fail closed).
Every tool call is audited. Same query, different bound principal → different
authorized results (`public` gets only UNCLASSIFIED; `director` reaches the
classified silos).

## Things worth understanding

- **Two-stage retrieval.** The bi-encoder embeds query and document *separately*
  (fast, searches all 34k chunks); the cross-encoder reads `(question, chunk)`
  *together* (accurate, but only affordable on the top-30 survivors).
- **Air-gap by design.** `llm.py` hides the generator behind one interface, so
  the identical pipeline runs on Claude or a local Ollama model with no code
  change and no network — the story for classified / offline deployments.
- **Query prefix.** bge models want an instruction prefix on the *query* only
  (`config.QUERY_PREFIX`); documents get none. Skipping it quietly hurts recall.
- **Beating the efetch cap.** PubMed refuses to page past ~10k records per query;
  subtopic partitioning + union is how the corpus gets to 27k without gaps.
- **Real citations.** Each retrieved chunk is a `document` block with
  `citations.enabled=true`; Claude returns text blocks whose `.citations` point at
  the exact source chunk and character span it used.

## Stack

Python · [sentence-transformers](https://www.sbert.net/) (bge-small + bge-reranker) ·
[ChromaDB](https://www.trychroma.com/) · [Anthropic Claude](https://docs.claude.com/) ·
[Ollama](https://ollama.com/) · FastAPI · local GPU embeddings.
