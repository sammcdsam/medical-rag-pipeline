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

## The pieces

| File | Role |
|------|------|
| `config.py` | Every knob: models, chunk size, top-k, rerank pool, PubMed queries, paths. |
| `embedder.py` | Loads `BAAI/bge-small-en-v1.5` once; `embed_documents` / `embed_query`. |
| `reranker.py` | **Stage 2.** Cross-encoder (`BAAI/bge-reranker-base`) reranks the candidate pool. |
| `pubmed.py` / `download_corpus.py` | Fetch orthopedic abstracts from NCBI Entrez → cache to JSONL. |
| `ingest_pubmed.py` | Chunk → embed locally (GPU) → store in Chroma. Reads the local cache offline. |
| `query.py` | Question → retrieve (± rerank, ± access filter) → generate with citations → print answer + sources. |
| `access.py` | Access-control model — clearance + need-to-know compartments; builds the retrieval pre-filter. |
| `label_access.py` | One-time migration: stamp (synthetic) `classification` + `compartment` onto every chunk. |
| `access_demo.py` | Runs one query as three principals — shows correctly-scoped retrieval + leak prevention. |
| `audit.py` | Append-only JSONL log: who retrieved what, when, under which access filter. |
| `federated.py` | Federated retrieval across access-gated silos — silo-level authz, fan-out, merge. |
| `build_silos.py` | Partition the index into independent per-silo collections (vectors copied, no re-embed). |
| `federated_demo.py` | Same query, three principals — shows silo skipping + merged, provenance-tagged results. |
| `llm.py` | Pluggable generation: Claude API (native span citations) **or** local Ollama model. |
| `eval_ortho.py` | Synthetic, label-free retrieval eval — Hit@k, MRR, faithfulness, paired A/B. |
| `server.py` | FastAPI web demo: query UI, model dropdown, corpus + eval explainer pages. |
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

# 3. Evaluate the retriever (no gold labels needed — see below).
python eval_ortho.py --n 60 --hard --compare --relevance --faithfulness

# 4. Web demo.
python server.py                   # http://localhost:8022
```

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
