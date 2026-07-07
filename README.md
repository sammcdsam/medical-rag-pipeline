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
| `query.py` | Question → retrieve (± rerank) → generate with citations → print answer + sources. |
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
python eval_ortho.py --n 60 --hard --compare --faithfulness

# 4. Web demo.
python server.py                   # http://localhost:8022
```

## Evaluation — and an honest result

There are no gold labels on a live PubMed pull, so the eval **manufactures** a
signal: sample an abstract → have Claude write a question it answers → retrieve
against the whole corpus → check whether the source abstract's PMID comes back
(Hit@k, MRR). `--faithfulness` additionally runs the full RAG answer and has an
LLM judge score how well-grounded it is.

Two design choices make this more than a vanity metric:

- **`--hard` mode** generates *paraphrased* questions that strip the abstract's
  exact device names, cohort sizes, and numbers — removing the verbatim-overlap
  crutch. Hit@5 drops from ~90% (easy) to ~65–70% (hard): a real difficulty test.
- **`--compare` runs a *paired* A/B** — the same questions retrieved with and
  without the reranker — so any difference is the reranker's true effect, not
  question-sampling noise.

The interesting finding: on hard questions the reranker is **neutral on
source-recall** (Hit@5 ≈ unchanged) but **improves answer grounding**
(faithfulness ~88% → ~90%). Why? On a dense corpus where *many* abstracts answer
a paraphrased question, "did the one source PMID come back" is the wrong metric
for a reranker — it reshuffles *among* relevant docs, which shows up in answer
quality, not single-source recall. Catching that is the whole point of building
the eval before trusting the feature.

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
