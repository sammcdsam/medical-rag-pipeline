"""Evaluate the orthopedic corpus — which has NO gold labels.

The PubMedQA baseline shipped with expert yes/no/maybe answers; a live PubMed
pull does not. So we manufacture a gold signal instead of hand-labelling:

  SYNTHETIC RETRIEVAL EVAL (self-supervised, isolates the retriever)
    1. Sample N abstracts from the corpus.
    2. Ask Claude to write one specific question that abstract answers.
    3. Retrieve top-k against the WHOLE corpus.
    4. Check whether the abstract's own PMID comes back.
    Metrics: Hit@k, MRR (mean reciprocal rank), Recall@k.

  FAITHFULNESS / GROUNDEDNESS (optional, --faithfulness; LLM-as-judge)
    For each question, run the full RAG answer and have Claude score what
    fraction of the answer is supported by the retrieved context (0-100).

Results are written to eval_results.json so the /eval page can display them.

    python eval_ortho.py --n 40
    python eval_ortho.py --n 20 --faithfulness

Cost: one Haiku call per sample to generate the question (+2 more per sample
with --faithfulness). N=40 is a few cents.

CAVEAT (stated honestly on the /eval page): a generated question may be
answerable by OTHER abstracts too, so Hit@k on the *source* PMID slightly
under-counts a genuinely good retriever. It's a lower bound, not ground truth.
"""
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from anthropic import Anthropic

import config
import pubmed
from query import get_collection, retrieve

RESULTS_PATH = Path(__file__).parent / "eval_results.json"


def sample_abstracts(n: int, seed: int = 0) -> list[dict]:
    """Sample n abstracts (one per PMID) from the LOCAL corpus cache.

    We sample from the cache rather than dumping the whole collection out of
    Chroma (a full .get() trips a SQL-variable limit on large corpora).
    Retrieval still runs against the Chroma index — the cache is just the
    source of abstracts to generate questions from.
    """
    docs = pubmed.load_corpus()
    pool = [{"pmid": d["pmid"], "title": d.get("title", ""), "text": d["text"]}
            for d in docs if d.get("pmid")]
    random.Random(seed).shuffle(pool)
    return pool[:min(n, len(pool))]


EASY_PROMPT = (
    "Read this abstract and write ONE specific question a clinician "
    "might ask that it answers. Output only the question, no preamble.\n\n"
    "{abstract}"
)

# Hard mode strips the verbatim-overlap crutch. The easy prompt tends to echo the
# abstract's exact device names, cohort sizes and numbers, so the bi-encoder wins
# on surface vocabulary alone. Here we force a paraphrased, general question —
# closer to how a real clinician asks, and the regime where a reranker earns its
# keep (it can match meaning even when the words don't line up).
HARD_PROMPT = (
    "Read this abstract and write ONE general clinical question that it answers.\n"
    "Constraints:\n"
    "- Do NOT reuse the abstract's specific terminology: no device/product names, "
    "no exact cohort sizes, no numeric results, no study-specific acronyms.\n"
    "- Phrase it the way a busy clinician would ask a colleague, in plain words.\n"
    "- It must still be answerable specifically by this abstract.\n"
    "Output only the question, no preamble.\n\n"
    "{abstract}"
)


def generate_question(client: Anthropic, abstract: str, hard: bool = False) -> str:
    """Ask Claude for one question this abstract answers.

    hard=True paraphrases away the abstract's specific terms (see HARD_PROMPT),
    which is a much stiffer test of retrieval than the term-matching easy mode.
    """
    prompt = (HARD_PROMPT if hard else EASY_PROMPT).format(abstract=abstract)
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def judge_faithfulness(client: Anthropic, question: str, context: str, answer: str) -> int:
    """LLM-as-judge: what % of the answer is supported by the context (0-100)?"""
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=8,
        messages=[{
            "role": "user",
            "content": (
                f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:\n{answer}\n\n"
                "What percentage of the Answer's claims are directly supported by the "
                "Context? Reply with a single integer 0-100 and nothing else."
            ),
        }],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    digits = "".join(c for c in text if c.isdigit())
    return max(0, min(100, int(digits))) if digits else 0


def _rank(results, pmid: str) -> int:
    """1-based rank of pmid in the retrieved results, or 0 if absent."""
    retrieved = [m.get("pmid") for _t, m, _d in results]
    return retrieved.index(pmid) + 1 if pmid in retrieved else 0


def _metrics(ranks: list[int]) -> dict:
    """Hit@k / MRR from a list of 1-based ranks (0 = miss)."""
    n = len(ranks)
    hits = sum(1 for r in ranks if r > 0)
    rr = sum((1.0 / r) for r in ranks if r > 0)
    return {"hit_at_k": round(hits / n, 3), "mrr": round(rr / n, 3)}


def _faithfulness(client, q: str, results) -> int:
    """Full RAG answer over `results`, then LLM-judge % grounded."""
    from query import answer as rag_answer
    resp = rag_answer(q, results)
    ans = "".join(b.text for b in resp.content if b.type == "text")
    ctx = "\n\n".join(t for t, _m, _d in results)
    return judge_faithfulness(client, q, ctx, ans)


def run_compare(collection, client, samples, k: int, hard: bool, faithfulness: bool = False) -> dict:
    """PAIRED A/B: one question per sample, retrieved BOTH ways.

    Because the question set is identical for rerank-off and rerank-on, any
    metric difference is the reranker's true effect — not question-sampling
    noise. This is the honest way to decide whether stage 2 earns its place.

    With faithfulness=True we also generate the full RAG answer under each
    retrieval config and judge how grounded it is — the answer-quality metric
    where a reranker (which reshuffles AMONG relevant docs) should actually pay
    off, unlike single-source Hit@k.
    """
    print(f"PAIRED comparison: same questions, retrieval with vs without rerank")
    if faithfulness:
        print("(+ faithfulness: full answer judged under each config)")
    print(f"\n{'#':>3}  {'off':>3}  {'on':>3}  question")
    print("-" * 80)
    ranks_off, ranks_on = [], []
    faith_off, faith_on = [], []
    for i, s in enumerate(samples):
        q = generate_question(client, s["text"], hard=hard)
        res_off = retrieve(collection, q, k=k, rerank=False)
        res_on = retrieve(collection, q, k=k, rerank=True)
        ranks_off.append(_rank(res_off, s["pmid"]))
        ranks_on.append(_rank(res_on, s["pmid"]))
        if faithfulness:
            faith_off.append(_faithfulness(client, q, res_off))
            faith_on.append(_faithfulness(client, q, res_on))
        print(f"{i:>3}  {ranks_off[-1] or '-':>3}  {ranks_on[-1] or '-':>3}  {q[:58]}")

    out = {"off": _metrics(ranks_off), "on": _metrics(ranks_on)}
    if faithfulness:
        out["off"]["faithfulness"] = round(sum(faith_off) / len(faith_off), 1)
        out["on"]["faithfulness"] = round(sum(faith_on) / len(faith_on), 1)
    return out


def run_single(collection, client, samples, k: int, hard: bool, rerank: bool, faithfulness: bool) -> dict:
    """Single-config eval (one retrieval setting), optionally with faithfulness."""
    print(f"{'#':>3}  {'hit':>3}  {'rank':>4}  question")
    print("-" * 80)
    ranks, faith_scores = [], []
    for i, s in enumerate(samples):
        q = generate_question(client, s["text"], hard=hard)
        results = retrieve(collection, q, k=k, rerank=rerank)
        rank = _rank(results, s["pmid"])
        ranks.append(rank)

        if faithfulness:
            from query import answer as rag_answer
            resp = rag_answer(q, results)
            ans = "".join(b.text for b in resp.content if b.type == "text")
            ctx = "\n\n".join(t for t, _m, _d in results)
            faith_scores.append(judge_faithfulness(client, q, ctx, ans))

        print(f"{i:>3}  {'yes' if rank else 'no':>3}  {rank or '-':>4}  {q[:60]}")

    m = _metrics(ranks)
    m["recall_at_k"] = m["hit_at_k"]   # one relevant doc per query => Recall@k == Hit@k
    m["faithfulness"] = round(sum(faith_scores) / len(faith_scores), 1) if faith_scores else None
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic retrieval eval for the orthopedic corpus.")
    parser.add_argument("--n", type=int, default=40, help="Number of abstracts to sample.")
    parser.add_argument("-k", type=int, default=config.TOP_K, help="top-k for retrieval.")
    parser.add_argument("--faithfulness", action="store_true", help="Also run the LLM-judge groundedness pass.")
    parser.add_argument("--hard", action="store_true",
                        help="Generate paraphrased questions that avoid the abstract's exact terms (a stiffer test).")
    parser.add_argument("--rerank", action="store_true",
                        help="Add the cross-encoder rerank stage to retrieval (see reranker.py).")
    parser.add_argument("--compare", action="store_true",
                        help="Paired A/B: evaluate the SAME questions with and without rerank (isolates its effect).")
    args = parser.parse_args()

    collection = get_collection(config.COLLECTION_ORTHO)
    client = Anthropic()
    samples = sample_abstracts(args.n)
    n = len(samples)

    mode = "HARD (paraphrased)" if args.hard else "easy (specific)"
    print(f"Synthetic retrieval eval: {n} questions, top-{args.k}, "
          f"corpus '{config.COLLECTION_ORTHO}' ({collection.count()} chunks)")
    print(f"question mode: {mode}\n")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": config.COLLECTION_ORTHO,
        "corpus_chunks": collection.count(),
        "model": config.CLAUDE_MODEL,
        "embedder": config.EMBED_MODEL,
        "n": n,
        "k": args.k,
        "question_mode": "hard" if args.hard else "easy",
    }

    if args.compare:
        ab = run_compare(collection, client, samples, args.k, args.hard, args.faithfulness)
        summary["compare"] = ab
        print("-" * 80)
        print(f"{'':>22}{'rerank OFF':>12}{'rerank ON':>12}")
        print(f"{'Hit@'+str(args.k):>22}{ab['off']['hit_at_k']:>12.0%}{ab['on']['hit_at_k']:>12.0%}")
        print(f"{'MRR':>22}{ab['off']['mrr']:>12.3f}{ab['on']['mrr']:>12.3f}")
        if args.faithfulness:
            print(f"{'Faithfulness':>22}{ab['off']['faithfulness']:>11.1f}%{ab['on']['faithfulness']:>11.1f}%")
    else:
        m = run_single(collection, client, samples, args.k, args.hard, args.rerank, args.faithfulness)
        summary.update({"rerank": args.rerank, **m})
        print("-" * 80)
        print(f"Hit@{args.k}   (source abstract in top-{args.k}): {m['hit_at_k']:.0%}")
        print(f"MRR        (mean reciprocal rank):        {m['mrr']:.3f}")
        if m["faithfulness"] is not None:
            print(f"Faithfulness (LLM-judge, mean % grounded): {m['faithfulness']:.0f}%")

    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {RESULTS_PATH.name} (the /eval page reads this).")


if __name__ == "__main__":
    main()
