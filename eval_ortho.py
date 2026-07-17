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
import math
import random
import re
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


def _parse_grades(text: str, k: int) -> list[int]:
    """Pull k graded-relevance integers out of the judge's reply, robustly.

    We ask for a JSON array like [2,0,1,...]; parse that if we can, else fall
    back to the first k standalone digits. Clamp to 0-2 and pad short replies
    with 0 (treat a missing grade as 'not judged relevant')."""
    match = re.search(r"\[[^\]]*\]", text)
    grades: list[int] = []
    if match:
        try:
            grades = [int(x) for x in json.loads(match.group(0)) if isinstance(x, (int, float))]
        except Exception:
            grades = []
    if not grades:
        grades = [int(c) for c in re.findall(r"[0-2]", text)]
    grades = [min(2, max(0, g)) for g in grades[:k]]
    return grades + [0] * (k - len(grades))


def judge_relevance(client: Anthropic, question: str, results) -> list[int]:
    """LLM-as-judge graded relevance for EACH retrieved chunk, in ONE call.

    Unlike Hit@k (which only checks the single source PMID), this grades every
    passage — 0 irrelevant / 1 partially / 2 directly answers — so the metric
    credits surfacing ANY genuinely relevant abstract. Batching all k passages
    into one judge call keeps the cost at 1 call per (question, config), not k."""
    passages = "\n\n".join(f"[{i}] {t[:600]}" for i, (t, _m, _d) in enumerate(results))
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=120,
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n\nPassages:\n{passages}\n\n"
                "Grade EACH passage's relevance to the question: "
                "0 = irrelevant, 1 = partially relevant, 2 = directly answers it. "
                f"Reply with ONLY a JSON array of {len(results)} integers in passage "
                "order, e.g. [2,0,1,2,0]. No prose."
            ),
        }],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return _parse_grades(text, len(results))


def _graded_metrics(grades: list[int]) -> dict:
    """Precision@k and nDCG@k from graded relevance (0/1/2) in retrieved order.

    Precision@k = fraction of the k that are relevant (grade >= 1).
    nDCG@k = graded ranking quality: gain 2^rel - 1 (so a '2' is worth much more
    than a '1'), position-discounted by log2, normalised by the ideal ordering.
    nDCG rewards putting the MOST relevant passages first — exactly what a
    reranker is for, and what single-source Hit@k can't see."""
    k = len(grades)
    if k == 0:
        return {"precision_at_k": 0.0, "ndcg_at_k": 0.0}
    gain = lambda g: (2 ** g) - 1
    dcg = sum(gain(g) / math.log2(i + 2) for i, g in enumerate(grades))
    idcg = sum(gain(g) / math.log2(i + 2) for i, g in enumerate(sorted(grades, reverse=True)))
    precision = sum(1 for g in grades if g >= 1) / k
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    return {"precision_at_k": round(precision, 3), "ndcg_at_k": round(ndcg, 3)}


def _mean_graded(grade_lists: list[list[int]]) -> dict:
    """Average Precision@k / nDCG@k across a set of per-question grade lists."""
    per = [_graded_metrics(g) for g in grade_lists]
    n = len(per) or 1
    return {
        "precision_at_k": round(sum(p["precision_at_k"] for p in per) / n, 3),
        "ndcg_at_k": round(sum(p["ndcg_at_k"] for p in per) / n, 3),
    }


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


def _faithfulness(client, q: str, results, backend: str = "claude",
                  model: str | None = None) -> int:
    """Full RAG answer over `results`, then LLM-judge % grounded.

    The ANSWER can come from any backend (Claude, or a local Ollama model —
    that's the point: it's how you measure whether a local model earns its place
    on the air-gap path). The JUDGE is always Claude: grading grounding is the
    harder task, and judging a model's output with itself would flatter it.
    """
    import llm
    ans = llm.generate(q, results, backend=backend, model=model)["answer"]
    ctx = "\n\n".join(t for t, _m, _d in results)
    return judge_faithfulness(client, q, ctx, ans)


def run_compare(collection, client, samples, k: int, hard: bool,
                faithfulness: bool = False, relevance: bool = False) -> dict:
    """PAIRED A/B: one question per sample, retrieved BOTH ways.

    Because the question set is identical for rerank-off and rerank-on, any
    metric difference is the reranker's true effect — not question-sampling
    noise. This is the honest way to decide whether stage 2 earns its place.

    With faithfulness=True we also generate the full RAG answer under each
    retrieval config and judge how grounded it is. With relevance=True we grade
    every retrieved chunk (Precision@k / nDCG@k) — the multi-relevant metric that
    credits surfacing ANY good abstract, so it can finally SEE a reranker that
    reshuffles among relevant docs, unlike single-source Hit@k.
    """
    print(f"PAIRED comparison: same questions, retrieval with vs without rerank")
    if faithfulness:
        print("(+ faithfulness: full answer judged under each config)")
    if relevance:
        print("(+ relevance: every chunk graded 0/1/2 -> Precision@k, nDCG@k)")
    print(f"\n{'#':>3}  {'off':>3}  {'on':>3}  question")
    print("-" * 80)
    ranks_off, ranks_on = [], []
    faith_off, faith_on = [], []
    grades_off, grades_on = [], []
    for i, s in enumerate(samples):
        q = generate_question(client, s["text"], hard=hard)
        res_off = retrieve(collection, q, k=k, rerank=False)
        res_on = retrieve(collection, q, k=k, rerank=True)
        ranks_off.append(_rank(res_off, s["pmid"]))
        ranks_on.append(_rank(res_on, s["pmid"]))
        if faithfulness:
            faith_off.append(_faithfulness(client, q, res_off))
            faith_on.append(_faithfulness(client, q, res_on))
        if relevance:
            grades_off.append(judge_relevance(client, q, res_off))
            grades_on.append(judge_relevance(client, q, res_on))
        print(f"{i:>3}  {ranks_off[-1] or '-':>3}  {ranks_on[-1] or '-':>3}  {q[:58]}")

    out = {"off": _metrics(ranks_off), "on": _metrics(ranks_on)}
    if faithfulness:
        out["off"]["faithfulness"] = round(sum(faith_off) / len(faith_off), 1)
        out["on"]["faithfulness"] = round(sum(faith_on) / len(faith_on), 1)
    if relevance:
        out["off"].update(_mean_graded(grades_off))
        out["on"].update(_mean_graded(grades_on))
    return out


def run_compare_models(collection, client, samples, k: int, hard: bool, rerank: bool,
                       models: list[str], backend: str = "local") -> dict:
    """PAIRED A/B across ANSWER MODELS: same questions, same retrieved context.

    Why paired, in one process: generate_question() calls Claude at default
    temperature, so two separate runs get two different question sets — and a
    faithfulness gap between them would be question-sampling noise, not the
    model. (That's the exact error the early unpaired reranker eval made, which
    reversed once it was paired.) Here every model answers the IDENTICAL
    questions over the IDENTICAL retrieved chunks, so any delta is the model.

    The judge is Claude for all of them — one fixed yardstick.
    """
    # Guard against the silent-mislabel trap: an ollama tag ("llama3.1:8b") sent
    # to the claude backend used to run Claude and *label the column* with the
    # ollama name — producing a confident Claude-vs-Claude comparison. Fail loudly.
    if backend == "claude" and any(":" in m and not m.startswith("claude") for m in models):
        raise SystemExit(
            f"--compare-models {models} looks like local model tags but "
            "--answer-backend is 'claude'. Pass --answer-backend local, or the run "
            "would compare Claude with itself and label the columns with these names."
        )
    print(f"PAIRED model comparison (backend={backend}): {', '.join(models)}")
    print("Same questions, same retrieval — only the answering model changes.\n")
    scores = {m: [] for m in models}
    for i, s in enumerate(samples):
        q = generate_question(client, s["text"], hard=hard)
        results = retrieve(collection, q, k=k, rerank=rerank)   # retrieve ONCE
        row = []
        for m in models:
            sc = _faithfulness(client, q, results, backend, m)
            scores[m].append(sc)
            row.append(f"{m.split(':')[0][:12]}={sc:>3}")
        print(f"{i:>3}  {'  '.join(row)}  {q[:44]}")

    out = {}
    for m, vals in scores.items():
        mean = sum(vals) / len(vals)
        # Report the spread too: an LLM judge on n=30 is noisy, and a 2-point
        # gap inside a 30-point spread is not a finding.
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        out[m] = {"faithfulness": round(mean, 1), "stdev": round(var ** 0.5, 1), "n": len(vals)}
    return out


def run_single(collection, client, samples, k: int, hard: bool, rerank: bool,
               faithfulness: bool, relevance: bool = False,
               answer_backend: str = "claude", answer_model: str | None = None) -> dict:
    """Single-config eval (one retrieval setting), optional faithfulness/relevance."""
    print(f"{'#':>3}  {'hit':>3}  {'rank':>4}  question")
    print("-" * 80)
    ranks, faith_scores, grade_lists = [], [], []
    for i, s in enumerate(samples):
        q = generate_question(client, s["text"], hard=hard)
        results = retrieve(collection, q, k=k, rerank=rerank)
        rank = _rank(results, s["pmid"])
        ranks.append(rank)

        if faithfulness:
            faith_scores.append(_faithfulness(client, q, results, answer_backend, answer_model))
        if relevance:
            grade_lists.append(judge_relevance(client, q, results))

        print(f"{i:>3}  {'yes' if rank else 'no':>3}  {rank or '-':>4}  {q[:60]}")

    m = _metrics(ranks)
    m["recall_at_k"] = m["hit_at_k"]   # one relevant doc per query => Recall@k == Hit@k
    m["faithfulness"] = round(sum(faith_scores) / len(faith_scores), 1) if faith_scores else None
    if relevance:
        m.update(_mean_graded(grade_lists))
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic retrieval eval for the orthopedic corpus.")
    parser.add_argument("--n", type=int, default=40, help="Number of abstracts to sample.")
    parser.add_argument("-k", type=int, default=config.TOP_K, help="top-k for retrieval.")
    parser.add_argument("--faithfulness", action="store_true", help="Also run the LLM-judge groundedness pass.")
    parser.add_argument("--relevance", action="store_true",
                        help="Grade every retrieved chunk (0/1/2) -> Precision@k, nDCG@k (multi-relevant metric).")
    parser.add_argument("--hard", action="store_true",
                        help="Generate paraphrased questions that avoid the abstract's exact terms (a stiffer test).")
    parser.add_argument("--rerank", action="store_true",
                        help="Add the cross-encoder rerank stage to retrieval (see reranker.py).")
    parser.add_argument("--compare", action="store_true",
                        help="Paired A/B: evaluate the SAME questions with and without rerank (isolates its effect).")
    parser.add_argument("--answer-backend", choices=["claude", "local"], default="claude",
                        help="Which backend WRITES the answers judged by --faithfulness (judge is always Claude).")
    parser.add_argument("--answer-model", default=None,
                        help="Override the model for --answer-backend, e.g. 'llama3.1:8b'.")
    parser.add_argument("--compare-models", default=None,
                        help="Paired A/B of answer models on IDENTICAL questions+retrieval, "
                             "judged for faithfulness. Comma-separated, e.g. "
                             "'llama3.1:8b,mistral-small:24b'.")
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
        "model": config.CLAUDE_MODEL,          # the JUDGE / question generator
        "embedder": config.EMBED_MODEL,
        "n": n,
        "k": args.k,
        "question_mode": "hard" if args.hard else "easy",
        # Which model actually WROTE the answers being judged for grounding.
        "answer_backend": args.answer_backend,
        "answer_model": args.answer_model or (
            config.LOCAL_MODEL if args.answer_backend == "local" else config.CLAUDE_MODEL),
    }

    if args.compare_models:
        models = [m.strip() for m in args.compare_models.split(",") if m.strip()]
        mm = run_compare_models(collection, client, samples, args.k, args.hard,
                                args.rerank, models, args.answer_backend)
        summary["compare_models"] = mm
        summary["rerank"] = args.rerank
        print("-" * 80)
        print(f"{'answer model':>24}{'faithfulness':>14}{'stdev':>8}")
        for m, v in mm.items():
            print(f"{m:>24}{v['faithfulness']:>13.1f}%{v['stdev']:>8.1f}")
    elif args.compare:
        ab = run_compare(collection, client, samples, args.k, args.hard, args.faithfulness, args.relevance)
        summary["compare"] = ab
        print("-" * 80)
        print(f"{'':>22}{'rerank OFF':>12}{'rerank ON':>12}")
        print(f"{'Hit@'+str(args.k):>22}{ab['off']['hit_at_k']:>12.0%}{ab['on']['hit_at_k']:>12.0%}")
        print(f"{'MRR':>22}{ab['off']['mrr']:>12.3f}{ab['on']['mrr']:>12.3f}")
        if args.relevance:
            print(f"{'Precision@'+str(args.k):>22}{ab['off']['precision_at_k']:>12.0%}{ab['on']['precision_at_k']:>12.0%}")
            print(f"{'nDCG@'+str(args.k):>22}{ab['off']['ndcg_at_k']:>12.3f}{ab['on']['ndcg_at_k']:>12.3f}")
        if args.faithfulness:
            print(f"{'Faithfulness':>22}{ab['off']['faithfulness']:>11.1f}%{ab['on']['faithfulness']:>11.1f}%")
    else:
        m = run_single(collection, client, samples, args.k, args.hard, args.rerank,
                       args.faithfulness, args.relevance,
                       args.answer_backend, args.answer_model)
        summary.update({"rerank": args.rerank, **m})
        print("-" * 80)
        print(f"Hit@{args.k}   (source abstract in top-{args.k}): {m['hit_at_k']:.0%}")
        print(f"MRR        (mean reciprocal rank):        {m['mrr']:.3f}")
        if args.relevance:
            print(f"Precision@{args.k} (graded, mean relevant):   {m['precision_at_k']:.0%}")
            print(f"nDCG@{args.k}      (graded ranking quality):  {m['ndcg_at_k']:.3f}")
        if m["faithfulness"] is not None:
            print(f"Faithfulness (LLM-judge, mean % grounded): {m['faithfulness']:.0f}%")

    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {RESULTS_PATH.name} (the /eval page reads this).")


if __name__ == "__main__":
    main()
