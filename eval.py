"""Step 3 — measure the pipeline on gold questions. (The part to learn from.)

    python eval.py --n 10

Two metrics, and they measure DIFFERENT things — that separation is the point:

  1. Retrieval hit-rate  — of the eval questions, how often did we retrieve at
     least one chunk from that question's OWN gold abstract (matched by pubid)?
     This isolates the retriever: no LLM involved, $0 in API cost. A low number
     here means fix embeddings/chunking/top-k, not the prompt.

  2. Decision accuracy   — we ask Claude for a yes/no/maybe answer from the
     retrieved chunks and compare to the gold `final_decision`. This is a rough
     end-to-end answer-quality proxy: it can only be right if retrieval worked
     AND the model reasoned correctly over what it got.

Note: eval reads the FIRST --n dataset rows, which are also the first rows
ingest.py stores. So keep --n <= the --limit you ingested with, or those
questions' abstracts won't be in the store and hit-rate will look artificially
low. (Ingest with no --limit to eval on anything.)
"""
import argparse

from anthropic import Anthropic
from datasets import load_dataset

import config
from query import get_collection, retrieve

VALID_DECISIONS = {"yes", "no", "maybe"}


def gold_examples(n: int):
    ds = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split=config.DATASET_SPLIT)
    return ds.select(range(min(n, len(ds))))


def classify(client: Anthropic, question: str, hits) -> str:
    """Ask Claude for a single-word yes/no/maybe from the retrieved chunks.

    We keep this call deliberately tiny (plain context, ~1 output token) to keep
    eval cheap. Citations are off here — we only need the decision label.
    """
    context = "\n\n".join(f"[chunk {i}] {text}" for i, (text, _m, _d) in enumerate(hits))
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=8,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Based ONLY on the context, answer with exactly one word: "
                    "yes, no, or maybe."
                ),
            }
        ],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    word = text.strip().lower().split()[0].strip(".,!") if text.strip() else ""
    return word if word in VALID_DECISIONS else "maybe"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval + answer quality.")
    parser.add_argument("--n", type=int, default=10, help="Number of gold questions to evaluate.")
    args = parser.parse_args()

    collection = get_collection(config.COLLECTION_PUBMEDQA)  # eval is tied to the PubMedQA baseline
    client = Anthropic()
    examples = gold_examples(args.n)

    hits_at_k = 0
    correct = 0
    total = len(examples)

    print(f"Evaluating {total} questions (top-{config.TOP_K} retrieval)\n")
    print(f"{'#':>3}  {'hit':>3}  {'gold':>5}  {'pred':>5}  question")
    print("-" * 80)

    for idx, row in enumerate(examples):
        question = row["question"]
        gold_pubid = str(row["pubid"])
        gold_decision = row["final_decision"]

        hits = retrieve(collection, question)
        retrieved_pubids = {m["pubid"] for _t, m, _d in hits}
        hit = gold_pubid in retrieved_pubids
        hits_at_k += hit

        pred = classify(client, question, hits)
        correct += pred == gold_decision

        mark = "yes" if hit else "no"
        print(f"{idx:>3}  {mark:>3}  {gold_decision:>5}  {pred:>5}  {question[:52]}")

    print("-" * 80)
    print(f"Retrieval hit-rate (gold abstract in top-{config.TOP_K}): "
          f"{hits_at_k}/{total} = {hits_at_k / total:.0%}")
    print(f"Decision accuracy (Claude vs. gold yes/no/maybe):       "
          f"{correct}/{total} = {correct / total:.0%}")


if __name__ == "__main__":
    main()
