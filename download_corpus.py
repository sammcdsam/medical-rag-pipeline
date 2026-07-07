"""Step 1 (network, once): download the orthopedic corpus to a local JSONL cache.

    python download_corpus.py                 # fetch config.PUBMED_TARGET abstracts
    python download_corpus.py --target 5000

RESUMABLE: if NCBI throttles mid-download (transient HTTP 400s from too many
requests), just run it again — a sidecar `.progress` file remembers where it
stopped and it continues. No embedding happens here; that's ingest_pubmed.py.

Set NCBI_API_KEY in .env to raise the rate limit 3->10 req/s and avoid throttling.
"""
import argparse

import config
import pubmed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PubMed orthopedic abstracts to a local cache.")
    parser.add_argument("--target", type=int, default=config.PUBMED_TARGET,
                        help="Single-query mode: how many abstracts to cache.")
    parser.add_argument("--per-target", type=int, default=None,
                        help="Subtopic mode: max abstracts per subtopic.")
    parser.add_argument("--single", action="store_true",
                        help="Force the single PUBMED_QUERY instead of subtopic partitions.")
    args = parser.parse_args()

    # Subtopic partitions build a bigger, more diverse corpus; single query is
    # the fallback (or when --single is passed).
    if config.PUBMED_SUBTOPICS and not args.single:
        total = pubmed.download_corpus_multi(per_target=args.per_target)
    else:
        total = pubmed.download_corpus(target=args.target)
    print(f"\nCache ready: {total} abstracts. Now run `python ingest_pubmed.py --rebuild` (offline).")


if __name__ == "__main__":
    main()
