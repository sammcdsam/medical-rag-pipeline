"""Fetch orthopedic BACKGROUND articles from Wikipedia.

Why this exists: the PubMed corpus is all *research* — abstracts and full text
reporting what a study found. Ask it "what are the risk factors for DVT after
TKA?" and it shines (odds ratios, cohort sizes). Ask it "what IS a total knee
arthroplasty?" and it returns five papers that assume you already know, because
nothing in a research corpus explains the background. That's a corpus gap, not a
retrieval bug.

This adds the missing layer: encyclopedic articles on procedures, anatomy, and
conditions — the "what is / how does it work" material.

    PubMed papers  -> source_type "research"  ("what does the latest evidence show?")
    Wikipedia      -> source_type "reference" ("what is this thing?")

Both land in the SAME collection, tagged, so one query can surface whichever is
actually closer to the question — and the UI can honestly show which kind of
source answered.

Why Wikipedia and not StatPearls (the better clinical reference)? StatPearls is
NOT in PMC and NCBI's efetch won't serve Bookshelf full text, so the only route
would be bulk-scraping their site — against NCBI's usage policy — and its
CC BY-NC-ND licence forbids derivative works anyway. Wikipedia is CC BY-SA
(derivatives fine, attribution required) with a real bulk API. The honest
trade-off: Wikipedia is NOT peer-reviewed, which is exactly why every chunk is
tagged `source_type="reference"` and the UI labels it — a clinician should weigh
it differently than a paper.

API etiquette: Wikipedia asks for a descriptive User-Agent with contact info.

    python wiki.py                 # cache the ortho background corpus
    python wiki.py --target 100    # cap the number of articles
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

import config

API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks bots to identify themselves with contact info (same etiquette as
# NCBI's tool+email). A generic urllib UA gets throttled or blocked.
UA = f"rag_idea_ortho_demo/1.0 ({config.ENTREZ_EMAIL})"


def _get(params: dict) -> dict:
    """One MediaWiki API call, with the same backoff shape as pubmed._get."""
    params = {**params, "format": "json", "formatversion": "2"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 30))
    return {}


def category_members(category: str, limit: int = 500) -> list[str]:
    """Article titles in a category (pages only — no subcategories or files)."""
    titles, cont = [], None
    while len(titles) < limit:
        params = {"action": "query", "list": "categorymembers",
                  "cmtitle": f"Category:{category}", "cmlimit": "500",
                  "cmtype": "page"}
        if cont:
            params["cmcontinue"] = cont
        data = _get(params)
        titles += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        time.sleep(0.2)
    return titles[:limit]


def fetch_article(title: str) -> dict | None:
    """Plain-text extract for ONE article.

    Deliberately one-per-call: passing many titles looks like it works but the API
    answers "exlimit was too large for a whole article extracts request, lowered
    to 1" and returns a single extract with the rest silently empty. Whole-article
    extracts are one per request, so we pay one call each.
    """
    data = _get({
        "action": "query", "prop": "extracts", "titles": title,
        "explaintext": "1",           # plain text, not HTML — ready to chunk
        "exsectionformat": "plain",
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None
    page = pages[0]
    text = (page.get("extract") or "").strip()
    # Skip stubs and redirects: too short to carry real background information.
    if page.get("missing") or len(text) < 500:
        return None
    return {
        "pageid": str(page["pageid"]),
        "title": page["title"],
        "text": text,
        "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(page["title"].replace(" ", "_")),
    }


def load_reference(path: str | None = None, limit: int | None = None) -> list[dict]:
    """Read the cached background corpus back. No network (the air-gap path)."""
    path = path or config.REFERENCE_CACHE
    docs = []
    if not os.path.exists(path):
        return docs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
                if limit and len(docs) >= limit:
                    break
    return docs


def download_reference(path: str | None = None, target: int | None = None) -> int:
    """Walk the ortho categories, fetch each article once, cache to JSONL.

    Resumable: already-cached pageids are skipped, so a re-run only fetches what's
    new (same contract as pubmed.download_corpus / pmc.download_fulltext).
    """
    path = path or config.REFERENCE_CACHE
    target = target or config.REFERENCE_TARGET

    have = {d["pageid"] for d in load_reference(path)}
    print(f"{len(have)} articles already cached in {os.path.basename(path)}")

    # Collect candidate titles across every ortho category (deduped — articles
    # routinely sit in several categories at once).
    titles, seen = [], set()
    for cat, compartment in config.WIKI_CATEGORIES.items():
        members = category_members(cat)
        print(f"Category:{cat} -> {len(members)} articles")
        for t in members:
            if t not in seen:
                seen.add(t)
                titles.append((t, compartment))
        time.sleep(0.2)
    print(f"{len(titles)} unique candidate articles across {len(config.WIKI_CATEGORIES)} categories")

    compartment_of = dict(titles)
    todo = [t for t, _c in titles][:target * 2]   # over-fetch: stubs get dropped

    added = 0
    with open(path, "a") as f:
        for n, title in enumerate(todo, 1):
            if added >= target:
                break
            art = fetch_article(title)
            if art is None or art["pageid"] in have:
                continue
            art["compartment"] = compartment_of.get(art["title"], "general")
            f.write(json.dumps(art) + "\n")
            f.flush()              # crash-safe: the cache is always valid JSONL
            have.add(art["pageid"])
            added += 1
            if added % 25 == 0:
                print(f"  {added} cached ({n}/{len(todo)} titles tried)")
            time.sleep(0.2)        # be polite: ~5 req/s
    print(f"Added {added} articles -> {len(have)} total in {path}")
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache the Wikipedia orthopedic background corpus.")
    ap.add_argument("--target", type=int, default=None, help="Max articles to cache.")
    args = ap.parse_args()
    download_reference(target=args.target)


if __name__ == "__main__":
    main()
