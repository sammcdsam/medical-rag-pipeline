"""Fetch orthopedic-surgery abstracts from LIVE PubMed via NCBI Entrez.

The E-utilities flow, using the **history server** so we can pull more than the
9,999-per-call esearch cap:
  esearch (usehistory=y) -> stash the full result set server-side; get WebEnv +
                            query_key + total count
  efetch  (WebEnv paging) -> pull abstracts in batches, retstart/retmax paging

Standard library only (urllib + ElementTree). NCBI etiquette, which we follow:
identify every request with a tool name + email, and stay under ~3 requests/sec
(we sleep between efetch batches). No API key required at this volume.

This is the "forward-deployed" part of the project: instead of a canned dataset,
we build a domain knowledge base from real, current scientific literature.
"""
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import config

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# NCBI's efetch can only page to record ~9,999 via retstart — beyond that it
# returns "Search Backend failed". To build a LARGER corpus, partition the query
# (by subtopic or date range) so each slice stays under this cap, and union them.
EFETCH_MAX = 9999


def _get(endpoint: str, params: dict) -> bytes:
    """One E-utilities GET, with tool+email attached and robust backoff.

    NCBI throws transient 400/429/5xx in bursts (rate blips, history hiccups,
    IP throttling), so we retry up to 8 times with exponential backoff. Setting
    NCBI_API_KEY (in .env) raises the rate limit 3->10 req/s and largely avoids
    these blocks — recommended for large pulls.
    """
    query = {"tool": config.ENTREZ_TOOL, "email": config.ENTREZ_EMAIL, **params}
    if config.NCBI_API_KEY:
        query["api_key"] = config.NCBI_API_KEY
    url = f"{BASE}/{endpoint}?" + urllib.parse.urlencode(query)
    last_err = None
    for attempt in range(10):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except Exception as err:
            last_err = err
            time.sleep(min(2 ** attempt, 60))  # 1,2,4,8,16,32,60,60,60,60 — rides multi-minute blocks
    raise last_err


def search_history(query: str) -> dict:
    """esearch with usehistory=y — stashes the whole result set on NCBI's server.

    Returns the WebEnv + query_key handle (so efetch can page through ALL matches,
    not just the first 9,999) plus the total match count.
    """
    raw = _get("esearch.fcgi", {
        "db": "pubmed", "term": query, "sort": "pub_date",  # stable order across resume runs
        "usehistory": "y", "retmax": 0, "retmode": "json",
    })
    res = json.loads(raw)["esearchresult"]
    return {"webenv": res["webenv"], "query_key": res["querykey"], "count": int(res["count"])}


def _abstract_text(article: ET.Element) -> str:
    """Join an article's (possibly section-labelled) AbstractText nodes.

    itertext() flattens nested inline tags (<i>, <sup>, ...) so we don't lose
    text. Labelled sections (Background/Methods/...) are prefixed for readability.
    """
    parts = []
    for ab in article.findall(".//AbstractText"):
        text = "".join(ab.itertext()).strip()
        if not text:
            continue
        label = ab.get("Label")
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts)


def _parse_articles(xml_bytes: bytes) -> list[dict]:
    """Parse one efetch XML page into doc dicts (skipping abstract-less records)."""
    root = ET.fromstring(xml_bytes)
    docs = []
    for art in root.findall(".//PubmedArticle"):
        text = _abstract_text(art)
        if not text:
            continue
        title_el = art.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        docs.append({
            "pmid": art.findtext(".//PMID") or "",
            "title": title,
            "journal": art.findtext(".//Journal/Title") or "",
            "year": art.findtext(".//JournalIssue/PubDate/Year") or "",
            "text": text,
        })
    return docs


def get_corpus(query: str | None = None, target: int | None = None) -> list[dict]:
    """End to end: esearch(history) -> paged efetch -> parsed docs."""
    query = query or config.PUBMED_QUERY
    target = target or config.PUBMED_TARGET
    batch = config.PUBMED_BATCH

    hist = search_history(query)
    n = min(target, hist["count"], EFETCH_MAX)  # efetch retstart caps at ~9,999
    print(f"PubMed matched {hist['count']:,} articles; fetching {n} via the history server...")
    time.sleep(0.5)  # give NCBI a beat to make the history handle ready

    def _page(h: dict, start: int, size: int) -> bytes:
        return _get("efetch.fcgi", {
            "db": "pubmed",
            "WebEnv": h["webenv"], "query_key": h["query_key"],
            "retstart": start, "retmax": size,
            "rettype": "abstract", "retmode": "xml",
        })

    docs = []
    for start in range(0, n, batch):
        size = min(batch, n - start)
        try:
            xml = _page(hist, start, size)
        except Exception:
            # A history handle can go stale/flaky mid-run — refresh it and retry once.
            time.sleep(2.0)
            hist = search_history(query)
            time.sleep(0.5)
            xml = _page(hist, start, size)
        docs.extend(_parse_articles(xml))
        print(f"  fetched {start + size}/{n}...")
        time.sleep(config.PUBMED_PAGE_DELAY)  # gentle pacing to avoid throttling
    print(f"Parsed {len(docs)} usable abstracts.")
    return docs


# --- Local corpus cache (download once, ingest offline) --------------------

def _efetch_page(hist: dict, start: int, size: int) -> bytes:
    return _get("efetch.fcgi", {
        "db": "pubmed",
        "WebEnv": hist["webenv"], "query_key": hist["query_key"],
        "retstart": start, "retmax": size,
        "rettype": "abstract", "retmode": "xml",
    })


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def load_corpus(path: str | None = None, limit: int | None = None) -> list[dict]:
    """Read cached abstracts back from the local JSONL file. No network."""
    path = path or config.CORPUS_CACHE
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
            if limit and len(docs) >= limit:
                break
    return docs


def download_corpus(path: str | None = None, query: str | None = None,
                    target: int | None = None) -> int:
    """Resumable download of the corpus to a JSONL cache.

    A sidecar `<path>.progress` file records the next record offset, so if NCBI
    throttles mid-download you can just re-run this and it continues where it
    left off (esearch is sorted by pub_date for a stable order across runs).
    Returns the number of abstracts now cached.
    """
    path = path or config.CORPUS_CACHE
    query = query or config.PUBMED_QUERY
    target = target or config.PUBMED_TARGET
    batch = config.PUBMED_BATCH
    progress_path = path + ".progress"

    start = 0
    if os.path.exists(progress_path):
        start = int((open(progress_path).read().strip() or "0"))

    hist = search_history(query)
    n = min(target, hist["count"], EFETCH_MAX)  # efetch retstart caps at ~9,999
    if start >= n:
        if os.path.exists(progress_path):
            os.remove(progress_path)   # nothing left to fetch; clear the resume marker
        print(f"Cache complete ({_count_lines(path)} abstracts, at NCBI's efetch cap): {path}")
        return _count_lines(path)

    print(f"PubMed matched {hist['count']:,}; downloading records {start}..{n} "
          f"(resumable) -> {path}")
    time.sleep(0.5)

    mode = "a" if start > 0 and os.path.exists(path) else "w"
    with open(path, mode) as out:
        for s in range(start, n, batch):
            size = min(batch, n - s)
            try:
                xml = _efetch_page(hist, s, size)
            except Exception:
                # Refresh the (possibly stale) history handle once and retry the page.
                time.sleep(2.0)
                hist = search_history(query)
                time.sleep(0.5)
                xml = _efetch_page(hist, s, size)
            for d in _parse_articles(xml):
                out.write(json.dumps(d) + "\n")
            out.flush()
            with open(progress_path, "w") as p:
                p.write(str(s + size))
            print(f"  saved through record {s + size}/{n}")
            time.sleep(config.PUBMED_PAGE_DELAY)  # gentle pacing to avoid throttling

    if os.path.exists(progress_path):
        os.remove(progress_path)
    total = _count_lines(path)
    print(f"Done. {total} abstracts cached at {path}")
    return total


def download_corpus_multi(path: str | None = None, subtopics: dict | None = None,
                          per_target: int | None = None) -> int:
    """Build a larger, diverse corpus by unioning subtopic queries.

    Each subtopic query stays under the ~10k efetch cap; results are unioned and
    deduped by PMID, and each abstract is tagged with its subtopic. Resumable per
    subtopic via a `.mprogress` sidecar (records completed subtopics + the offset
    within the in-progress one), so a blip just continues where it stopped.
    """
    path = path or config.CORPUS_CACHE
    subtopics = subtopics or config.PUBMED_SUBTOPICS
    per_target = per_target or config.PUBMED_SUBTOPIC_TARGET
    batch = config.PUBMED_BATCH
    prog_path = path + ".mprogress"

    if os.path.exists(prog_path):
        state = json.loads(open(prog_path).read())
        seen = {d["pmid"] for d in load_corpus(path)} if os.path.exists(path) else set()
        print(f"Resuming: {len(state['done'])} subtopics done, {len(seen)} abstracts so far")
    else:
        state = {"done": [], "current": None, "offset": 0}
        open(path, "w").close()   # fresh start — overwrite any old single-query cache
        seen = set()

    with open(path, "a") as out:
        for name, term in subtopics.items():
            if name in state["done"]:
                continue
            query = f"({term}) AND {config.PUBMED_BASE_FILTER}"
            hist = search_history(query)
            n = min(per_target, hist["count"], EFETCH_MAX)
            start = state["offset"] if state.get("current") == name else 0
            print(f"[{name}] matched {hist['count']:,}; fetching {start}..{n}")
            time.sleep(0.5)
            for s in range(start, n, batch):
                size = min(batch, n - s)
                try:
                    xml = _efetch_page(hist, s, size)
                except Exception:
                    time.sleep(2.0)
                    hist = search_history(query)
                    time.sleep(0.5)
                    xml = _efetch_page(hist, s, size)
                added = 0
                for d in _parse_articles(xml):
                    if d["pmid"] and d["pmid"] not in seen:
                        seen.add(d["pmid"])
                        d["subtopic"] = name
                        out.write(json.dumps(d) + "\n")
                        added += 1
                out.flush()
                state.update(current=name, offset=s + size)
                with open(prog_path, "w") as p:
                    p.write(json.dumps(state))
                print(f"  [{name}] {s + size}/{n}  (+{added} new, {len(seen)} total)")
                time.sleep(config.PUBMED_PAGE_DELAY)
            state["done"].append(name)
            state["current"], state["offset"] = None, 0
            with open(prog_path, "w") as p:
                p.write(json.dumps(state))

    if os.path.exists(prog_path):
        os.remove(prog_path)
    total = _count_lines(path)
    print(f"Done. {total} unique abstracts across {len(subtopics)} subtopics -> {path}")
    return total
