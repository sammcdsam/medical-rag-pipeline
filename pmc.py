"""Fetch FULL-TEXT open-access articles from PubMed Central (PMC).

Abstracts come from PubMed (pubmed.py). Full text lives in PMC, but only the Open
Access subset is fetchable as machine-readable JATS XML. The flow:

    PMID --(idconv)--> PMCID --(efetch db=pmc)--> JATS XML --> sections + references

Only ~half of recent orthopedic PMIDs are OA; the rest have no <body> and are
skipped. Full text is ~50x longer than an abstract and, crucially, carries the
article's REFERENCE LIST — the raw material for a citation graph later.

Reuses pubmed._get (NCBI etiquette + backoff) for efetch. The ID Converter API is
a different host, so it gets its own small batched GET.

    python pmc.py --target 50        # cache 50 OA full-text articles
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, deque

import config
import pubmed

IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"


def _idconv(pmids: list[str]) -> dict:
    """Map PMIDs -> PMCIDs in batches (up to 200/call) via the ID Converter API.

    Cleaner and far fewer requests than one elink per PMID. Not every PMID has a
    PMCID (only ones deposited in PMC); missing ones are simply absent from the map.
    """
    mapping = {}
    for i in range(0, len(pmids), 200):
        batch = pmids[i:i + 200]
        params = {"ids": ",".join(batch), "format": "json", "versions": "no",
                  "tool": config.ENTREZ_TOOL, "email": config.ENTREZ_EMAIL}
        url = IDCONV + "?" + urllib.parse.urlencode(params)
        data = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = json.loads(r.read().decode("utf-8", "replace"), strict=False)
                break
            except Exception:
                time.sleep(min(2 ** attempt, 30))
        if data:
            for rec in data.get("records", []):
                if rec.get("pmcid"):
                    # idconv may echo pmid as a JSON number — coerce to str so keys
                    # match the string PMIDs in the abstract cache.
                    mapping[str(rec["pmid"])] = rec["pmcid"]   # e.g. "PMC13162543"
        print(f"  mapped {min(i + 200, len(pmids))}/{len(pmids)} PMIDs -> {len(mapping)} in PMC")
        time.sleep(0.34)
    return mapping


def _sections(sec: ET.Element, prefix: str = ""):
    """Recursively yield {heading, text} for a JATS <sec> and its nested <sec>s.

    Uses direct <p> children only (not descendants), so nested sections don't
    double-count paragraphs. Headings are breadcrumbed (Parent > Child)."""
    title = (sec.findtext("title") or "").strip()
    heading = f"{prefix} > {title}".strip(" >") if prefix else title
    paras = [" ".join("".join(p.itertext()).split()) for p in sec.findall("p")]
    text = "\n".join(t for t in paras if t)
    if text:
        yield {"heading": heading or "Section", "text": text}
    for child in sec.findall("sec"):
        yield from _sections(child, heading)


def _references(root: ET.Element) -> list[dict]:
    """Extract the reference list. Keeping each ref's PMID (when present) is what
    lets us build a citation graph later."""
    refs = []
    for ref in root.findall(".//ref-list/ref"):
        refs.append({
            "pmid": ref.findtext('.//pub-id[@pub-id-type="pmid"]'),
            "title": (ref.findtext(".//article-title") or "").strip(),
            "raw": " ".join("".join(ref.itertext()).split()),
        })
    return refs


def parse_jats(xml_bytes: bytes) -> dict | None:
    """Parse PMC JATS XML into {sections, references}, or None if not OA full text."""
    root = ET.fromstring(xml_bytes)
    body = root.find(".//body")
    if body is None:
        return None   # not in the OA subset -> no machine-readable full text
    sections = []
    # Paragraphs that sit directly under <body> (before the first <sec>).
    intro = "\n".join(t for t in
                      (" ".join("".join(p.itertext()).split()) for p in body.findall("p")) if t)
    if intro:
        sections.append({"heading": "", "text": intro})
    for sec in body.findall("sec"):
        sections.extend(_sections(sec))
    return {"sections": sections, "references": _references(root)}


def fetch_fulltext(pmcid: str) -> dict | None:
    """efetch one PMC article and parse it; None if it isn't OA full text."""
    numeric = pmcid.replace("PMC", "")
    xml = pubmed._get("efetch.fcgi", {"db": "pmc", "id": numeric, "retmode": "xml"})
    try:
        return parse_jats(xml)
    except ET.ParseError:
        return None


def load_fulltext(path: str | None = None, limit: int | None = None) -> list[dict]:
    """Read cached full-text articles back from the JSONL file. No network."""
    path = path or config.FULLTEXT_CACHE
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


def download_fulltext(path: str | None = None, target: int | None = None,
                      source: str | None = None, delay: float | None = None) -> int:
    """Map abstract PMIDs -> PMC, fetch OA full text, cache to JSONL (resumable, balanced).

    BALANCED across subtopics: aims for target/N_subtopics articles per subtopic and
    round-robins across them, so the corpus spans all of orthopedics rather than
    filling up on whichever subtopic comes first. Resumes by skipping cached PMIDs
    (and counts what's already saved toward each subtopic's quota). Robust for long
    unattended runs: `delay` seconds between fetches (gentle on NCBI), and a failed
    article is skipped rather than aborting the run.
    """
    path = path or config.FULLTEXT_CACHE
    target = target or config.FULLTEXT_TARGET
    delay = config.PUBMED_PAGE_DELAY if delay is None else delay

    by_pmid = {d["pmid"]: d for d in pubmed.load_corpus(source) if d.get("pmid")}
    cached = load_fulltext(path)
    done = {d["pmid"] for d in cached}
    remaining = [p for p in by_pmid if p not in done]
    print(f"{len(by_pmid)} abstracts; {len(done)} full-text already cached; "
          f"mapping {len(remaining)} remaining PMIDs -> PMC...")

    mapping = _idconv(remaining)   # pmid -> pmcid, PMC-linked only

    # One queue of candidate PMIDs per subtopic; a per-subtopic quota for balance.
    def sub_of(pmid: str) -> str:
        return by_pmid[pmid].get("subtopic") or "(untagged)"

    subs = sorted({sub_of(p) for p in mapping})
    per_target = max(1, target // max(len(subs), 1))
    queues = {s: deque() for s in subs}
    for pmid in mapping:
        queues[sub_of(pmid)].append(pmid)
    saved_by_sub = Counter(sub_of(d["pmid"]) if d["pmid"] in by_pmid else (d.get("subtopic") or "(untagged)")
                           for d in cached)
    print(f"target {target} across {len(subs)} subtopics (~{per_target} each); "
          f"gentle delay {delay}s/fetch")

    saved = len(done)
    with open(path, "a") as out:
        progressed = True
        while saved < target and progressed:
            progressed = False
            for s in subs:
                if saved >= target or saved_by_sub[s] >= per_target or not queues[s]:
                    continue
                progressed = True
                pmid = queues[s].popleft()
                try:
                    parsed = fetch_fulltext(mapping[pmid])
                except Exception:
                    parsed = None
                time.sleep(delay)
                if not parsed or not parsed["sections"]:
                    continue   # not open-access / unusable — doesn't count toward the quota
                src = by_pmid[pmid]
                out.write(json.dumps({
                    "pmid": pmid, "pmcid": mapping[pmid],
                    "title": src.get("title", ""), "journal": src.get("journal", ""),
                    "year": src.get("year", ""), "subtopic": src.get("subtopic", ""),
                    "sections": parsed["sections"], "references": parsed["references"],
                }) + "\n")
                out.flush()
                saved += 1
                saved_by_sub[s] += 1
                if saved % 25 == 0:
                    mix = ", ".join(f"{k}:{v}" for k, v in sorted(saved_by_sub.items()))
                    print(f"  {saved}/{target} cached | {mix}", flush=True)

    print(f"Done. {saved} full-text articles cached -> {path}")
    print("by subtopic:", dict(sorted(saved_by_sub.items())))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PMC open-access full text for the abstract corpus.")
    parser.add_argument("--target", type=int, default=config.FULLTEXT_TARGET,
                        help="Total OA full-text articles to cache, balanced across subtopics.")
    parser.add_argument("--delay", type=float, default=None,
                        help="Seconds between fetches (gentle on NCBI for long runs; default from config).")
    args = parser.parse_args()
    download_fulltext(target=args.target, delay=args.delay)


if __name__ == "__main__":
    main()
