"""Extract orthopedic chapters from the StatPearls bulk archive.

StatPearls is a peer-reviewed clinical reference — chapters are written as
BACKGROUND ("Indications", "Technique or Treatment", "Complications"), which is
exactly what a corpus of research papers lacks. It's the reference layer
Wikipedia was standing in for, but clinically authored and reviewed.

Getting it took three tries, so the routes that DON'T work, to save the next
person the search:
  - NOT in PMC — the ID converter maps 0/5 chapter PMIDs, so pmc.py can't fetch it.
  - efetch db=books returns an ID list, no full text. E-utilities is a dead end.
  - The live site would need bulk scraping, which NCBI's usage policy forbids.
The route that DOES work is NCBI's Literature Archive — the sanctioned bulk
channel, listing every OA book in `file_list.csv`:

    https://ftp.ncbi.nlm.nih.gov/pub/litarch/   -> statpearls_NBK430685.tar.gz (~1.8 GB)

LICENCE: CC BY-NC-ND 4.0. Verbatim from the chapters' own <permissions> block:
"permits others to distribute the work, provided that the article is not altered
or used commercially. You are not required to obtain permission to distribute
this article, provided that you credit the author and journal." So: a
non-commercial demo may distribute unaltered excerpts WITH CREDIT. Every chunk
therefore carries `source` = the attribution string, and the UI shows it. Don't
strip that — it's the licence's one condition, not decoration.

    python statpearls.py --archive statpearls.tar.gz     # -> ortho_statpearls.jsonl
"""
import argparse
import json
import os
import re
import tarfile
import xml.etree.ElementTree as ET

import config

ATTRIBUTION = "StatPearls Publishing (CC BY-NC-ND 4.0)"

# Chapter selection. StatPearls covers all of medicine (9,640 chapters); we keep
# the orthopedic clinical chapters PLUS all anatomy chapters (anatomy is
# foundational background for orthopedics — the "what is this structure" layer).
# Matching on the TITLE only — not the body — because a cardiology chapter can
# mention "fracture" in passing, and body matching drags in most of the book.
# Each pattern maps onto the same need-to-know compartments the PubMed corpus
# uses (access.py), so background is access-controlled the same; general anatomy
# lands in a dedicated "anatomy" compartment.
ORTHO_PATTERNS = [
    (r"\b(arthroplasty|joint replacement|arthrodesis)", "arthroplasty"),
    (r"\b(fracture|dislocation|nonunion|malunion|orthopedic trauma|internal fixation)", "trauma"),
    (r"\b(acl\b|anterior cruciate|posterior cruciate|menisc|patell|knee|tibial plateau)", "sports_knee"),
    (r"\b(spine|spinal|vertebra|scolios|laminectomy|discectomy|kypho|spondyl|lumbar|thoracic disc)", "spine"),
    (r"\b(rotator cuff|shoulder|clavicle|humer|elbow|olecranon|acromio)", "shoulder_elbow"),
    (r"\b(carpal|scaphoid|wrist|hand|finger|thumb|metacarpal|phalan|radius|ulna)", "hand_wrist"),
    (r"\b(ankle|foot|hallux|calcaneus|metatars|achilles|plantar|talus)", "foot_ankle"),
    (r"\b(bone tumor|osteosarcoma|sarcoma|bone neoplasm|bone cancer|giant cell tumor)", "oncology"),
    (r"\b(osteomyelitis|septic arthritis|periprosthetic joint infection|prosthetic joint infection)", "infection"),
    (r"\b(osteotomy|osteoarthritis|osteoporos|osteogenesis|orthopedic|arthroscop)", "arthroplasty"),
]

# Anatomy-of-everything is StatPearls' biggest false-positive source: "Larynx
# Cartilage" matched `cartilage`, "Cardinal Ligaments" (a gynaecological
# structure) matched `ligament`, "Maxillary Sinus Fracture" matched `fracture`.
# Region words are far more reliable as a VETO than the injury words are as a
# signal, so anything naming a non-musculoskeletal region is dropped outright.
EXCLUDE_PATTERNS = [
    r"\bhead and neck\b", r"\blarynx|laryng|trachea|pharyn|nasal|sinus|maxill|mandib|dental|tooth|orbit\b",
    r"\babdomen and pelvis\b", r"\bthorax\b", r"\bcardiac|heart|coronary|aortic|pulmonary|lung\b",
    r"\brenal|kidney|bladder|hepatic|liver|bowel|intestin|gastr\b",
    r"\buter|ovar|cervix|vagin|prostat|testic|penil|breast\b",
    r"\bbrain|cerebr|cranial|skull|scalp\b", r"\bocular|eye|cornea|retina\b",
    r"\bcardinal ligament", r"\bvocal\b",
]


def _excluded(title: str) -> bool:
    t = title.lower()
    # "(Archived)" chapters are superseded content — a demo that answers clinical
    # questions shouldn't quote guidance StatPearls itself retired.
    if "(archived)" in t:
        return True
    return any(re.search(p, t) for p in EXCLUDE_PATTERNS)


def _txt(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def classify_title(title: str) -> str | None:
    """Which compartment does this chapter belong to — or None if out of scope.

    Two ways in:
      1. Orthopedic clinical chapters — matched by ORTHO_PATTERNS, with the region
         veto applied first so "Maxillary Sinus Fracture" is dropped, not filed
         under trauma.
      2. ANATOMY chapters — kept wholesale (the user wants all anatomy as
         orthopedic-supporting background). The region veto does NOT apply here:
         "Anatomy, Head and Neck, Larynx" is legitimate anatomy, just not
         musculoskeletal. A musculoskeletal anatomy chapter is filed under its
         ortho region; the rest go to a general `anatomy` compartment.
    """
    t = title.lower()

    # (2) Anatomy chapters — StatPearls titles them "Anatomy, <region>, <part>".
    if re.match(r"anatomy\b", t):
        for pattern, compartment in ORTHO_PATTERNS:
            # Reuse the region keywords, but skip the "arthroplasty" catch-all
            # rows (osteo*/arthr*) — they'd sweep unrelated anatomy in. Only the
            # concrete body-part rows should re-file anatomy onto an ortho region.
            if compartment != "arthroplasty" and re.search(pattern, t):
                return compartment
        return "anatomy"

    # (1) Orthopedic clinical chapters.
    if _excluded(title):
        return None
    for pattern, compartment in ORTHO_PATTERNS:
        if re.search(pattern, t):
            return compartment
    return None


def parse_chapter(xml_bytes: bytes) -> dict | None:
    """BITS chapter -> {title, sections, text}.

    NOTE the schema: this is BITS (book), not the JATS article DTD pmc.py parses,
    so the title lives at book-part-meta/title-group/title. Sections are kept as
    (heading, text) pairs — a chunk should never straddle "Indications" and
    "Complications", the same rule ingest_fulltext.py follows for papers.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    tel = root.find(".//book-part-meta/title-group/title")
    body = root.find(".//body")
    if tel is None or body is None:
        return None
    title = _txt(tel)
    if not title:
        return None

    sections = []
    for sec in body.findall("./sec"):
        head = sec.find("title")
        head = _txt(head) if head is not None else ""
        # Drop the CME/quiz furniture: not clinical content, and it pollutes
        # retrieval with "Click here for a review question" style text.
        if head.lower() in {"review questions", "continuing education activity",
                            "enhancing healthcare team outcomes"}:
            continue
        text = _txt(sec)
        if head and text.startswith(head):
            text = text[len(head):].strip()
        if len(text) >= 200:
            sections.append({"heading": head or "(untitled)", "text": text})
    if not sections:
        return None
    return {"title": title, "sections": sections}


def extract(archive: str, out_path: str | None = None, limit: int | None = None) -> int:
    """Stream the tarball, keep orthopedic chapters, write JSONL.

    Streamed rather than fully extracted: the archive is ~1.8 GB of mostly images
    and non-orthopedic chapters, and we want neither on disk.
    """
    out_path = out_path or config.STATPEARLS_CACHE
    kept = skipped = 0
    with tarfile.open(archive, "r:gz") as tar, open(out_path, "w") as out:
        for member in tar:
            if not member.name.endswith(".nxml"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            chap = parse_chapter(f.read())
            if chap is None:
                continue
            compartment = classify_title(chap["title"])
            if compartment is None:
                skipped += 1
                continue
            chap.update(
                chapter_id=os.path.basename(member.name).replace(".nxml", ""),
                compartment=compartment,
                source=ATTRIBUTION,
            )
            out.write(json.dumps(chap) + "\n")
            kept += 1
            if kept % 50 == 0:
                print(f"  kept {kept} orthopedic chapters ({skipped} non-ortho skipped)")
            if limit and kept >= limit:
                break
    print(f"Kept {kept} orthopedic chapters -> {out_path} ({skipped} non-ortho skipped)")
    return kept


def load(path: str | None = None, limit: int | None = None) -> list[dict]:
    """Read the cached chapters back. No network (the air-gap path)."""
    path = path or config.STATPEARLS_CACHE
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract orthopedic StatPearls chapters.")
    ap.add_argument("--archive", default="statpearls.tar.gz", help="Path to the litarch tarball.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not os.path.exists(args.archive):
        raise SystemExit(
            f"{args.archive} not found. Download it once:\n"
            "  curl -O https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"
        )
    extract(args.archive, limit=args.limit)


if __name__ == "__main__":
    main()
