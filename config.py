"""Shared configuration.

Every script imports from here so ingest / query / eval can never disagree
about which model, collection, or chunk size is in play. If you want to
experiment, change a value here and re-run — that's the whole point.
"""
import os
from pathlib import Path


# --- Secrets ---------------------------------------------------------------
# Load a local .env file (if present) into the environment BEFORE anything else
# runs, so the anthropic SDK finds ANTHROPIC_API_KEY automatically. Keeping the
# key in .env (which .gitignore ignores) means it never touches source control
# or the shell history. This is a minimal parser — no python-dotenv dependency.
def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # Don't clobber a value already exported in the real environment.
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

# --- Dataset ---------------------------------------------------------------
# "qiaojin/PubMedQA" is the canonical Parquet hosting of PubMedQA on the Hub.
# (The bare "pubmed_qa" name uses a legacy loading script that newer versions
# of `datasets` refuse to run without trust_remote_code=True — this avoids it.)
DATASET_NAME = "qiaojin/PubMedQA"
DATASET_CONFIG = "pqa_labeled"   # 1,000 expert-labeled question/abstract pairs
DATASET_SPLIT = "train"

# --- Embeddings (local, via sentence-transformers) -------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models were trained with an instruction prefix on the QUERY side only.
# Documents/passages get NO prefix. Matching this at query time is worth a few
# points of retrieval accuracy — a classic gotcha when using bge/e5 models.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Chunking --------------------------------------------------------------
# bge-small silently truncates anything past 512 tokens, so we keep chunks
# under that. PubMed abstracts are short, so most become a single chunk —
# the chunker still matters for the occasional long one.
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP_SENTENCES = 1   # carry 1 sentence across a split for continuity

# --- Vector store (Chroma, persisted to disk) ------------------------------
CHROMA_DIR = str(Path(__file__).parent / "chroma_db")
# Two corpora live in the same on-disk store as separate collections:
COLLECTION_PUBMEDQA = "pubmed_contexts"     # original PubMedQA baseline (ingest.py / eval.py)
COLLECTION_ORTHO = "orthopedic_pubmed"      # live-PubMed orthopedic corpus (ingest_pubmed.py)
COLLECTION_FULLTEXT = "orthopedic_fulltext" # PMC open-access FULL-TEXT corpus (pmc.py / ingest_fulltext.py)
# The ACTIVE collection that query.py and server.py read from — flip this to
# switch the whole demo between the baseline and the orthopedic corpus.
COLLECTION_NAME = COLLECTION_ORTHO

# --- Live PubMed source (NCBI Entrez E-utilities) --------------------------
# ingest_pubmed.py pulls real orthopedic abstracts straight from PubMed via the
# two-step esearch -> efetch dance. NCBI etiquette: identify yourself with a
# tool name + email, and stay under ~3 requests/second (no API key needed).
ENTREZ_TOOL = "rag_idea_ortho_demo"
ENTREZ_EMAIL = "sammcdsam@gmail.com"
# Optional NCBI API key (get one free at ncbi.nlm.nih.gov -> Account -> Settings).
# Raises the rate limit from 3 to 10 requests/sec and avoids throttling on big
# pulls. Read from the environment / .env; empty string means "no key".
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
# A focused orthopedic-surgery search: MeSH terms, must have an abstract, English.
PUBMED_QUERY = (
    '("Orthopedic Procedures"[Mesh] OR "Orthopedics"[Mesh] OR arthroplasty '
    'OR "ACL reconstruction" OR "fracture fixation" OR "rotator cuff" '
    'OR "spinal fusion") AND hasabstract[text] AND English[lang]'
)
PUBMED_TARGET = 9999   # NCBI's efetch caps paging at ~9,999 records per query.
#                        To exceed this, partition the query (subtopic/date) and union.
PUBMED_BATCH = 200     # records per efetch page
# Seconds to wait between efetch pages. Higher = gentler on NCBI (fewer throttle
# 400s) but a longer total run. Without an NCBI_API_KEY, keep this conservative.
PUBMED_PAGE_DELAY = 0.5

# --- Subtopic partitions (build a LARGER, more diverse corpus) --------------
# efetch caps at ~10k records per query, so one broad "orthopedic" query maxes
# out (~10k, skewed to the most recent arthroplasty papers). Splitting into
# subtopic queries — each under the cap — and unioning them (deduped by PMID)
# yields a much larger corpus spanning all of orthopedics. Set to {} to fall
# back to the single PUBMED_QUERY.
PUBMED_BASE_FILTER = "hasabstract[text] AND English[lang]"
PUBMED_SUBTOPICS = {
    "arthroplasty":   '"Arthroplasty, Replacement"[Mesh] OR "joint replacement" OR "total knee" OR "total hip"',
    "spine":          '"Spinal Fusion"[Mesh] OR "spine surgery" OR scoliosis OR laminectomy OR "disc arthroplasty"',
    "sports_knee":    '"Anterior Cruciate Ligament Reconstruction"[Mesh] OR "ACL reconstruction" OR meniscus OR "cartilage repair"',
    "trauma":         '"Fracture Fixation"[Mesh] OR "intramedullary nail" OR nonunion OR "orthopedic trauma"',
    "shoulder_elbow": '"rotator cuff" OR "shoulder arthroplasty" OR "shoulder instability" OR "elbow arthroplasty"',
    "hand_wrist":     '"hand surgery"[ti] OR "carpal tunnel" OR scaphoid OR "wrist arthroscopy" OR "distal radius"',
    "foot_ankle":     '"ankle arthroplasty" OR "hallux valgus" OR "ankle fracture" OR "foot surgery"[ti]',
    "pediatric":      '"Orthopedic Procedures"[Mesh] AND (pediatric OR adolescent OR "developmental dysplasia")',
    "oncology":       '"Bone Neoplasms/surgery"[Mesh] OR "orthopedic oncology" OR "musculoskeletal sarcoma"',
    "infection":      '"periprosthetic joint infection" OR "prosthetic joint infection" OR osteomyelitis[ti]',
}
PUBMED_SUBTOPIC_TARGET = 3000   # max abstracts per subtopic (capped at the ~9,999 efetch limit)
# Local corpus cache: download PubMed ONCE to this JSONL file, then ingest from
# disk any number of times with no network. (Also the air-gap story: fetch once,
# operate offline.) Ignored by git via .gitignore below if you add it there.
CORPUS_CACHE = str(Path(__file__).parent / "ortho_corpus.jsonl")

# --- Full text (PubMed Central Open Access subset) -------------------------
# Abstracts come from PubMed; full text lives in PMC, but only the Open Access
# subset is fetchable as machine-readable JATS XML (~half of recent ortho PMIDs).
# Full text is ~50x longer than an abstract and carries the reference list — the
# raw material for a citation graph. efetch is one call per article, so we cap the
# number pulled for the demo. See pmc.py.
FULLTEXT_CACHE = str(Path(__file__).parent / "ortho_fulltext.jsonl")
FULLTEXT_TARGET = 1500   # max OA full-text articles to pull (each is one efetch call)

# --- Background reference corpus (Wikipedia) -------------------------------
# PubMed gives us RESEARCH ("what did this study find?"); it has no BACKGROUND
# ("what is a total knee arthroplasty?"). These categories add the missing layer:
# procedures, anatomy, and conditions. Each chunk is tagged source_type=
# "reference" (vs "research") so the UI can label it and retrieval can filter —
# Wikipedia is not peer-reviewed and shouldn't be dressed up as if it were.
# The compartment value maps each category onto the SAME need-to-know
# compartments the PubMed corpus already uses (see access.py), so background
# articles are access-controlled identically. See wiki.py.
WIKI_CATEGORIES = {
    "Orthopedic surgical procedures": "arthroplasty",
    "Bone fractures": "trauma",
    "Arthroplasty": "arthroplasty",
    "Spinal cord disorders": "spine",
    "Vertebral column": "spine",
    "Knee surgery": "sports_knee",
    "Shoulder surgery": "shoulder_elbow",
    "Hand surgery": "hand_wrist",
    "Foot and ankle surgery": "foot_ankle",
    "Bone tumors": "oncology",
    "Musculoskeletal disorders": "arthroplasty",
    "Sports injuries": "sports_knee",
}
REFERENCE_CACHE = str(Path(__file__).parent / "ortho_reference.jsonl")
REFERENCE_TARGET = 400   # background articles to cache

# StatPearls — a PEER-REVIEWED clinical reference, the better background layer
# (Wikipedia is the fallback: fine content, but not peer-reviewed). Chapters are
# section-structured exactly like the background a clinician wants: Indications,
# Technique, Complications. Sourced from NCBI's Literature Archive bulk tarball,
# NOT PMC/efetch — see statpearls.py for the routes that don't work.
# Licence CC BY-NC-ND 4.0: distribution of unaltered excerpts is permitted for
# NON-COMMERCIAL use WITH CREDIT — every chunk carries the attribution string.
STATPEARLS_CACHE = str(Path(__file__).parent / "ortho_statpearls.jsonl")

# --- Retrieval -------------------------------------------------------------
TOP_K = 5

# --- Reranking (two-stage retrieval) ---------------------------------------
# The bi-encoder (bge-small) scores query and chunk INDEPENDENTLY — fast enough
# to search all 34k chunks, but it never lets them "look at each other". A
# cross-encoder reranker takes (question, chunk) as ONE input and scores their
# joint relevance far more accurately — too slow for 34k, so we run it only on
# the top-N bi-encoder candidates and keep the best TOP_K. Same BAAI family as
# the embedder, so it's a matched pair. Loads via sentence-transformers.
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_CANDIDATES = 30   # first-stage pool the reranker narrows down to TOP_K

# --- LLM (pluggable: Claude frontier API OR a local Ollama model) ----------
CLAUDE_MODEL = "claude-haiku-4-5"   # cheap + fast; good enough for grounded Q&A
MAX_TOKENS = 1024
# Local model for the air-gap / offline path — runs on the GPU via Ollama, $0,
# no internet. The whole point: identical RAG pipeline, no API needed.
# A 35B mixture-of-experts with ~3B active params: fits the 3090's 24GB and runs
# near 8B speed while following the "cite [n], don't invent" instruction far
# better than llama3.1:8b did — which is the job here, since RAG hands the model
# its facts. Swap for any pulled model (`ollama list`).
LOCAL_MODEL = "mistral-small:24b"
OLLAMA_URL = "http://localhost:11434"
