"""Web demo — ask the RAG pipeline a question in a browser.

This is a thin HTTP skin over the SAME functions the CLI uses (`retrieve` and
`answer` from query.py) — no second pipeline to keep in sync. Reading it is a
good way to see the pieces you learned wired into a request/response.

Run it:
    python server.py                    # binds 0.0.0.0:8022
    # or, with autoreload while editing:
    uvicorn server:app --host 0.0.0.0 --port 8022 --reload

Reach it from your OTHER computer (you're on SSH), two ways:

  1. SSH tunnel (recommended — nothing new exposed to the network):
        ssh -L 8022:localhost:8022 <you>@<this-server>
     then open http://localhost:8022 in the browser on your laptop.

  2. Direct, if the two machines share a network and the port is reachable:
        open http://<this-server-ip>:8022

Retrieval works with NO API key (you'll see the retrieved chunks). Set
ANTHROPIC_API_KEY in the environment before launching to also get Claude's
grounded answer + citations.
"""
import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import config
import llm
from query import get_collection, retrieve, answer, _provenance

app = FastAPI(title="Minimal RAG demo")

# --- Shared state ----------------------------------------------------------
# Load the Chroma collection once and reuse it across requests (the embedding
# model behind retrieve() is likewise cached in embedder.py). We look it up
# lazily so importing this module never crashes if ingest hasn't run yet.
_collection = None


def collection():
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection


def do_retrieve(question: str, k: int, rerank: bool = False):
    """Retrieve, re-opening the collection if the cached handle went stale
    (e.g. the corpus was rebuilt while this server was running)."""
    global _collection
    try:
        return retrieve(collection(), question, k, rerank=rerank)
    except Exception:
        _collection = None
        return retrieve(collection(), question, k, rerank=rerank)


# --- API -------------------------------------------------------------------
class Query(BaseModel):
    question: str
    k: int = config.TOP_K
    backend: str = "claude"   # "claude" (frontier API) or "local" (offline Ollama)
    rerank: bool = False      # add the cross-encoder stage-2 reranker


@app.get("/api/config")
def api_config():
    """What's under the hood — the page shows this, and it's where future
    pluggable model 'versions' will surface automatically."""
    return {
        "embedder": config.EMBED_MODEL,
        "llm": config.CLAUDE_MODEL,
        "local_model": config.LOCAL_MODEL,
        "reranker": config.RERANK_MODEL,
        "rerank_candidates": config.RERANK_CANDIDATES,
        "top_k": config.TOP_K,
        "collection": config.COLLECTION_NAME,
        "has_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


@app.post("/api/query")
def api_query(q: Query):
    """question -> retrieve top-k (optionally reranked) -> (optionally) Claude answer + citations."""
    hits = do_retrieve(q.question, q.k, q.rerank)

    chunks = []
    for i, (t, m, d) in enumerate(hits):
        kind, ident = _provenance(m)
        chunks.append({
            "rank": i,
            "source": f"{kind} {ident}",
            "title": m.get("title", ""),
            "distance": round(float(d), 3),
            "text": t,
        })
    result = {
        "question": q.question, "chunks": chunks,
        "answered": False, "answer": None, "citations": [],
        "backend": q.backend, "model": None, "reranked": q.rerank,
    }

    # Route to the chosen backend. Claude needs a key; local needs none (offline).
    try:
        if q.backend == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
            result["answer"] = "(no ANTHROPIC_API_KEY set — showing retrieval only)"
        else:
            gen = llm.generate(q.question, hits, backend=q.backend)
            result.update(answered=True, answer=gen["answer"],
                          citations=gen["citations"], model=gen["model"])
    except Exception as e:
        result["answer"] = f"(generation error: {e})"

    return result


@app.get("/api/eval")
def api_eval():
    """Latest eval_ortho.py results (empty if it hasn't been run yet)."""
    path = Path(__file__).parent / "eval_results.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


@app.get("/api/corpus")
def api_corpus():
    """Descriptive stats about the corpus, from the local cache + the index."""
    import statistics

    cache = Path(config.CORPUS_CACHE)
    if not cache.exists():
        return {"abstracts": 0}

    years, journals, subs, words, samples = {}, {}, {}, [], []
    with cache.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            y = d.get("year") or "unknown"
            years[y] = years.get(y, 0) + 1
            j = d.get("journal") or "unknown"
            journals[j] = journals.get(j, 0) + 1
            st = d.get("subtopic") or "(untagged)"
            subs[st] = subs.get(st, 0) + 1
            words.append(len(d.get("text", "").split()))
            if len(samples) < 8:
                samples.append({k: d.get(k, "") for k in ("pmid", "title", "journal", "year")})

    n = len(words)
    try:
        chunks = collection().count()
    except Exception:
        chunks = 0

    numeric_years = sorted(int(y) for y in years if y.isdigit())
    year_hist = sorted(((y, c) for y, c in years.items() if y.isdigit()),
                       key=lambda x: -int(x[0]))[:12]

    return {
        "source": "live PubMed (NCBI Entrez E-utilities)",
        "query": config.PUBMED_QUERY,
        "cache_file": os.path.basename(config.CORPUS_CACHE),
        "abstracts": n,
        "chunks": chunks,
        "chunks_per_abstract": round(chunks / n, 2) if n else 0,
        "distinct_journals": len(journals),
        "year_min": numeric_years[0] if numeric_years else None,
        "year_max": numeric_years[-1] if numeric_years else None,
        "words": {
            "min": min(words) if words else 0,
            "median": int(statistics.median(words)) if words else 0,
            "mean": int(statistics.mean(words)) if words else 0,
            "max": max(words) if words else 0,
        },
        "year_hist": year_hist,
        "top_journals": sorted(journals.items(), key=lambda x: -x[1])[:12],
        "subtopics": sorted(subs.items(), key=lambda x: -x[1]),
        "samples": samples,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/eval", response_class=HTMLResponse)
def eval_page():
    return EVAL_HTML


@app.get("/corpus", response_class=HTMLResponse)
def corpus_page():
    return CORPUS_HTML


# --- Frontend --------------------------------------------------------------
# A single self-contained page: no build step, no external assets. Plain CSS
# and fetch(). Kept deliberately small so it's readable end to end.
HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Minimal RAG demo</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 system-ui, sans-serif;
         background: #0e1116; color: #e6edf3; }
  header { padding: 20px 24px; border-bottom: 1px solid #232b36; }
  .topbar { display: flex; justify-content: space-between; align-items: baseline;
            gap: 16px; flex-wrap: wrap; }
  .nav { display: flex; gap: 16px; }
  .navlink { color: #2f81f7; text-decoration: none; font-size: 14px; font-weight: 600;
             white-space: nowrap; }
  .navlink:hover { text-decoration: underline; }
  h1 { margin: 0 0 6px; font-size: 18px; }
  .badges { display: flex; gap: 8px; flex-wrap: wrap; }
  .badge { font-size: 12px; padding: 3px 8px; border-radius: 999px;
           background: #1b2430; color: #9fb2c8; border: 1px solid #232b36; }
  .badge b { color: #cfe0f3; font-weight: 600; }
  main { max-width: 820px; margin: 0 auto; padding: 24px; }
  form { display: flex; gap: 10px; align-items: center; }
  input[type=text] { flex: 1; padding: 11px 13px; border-radius: 8px;
    border: 1px solid #2a3441; background: #131a22; color: #e6edf3; font-size: 15px; }
  .kbox { display: flex; align-items: center; gap: 6px; font-size: 13px; color: #9fb2c8; }
  input[type=number] { width: 56px; padding: 8px; border-radius: 8px;
    border: 1px solid #2a3441; background: #131a22; color: #e6edf3; }
  select { padding: 8px; border-radius: 8px; border: 1px solid #2a3441;
    background: #131a22; color: #e6edf3; font-size: 14px; }
  button { padding: 11px 18px; border: 0; border-radius: 8px; cursor: pointer;
    background: #2f81f7; color: #fff; font-size: 15px; font-weight: 600; }
  button:disabled { opacity: .5; cursor: default; }
  .card { margin-top: 22px; padding: 16px 18px; border-radius: 10px;
    background: #131a22; border: 1px solid #232b36; }
  .card h2 { margin: 0 0 10px; font-size: 13px; text-transform: uppercase;
    letter-spacing: .06em; color: #7d8fa3; }
  .answer { white-space: pre-wrap; }
  .cite { font-size: 13px; color: #9fb2c8; margin-top: 6px; padding-left: 10px;
    border-left: 2px solid #2f81f7; }
  .chunk { padding: 10px 0; border-top: 1px solid #1e2731; }
  .chunk:first-of-type { border-top: 0; }
  .chunk .meta { font-size: 12px; color: #7d8fa3; display: flex; gap: 12px;
    align-items: center; margin-bottom: 4px; }
  .simbar { height: 6px; border-radius: 3px; background: #1e2731; width: 120px; overflow: hidden; }
  .simbar > span { display: block; height: 100%; background: #2f81f7; }
  .ctitle { font-size: 13px; font-weight: 600; color: #cfe0f3; margin-bottom: 3px; }
  .chunk .text { font-size: 13.5px; color: #c3d0de; }
  .muted { color: #7d8fa3; }
  .hidden { display: none; }
</style>
</head>
<body>
<header>
  <div class="topbar">
    <h1>Orthopedic RAG demo <span class="muted" style="font-weight:400">— live PubMed</span></h1>
    <nav class="nav">
      <a class="navlink" href="/corpus">Corpus &rarr;</a>
      <a class="navlink" href="/eval">Eval &rarr;</a>
    </nav>
  </div>
  <div class="badges" id="badges"></div>
</header>
<main>
  <form id="f">
    <input type="text" id="q" placeholder="Ask an orthopedic question…"
           value="What are risk factors for deep vein thrombosis after total knee arthroplasty?" autofocus>
    <div class="kbox">model
      <select id="backend">
        <option value="claude">Claude (API)</option>
        <option value="local">Local (offline)</option>
      </select></div>
    <div class="kbox">top-k <input type="number" id="k" min="1" max="20" value="5"></div>
    <label class="kbox" title="Two-stage retrieval: pull a wider candidate pool, then re-rank with a cross-encoder">
      <input type="checkbox" id="rerank"> rerank</label>
    <button id="go" type="submit">Ask</button>
  </form>

  <div class="card hidden" id="answerCard">
    <h2 id="answerHead">Answer</h2>
    <div class="answer" id="answer"></div>
    <div id="cites"></div>
  </div>

  <div class="card hidden" id="chunkCard">
    <h2 id="chunkHead">Retrieved chunks (lower distance = closer)</h2>
    <div id="chunks"></div>
  </div>
</main>

<script>
const $ = id => document.getElementById(id);

// Show what's under the hood on load.
fetch('/api/config').then(r => r.json()).then(c => {
  $('badges').innerHTML =
    `<span class="badge">embedder <b>${c.embedder}</b></span>` +
    `<span class="badge">reranker <b>${c.reranker}</b></span>` +
    `<span class="badge">Claude <b>${c.llm}</b></span>` +
    `<span class="badge">local <b>${c.local_model}</b></span>` +
    `<span class="badge">top-k <b>${c.top_k}</b></span>` +
    `<span class="badge">Claude key <b>${c.has_key ? 'set' : 'not set'}</b></span>`;
  $('k').value = c.top_k;
});

$('f').addEventListener('submit', async e => {
  e.preventDefault();
  const question = $('q').value.trim();
  if (!question) return;
  $('go').disabled = true; $('go').textContent = '…';
  try {
    const res = await fetch('/api/query', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ question, k: Number($('k').value),
                             backend: $('backend').value, rerank: $('rerank').checked })
    });
    const data = await res.json();
    render(data);
  } catch (err) {
    $('answerCard').classList.remove('hidden');
    $('answer').textContent = 'Error: ' + err;
  } finally {
    $('go').disabled = false; $('go').textContent = 'Ask';
  }
});

function render(data) {
  // Answer
  $('answerCard').classList.remove('hidden');
  $('answerHead').textContent = data.model ? ('Answer — ' + data.backend + ' · ' + data.model) : 'Answer';
  $('answer').textContent = data.answer || '';
  $('cites').innerHTML = (data.citations || [])
    .map(c => `<div class="cite">${escapeHtml(c.title)}: "${escapeHtml(c.text)}"</div>`).join('');

  // Chunks (similarity bar = 1 - cosine distance)
  $('chunkCard').classList.remove('hidden');
  $('chunkHead').textContent = data.reranked
    ? 'Retrieved chunks — reordered by cross-encoder (distance no longer monotonic)'
    : 'Retrieved chunks (lower distance = closer)';
  $('chunks').innerHTML = data.chunks.map(ch => {
    const sim = Math.max(0, 1 - ch.distance);
    return `<div class="chunk">
      <div class="meta">
        <span>#${ch.rank}</span>
        <span>${escapeHtml(ch.source)}</span>
        <span>dist ${ch.distance.toFixed(3)}</span>
        <span class="simbar"><span style="width:${(sim*100).toFixed(0)}%"></span></span>
      </div>
      ${ch.title ? `<div class="ctitle">${escapeHtml(ch.title)}</div>` : ''}
      <div class="text">${escapeHtml(ch.text)}</div>
    </div>`;
  }).join('');
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}
</script>
</body>
</html>"""


# --- /eval explainer page --------------------------------------------------
# Walks through how you evaluate a RAG system, what we actually implement here,
# and why an unlabeled domain corpus (live PubMed) needs different methods than
# the labelled PubMedQA baseline. Reads /api/eval for the latest numbers.
EVAL_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval methods & metrics — Orthopedic RAG</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.6 system-ui, sans-serif; background: #0e1116; color: #e6edf3; }
  header { padding: 20px 24px; border-bottom: 1px solid #232b36; display: flex;
           justify-content: space-between; align-items: baseline; gap: 16px; flex-wrap: wrap; }
  h1 { margin: 0; font-size: 18px; }
  a.navlink { color: #2f81f7; text-decoration: none; font-size: 14px; font-weight: 600; }
  a.navlink:hover { text-decoration: underline; }
  main { max-width: 900px; margin: 0 auto; padding: 24px; }
  h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .06em; color: #7d8fa3;
       margin: 34px 0 12px; }
  p { color: #c3d0de; }
  .lead { font-size: 16px; }
  .card { background: #131a22; border: 1px solid #232b36; border-radius: 10px; padding: 16px 18px; }
  .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
             gap: 12px; margin-top: 4px; }
  .metric { background: #0e1116; border: 1px solid #232b36; border-radius: 8px; padding: 12px 14px; }
  .metric .v { font-size: 26px; font-weight: 700; color: #cfe0f3; }
  .metric .l { font-size: 12px; color: #7d8fa3; text-transform: uppercase; letter-spacing: .05em; }
  .stamp { font-size: 12px; color: #7d8fa3; margin-top: 10px; }
  code { background: #1b2430; padding: 1px 6px; border-radius: 5px; font-size: 13px; color: #cfe0f3; }
  pre { background: #0b0f14; border: 1px solid #232b36; border-radius: 8px; padding: 12px 14px;
        overflow-x: auto; }
  .tablewrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: 13.5px; min-width: 640px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #1e2731; vertical-align: top; }
  th { color: #7d8fa3; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: .05em; }
  .s-yes { color: #3fb950; font-weight: 600; }
  .s-plan { color: #d29922; font-weight: 600; }
  .s-na  { color: #7d8fa3; }
  .catrow td { background: #10161d; color: #9fb2c8; font-weight: 600; }
  ol.steps { padding-left: 20px; } ol.steps li { margin: 6px 0; }
  .note { border-left: 3px solid #d29922; padding-left: 12px; color: #c3d0de; }
  .muted { color: #7d8fa3; }
</style>
</head>
<body>
<header>
  <h1>Evaluation — methods &amp; metrics <span class="muted" style="font-weight:400">/ orthopedic corpus</span></h1>
  <a class="navlink" href="/">&larr; back to demo</a>
</header>
<main>

  <p class="lead">The PubMedQA baseline came with expert <code>yes/no/maybe</code> labels, so we could
  score decisions directly. A <b>live PubMed pull has no labels</b> — so evaluating it needs
  different methods. This page lays out the landscape, marks what this project implements, and
  flags what a medical RAG system ultimately needs.</p>

  <h2>Latest run</h2>
  <div class="card" id="results">
    <p class="muted">Loading results…</p>
  </div>

  <h2>The evaluation landscape</h2>
  <p>Two families of metrics, measuring different failures — keep them separate so a bad number
  points at the right fix. <span class="s-yes">✅ implemented</span> ·
  <span class="s-plan">○ planned</span> · <span class="s-na">— n/a for this corpus</span></p>
  <div class="tablewrap">
  <table>
    <thead><tr><th>Method</th><th>What it catches</th><th>Metric(s)</th><th>Status</th><th>Orthopedic note</th></tr></thead>
    <tbody>
      <tr class="catrow"><td colspan="5">Retrieval quality — no LLM, cheap, $0</td></tr>
      <tr><td>Did the right doc come back?</td><td>Retriever misses the source entirely</td>
          <td>Hit@k / Recall@k</td><td class="s-yes">✅</td>
          <td>Our synthetic eval: is the source PMID in top-k?</td></tr>
      <tr><td>How high did it rank?</td><td>Right doc buried below noise</td>
          <td>MRR</td><td class="s-yes">✅</td>
          <td>Rewards putting the on-topic paper at rank 0.</td></tr>
      <tr><td>How many of k are relevant?</td><td>Padding top-k with off-topic hits</td>
          <td>Precision@k</td><td class="s-plan">○</td>
          <td>Needs multi-relevant labels (a clinician tags them).</td></tr>
      <tr><td>Graded ranking quality</td><td>Slightly-relevant ranked over highly-relevant</td>
          <td>nDCG@k, MAP</td><td class="s-plan">○</td>
          <td>Useful once we have graded relevance judgements.</td></tr>
      <tr class="catrow"><td colspan="5">Answer quality — LLM in the loop</td></tr>
      <tr><td>Is the answer grounded in the context?</td><td><b>Hallucination</b> — confident claims not in the sources</td>
          <td>Faithfulness / groundedness</td><td class="s-yes">✅</td>
          <td><b>Patient-safety critical.</b> LLM-as-judge scores % supported.</td></tr>
      <tr><td>Do the citations actually back the claim?</td><td>Real-looking cite pointing at the wrong span</td>
          <td>Citation accuracy</td><td class="s-plan">○</td>
          <td>We already emit native char-span citations — directly checkable.</td></tr>
      <tr><td>Does it abstain when the corpus can't answer?</td><td>Answering from training memory, not the docs</td>
          <td>Refusal / abstention rate</td><td class="s-plan">○</td>
          <td>We've observed good abstention on thin corpora — worth measuring.</td></tr>
      <tr><td>Does the answer address the question?</td><td>On-topic but non-responsive</td>
          <td>Answer relevance</td><td class="s-plan">○</td></tr>
      <tr><td>Correct final decision vs. a gold label</td><td>Wrong conclusion end-to-end</td>
          <td>Decision accuracy</td><td class="s-na">—</td>
          <td>Needs labels; this is the PubMedQA baseline's <code>eval.py</code>.</td></tr>
      <tr class="catrow"><td colspan="5">System &amp; human</td></tr>
      <tr><td>Is it fast / affordable enough?</td><td>Unshippable latency or cost</td>
          <td>Latency, tokens, $/query</td><td class="s-plan">○</td></tr>
      <tr><td>Would a clinician trust it?</td><td>Everything the proxies miss</td>
          <td>Expert review</td><td class="s-plan">○</td>
          <td><b>The real gold standard in medicine</b> — automated metrics are a proxy for it.</td></tr>
    </tbody>
  </table>
  </div>

  <h2>What we implement — the synthetic retrieval eval</h2>
  <p>With no labels, we manufacture a gold signal (self-supervised), which isolates the
  <b>retriever</b> with no human annotation:</p>
  <div class="card">
    <ol class="steps">
      <li>Sample N abstracts from the corpus.</li>
      <li>Ask Claude to write one specific question that abstract answers.</li>
      <li>Retrieve top-k against the <i>whole</i> corpus for that question.</li>
      <li>Check whether the abstract's own PMID comes back → <b>Hit@k</b>; its rank → <b>MRR</b>.</li>
    </ol>
    <p style="margin-bottom:0">Optionally (<code>--faithfulness</code>) run the full RAG answer and have Claude
    judge what fraction is supported by the retrieved context (LLM-as-judge groundedness).</p>
  </div>
  <p style="margin-top:14px">Run it:</p>
  <pre>python eval_ortho.py --n 40                              # easy retrieval eval
python eval_ortho.py --n 60 --hard                      # paraphrased (harder) questions
python eval_ortho.py --n 60 --hard --compare            # paired A/B: rerank off vs on
python eval_ortho.py --n 40 --hard --compare --faithfulness   # + answer grounding</pre>

  <p class="note"><b>Honest caveat.</b> A generated question may be answerable by <i>other</i>
  abstracts too, so checking only the source PMID slightly under-counts a genuinely good
  retriever — treat Hit@k here as a <b>lower bound</b>, not ground truth. Fixing this properly
  means graded relevance judgements, which is where a clinician-in-the-loop comes in.</p>

  <h2>Two-stage retrieval &amp; the reranker A/B</h2>
  <p>Retrieval runs in two stages: the <b>bi-encoder</b> (bge-small) scores query and chunk
  independently — fast enough to search all 34k chunks — then an optional <b>cross-encoder
  reranker</b> (bge-reranker-base) reads each <code>(question, chunk)</code> pair together and
  re-ranks the top candidates. <code>--hard</code> makes the questions paraphrase away the
  abstract's exact terms (a stiffer test); <code>--compare</code> evaluates the <b>same</b>
  questions with and without the reranker so the delta is the reranker's true effect, not
  question-sampling noise. The "Latest run" panel above renders that paired table when present.</p>

  <h2>Planned metrics — and why they matter</h2>
  <p>Two metrics are marked <span class="s-plan">○ planned</span> above because they're the
  most valuable next additions for a <i>medical</i> RAG system. Here's what each measures and how
  we'd build it.</p>

  <div class="card" style="margin-bottom:14px">
    <p style="margin-top:0"><b>1. Citation accuracy</b> — does the cited span actually back the claim?</p>
    <p>Every answer here already ships <b>native character-span citations</b>: each grounded sentence
    points at an exact chunk and the exact characters within it. Citation accuracy asks the next
    question — when the model says "<i>[PMID 12345] found a 2% infection rate</i>," does that cited span
    <i>really</i> say that?</p>
    <p>This is <b>distinct from faithfulness</b>. Faithfulness scores whether the answer as a whole is
    supported by the retrieved context; citation accuracy checks each individual pointer. An answer can
    be broadly grounded yet cite the <i>wrong</i> sentence — which, in medicine, is the difference
    between "trust me" and "here is the exact line in this paper." Verifiable citations are what let a
    clinician audit an answer instead of taking it on faith.</p>
    <p style="margin-bottom:0"><b>How we'd measure it:</b> for each cited claim, run an entailment check
    — an LLM judge (or a natural-language-inference model) decides whether the cited span <i>entails</i>
    the claim. Report the fraction of citations that genuinely support their claim. Cheap, because we
    already have the exact spans to check.</p>
  </div>

  <div class="card">
    <p style="margin-top:0"><b>2. Multi-relevant retrieval</b> — Precision@k &amp; nDCG, not just "the one source."</p>
    <p>Our current Hit@k asks only "<i>did the single source PMID come back?</i>" But on a paraphrased
    question, <b>many</b> of the 34k abstracts answer it equally well — so Hit@k under-credits a good
    retriever, and (as the reranker A/B showed) it's <b>blind to reranking</b>, which reshuffles
    <i>among</i> relevant docs. A multi-relevant metric fixes the blind spot.</p>
    <p>Instead of checking one PMID, we judge <b>each</b> retrieved chunk for relevance — graded
    <code>0 / 1 / 2</code> (irrelevant / partial / on-point) by an LLM judge — then compute:</p>
    <ul style="margin:6px 0">
      <li><b>Precision@k</b> — what fraction of the top-k are actually relevant (catches padding the
      results with off-topic hits).</li>
      <li><b>nDCG@k</b> — graded ranking quality: rewards putting <i>highly</i>-relevant docs at the top,
      not merely somewhere in the k.</li>
    </ul>
    <p style="margin-bottom:0">This is the metric where the reranker <b>should</b> visibly win, because
    it credits surfacing <i>any</i> good abstract rather than one specific source. <b>Cost:</b> ~k judge
    calls per question — more than Hit@k's zero, which is why it's a deliberate next step rather than the
    default.</p>
  </div>

  <h2>Orthopedic-specific considerations</h2>
  <div class="card">
    <ul>
      <li><b>Faithfulness &gt; fluency.</b> A confident, wrong orthopedic answer is a safety risk — groundedness and abstention matter more here than in a general chatbot.</li>
      <li><b>Domain jargon stresses the embedder.</b> Terms like <i>arthroplasty</i>, <i>osteotomy</i>, <i>TKA/THA</i> — a low Hit@k would point at retrieval (bigger/fine-tuned embedder, or a reranker), not the prompt.</li>
      <li><b>Currency.</b> Guidelines change; because we pull <i>live</i> PubMed, corpus freshness is itself an eval axis (re-ingest cadence).</li>
      <li><b>Clinician validation is the ceiling.</b> Every automated metric here is a cheap proxy for "would an orthopedic surgeon trust this?" — the honest north star.</li>
    </ul>
  </div>

  <p class="muted" style="margin:30px 0">Two metrics on purpose: low Hit@k → fix retrieval (chunking, embedder, reranker, top-k). High Hit@k but low faithfulness → fix the prompt/model, not the retriever.</p>
</main>

<script>
fetch('/api/eval').then(r => r.json()).then(d => {
  const el = document.getElementById('results');
  if (!d || !d.n) {
    el.innerHTML = "<p class='muted'>No run yet. Run <code>python eval_ortho.py --n 60 --hard --compare --faithfulness</code> and refresh.</p>";
    return;
  }
  const pct = x => (x * 100).toFixed(0) + '%';
  const mode = d.question_mode === 'hard' ? 'hard (paraphrased)' : 'easy (specific)';
  const stamp = `<div class="stamp">corpus <code>${d.corpus}</code> · ${d.corpus_chunks.toLocaleString()} chunks · `
    + `${d.n} questions · top-${d.k} · question mode: <b>${mode}</b> · judge ${d.model} · ${d.generated_at}</div>`;

  // Paired A/B (--compare): same questions retrieved with vs without the reranker.
  if (d.compare) {
    const off = d.compare.off, on = d.compare.on;
    let rows =
      `<tr><td>Hit@${d.k}</td><td>${pct(off.hit_at_k)}</td><td>${pct(on.hit_at_k)}</td></tr>` +
      `<tr><td>MRR</td><td>${off.mrr.toFixed(3)}</td><td>${on.mrr.toFixed(3)}</td></tr>`;
    if (off.faithfulness != null)
      rows += `<tr><td>Faithfulness</td><td>${off.faithfulness.toFixed(1)}%</td><td>${on.faithfulness.toFixed(1)}%</td></tr>`;
    el.innerHTML =
      `<p style="margin-top:0">Paired A/B — the <b>same</b> questions retrieved with and without the `
      + `stage-2 cross-encoder reranker, so any gap is the reranker's true effect, not question-sampling noise.</p>`
      + `<div class="tablewrap"><table style="min-width:auto">`
      + `<thead><tr><th>metric</th><th>rerank OFF</th><th>rerank ON</th></tr></thead>`
      + `<tbody>${rows}</tbody></table></div>`
      + (off.faithfulness != null
          ? `<p class="note" style="margin-top:14px">Typical result on hard questions: the reranker is roughly `
            + `<b>neutral on source-recall</b> (Hit@k/MRR) but <b>lifts faithfulness</b>. On a dense corpus where many `
            + `abstracts answer a paraphrased question, single-source Hit@k is the wrong lens for a reranker — it `
            + `reshuffles <i>among</i> relevant docs, which shows up in answer grounding, not source recall.</p>`
          : '')
      + stamp;
    return;
  }

  // Single-config run.
  const m = (v, l) => `<div class="metric"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  let cards = m(pct(d.hit_at_k), `Hit@${d.k}`) + m(d.mrr.toFixed(3), 'MRR');
  if (d.faithfulness !== null && d.faithfulness !== undefined)
    cards += m(d.faithfulness.toFixed(0) + '%', 'Faithfulness');
  cards += m(d.rerank ? 'on' : 'off', 'reranker') + m(d.n, 'questions');
  el.innerHTML = `<div class="metrics">${cards}</div>` + stamp;
});
</script>
</body>
</html>"""


# --- /corpus review page ---------------------------------------------------
# Descriptive stats about the knowledge base: size, chunking, publication-year
# distribution, top journals, abstract length, the build query, and samples.
# Reads /api/corpus (computed from the local cache + the live index).
CORPUS_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Corpus — Orthopedic RAG</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.6 system-ui, sans-serif; background: #0e1116; color: #e6edf3; }
  header { padding: 20px 24px; border-bottom: 1px solid #232b36; display: flex;
           justify-content: space-between; align-items: baseline; gap: 16px; flex-wrap: wrap; }
  h1 { margin: 0; font-size: 18px; }
  .nav { display: flex; gap: 16px; }
  a.navlink { color: #2f81f7; text-decoration: none; font-size: 14px; font-weight: 600; }
  a.navlink:hover { text-decoration: underline; }
  main { max-width: 900px; margin: 0 auto; padding: 24px; }
  h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .06em; color: #7d8fa3; margin: 34px 0 12px; }
  p { color: #c3d0de; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
  .tile { background: #131a22; border: 1px solid #232b36; border-radius: 10px; padding: 14px 16px; }
  .tile .v { font-size: 26px; font-weight: 700; color: #cfe0f3; }
  .tile .l { font-size: 12px; color: #7d8fa3; text-transform: uppercase; letter-spacing: .05em; }
  .card { background: #131a22; border: 1px solid #232b36; border-radius: 10px; padding: 16px 18px; }
  .bar { display: grid; grid-template-columns: 90px 1fr 54px; align-items: center; gap: 10px; margin: 5px 0; font-size: 13px; }
  .bar .lab { color: #9fb2c8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .bar .track { background: #0e1116; border-radius: 4px; height: 16px; overflow: hidden; }
  .bar .fill { height: 100%; background: #2f81f7; }
  .bar .n { text-align: right; color: #7d8fa3; font-variant-numeric: tabular-nums; }
  .jbar { grid-template-columns: 1fr 44px; }
  code { background: #1b2430; padding: 1px 6px; border-radius: 5px; font-size: 13px; color: #cfe0f3; }
  pre { background: #0b0f14; border: 1px solid #232b36; border-radius: 8px; padding: 12px 14px; overflow-x: auto;
        font-size: 13px; color: #c3d0de; white-space: pre-wrap; }
  .tablewrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: 13.5px; min-width: 640px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #1e2731; vertical-align: top; }
  th { color: #7d8fa3; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: .05em; }
  td a { color: #2f81f7; text-decoration: none; } td a:hover { text-decoration: underline; }
  .muted { color: #7d8fa3; }
</style>
</head>
<body>
<header>
  <h1>Corpus <span class="muted" style="font-weight:400">/ orthopedic PubMed</span></h1>
  <nav class="nav">
    <a class="navlink" href="/">&larr; demo</a>
    <a class="navlink" href="/eval">Eval &rarr;</a>
  </nav>
</header>
<main id="body">
  <p class="muted">Loading corpus stats…</p>
</main>

<script>
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

fetch('/api/corpus').then(r => r.json()).then(d => {
  const body = document.getElementById('body');
  if (!d.abstracts) {
    body.innerHTML = "<p class='muted'>No corpus cache found. Run <code>python download_corpus.py</code> then <code>python ingest_pubmed.py</code>.</p>";
    return;
  }
  const tile = (v, l) => `<div class="tile"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  const num = x => x.toLocaleString();

  // stat tiles
  let html = `<p>A knowledge base built from <b>live PubMed</b> abstracts, sorted by publication date — so it reflects the most recent orthopedic literature.</p>`;
  html += '<div class="tiles">' +
    tile(num(d.abstracts), 'abstracts') +
    tile(num(d.chunks), 'chunks') +
    tile(d.chunks_per_abstract, 'chunks / abstract') +
    tile(num(d.words.median), 'median words') +
    tile(num(d.distinct_journals), 'journals') +
    tile((d.year_min && d.year_max) ? (d.year_min + '–' + d.year_max) : '—', 'year span') +
    '</div>';

  // subtopic composition (the diversity view)
  if (d.subtopics && d.subtopics.length) {
    const smax = Math.max(...d.subtopics.map(x => x[1]), 1);
    html += '<h2>Subtopic composition</h2><div class="card">' +
      d.subtopics.map(([s, c]) =>
        `<div class="bar"><span class="lab">${esc(s)}</span>
          <span class="track"><span class="fill" style="width:${(c/smax*100).toFixed(1)}%"></span></span>
          <span class="n">${num(c)}</span></div>`).join('') +
      '</div>';
  }

  // year distribution
  const ymax = Math.max(...d.year_hist.map(x => x[1]), 1);
  html += '<h2>Publication years</h2><div class="card">' +
    d.year_hist.map(([y, c]) =>
      `<div class="bar"><span class="lab">${y}</span>
        <span class="track"><span class="fill" style="width:${(c/ymax*100).toFixed(1)}%"></span></span>
        <span class="n">${num(c)}</span></div>`).join('') +
    '</div>';

  // top journals
  const jmax = Math.max(...d.top_journals.map(x => x[1]), 1);
  html += '<h2>Top journals</h2><div class="card">' +
    d.top_journals.map(([j, c]) =>
      `<div class="bar jbar"><span class="lab" title="${esc(j)}">${esc(j)}</span>
        <span class="n">${num(c)}</span>
        <span class="track" style="grid-column:1/3;margin-top:2px"><span class="fill" style="width:${(c/jmax*100).toFixed(1)}%"></span></span></div>`).join('') +
    '</div>';

  // abstract length
  html += '<h2>Abstract length (words)</h2><div class="tiles">' +
    tile(num(d.words.min), 'min') + tile(num(d.words.median), 'median') +
    tile(num(d.words.mean), 'mean') + tile(num(d.words.max), 'max') + '</div>';

  // build recipe
  html += '<h2>How it was built</h2><div class="card"><p style="margin-top:0">Source: <b>' +
    esc(d.source) + '</b> → cached locally to <code>' + esc(d.cache_file) +
    '</code> → chunked &amp; embedded into the index.</p><p style="margin-bottom:6px">Search query:</p><pre>' +
    esc(d.query) + '</pre></div>';

  // samples
  html += '<h2>Sample records</h2><div class="tablewrap"><table>' +
    '<thead><tr><th>PMID</th><th>Year</th><th>Journal</th><th>Title</th></tr></thead><tbody>' +
    d.samples.map(s =>
      `<tr><td><a href="https://pubmed.ncbi.nlm.nih.gov/${esc(s.pmid)}/" target="_blank" rel="noopener">${esc(s.pmid)}</a></td>
        <td>${esc(s.year)}</td><td>${esc(s.journal)}</td><td>${esc(s.title)}</td></tr>`).join('') +
    '</tbody></table></div>';

  body.innerHTML = html;
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    # host=0.0.0.0 so it's reachable from your other machine (or an SSH tunnel).
    uvicorn.run(app, host="0.0.0.0", port=8022)
