"""Pluggable LLM layer — the SAME RAG pipeline, Claude OR a local model.

Why this exists: in a classified / air-gapped enclave there is no internet, so
the answer step cannot call Claude's API. Putting generation behind one
interface lets the identical retrieval pipeline run on:

  - "claude" — the frontier API, with native character-span citations
  - "local"  — a model on the GPU via Ollama ($0, fully offline)

switchable with a single argument. This is the air-gap story made real: same
corpus, same retrieval, swap only the generator.

    from llm import generate
    result = generate(question, hits, backend="claude")   # or "local"
    # -> {"answer": str, "citations": [{title,text}], "model": str, "backend": str}
"""
import json
import urllib.request

import config
import query

BACKENDS = ("claude", "local")


def generate(question: str, hits, backend: str = "claude") -> dict:
    """Route to the chosen backend and return a normalized result dict."""
    if backend == "local":
        return _local(question, hits)
    return _claude(question, hits)


def _claude(question: str, hits) -> dict:
    """Frontier path: Claude with native document-block citations."""
    resp = query.answer(question, hits)
    text_parts, cites = [], []
    for block in resp.content:
        if block.type != "text":
            continue
        text_parts.append(block.text)
        for c in block.citations or []:
            cites.append({"title": c.document_title, "text": c.cited_text.strip()})
    return {"answer": "".join(text_parts), "citations": cites,
            "model": config.CLAUDE_MODEL, "backend": "claude"}


def _local(question: str, hits) -> dict:
    """Air-gap path: a local Ollama model. No native-citation API, so we number
    the sources and ask the model to cite [n]; the retrieved chunks (with PMIDs)
    are surfaced separately as the evidence."""
    context = "\n\n".join(
        f"[{i}] (PMID {m.get('pmid', '?')}) {t}" for i, (t, m, _d) in enumerate(hits)
    )
    # Prompt note: the earlier wording ended "If the sources do not contain the
    # answer, say so plainly." llama3.1 ignored it, but instruction-following
    # models take it literally and open with a paragraph hedging about the
    # sources' limitations before answering. Abstention is still wanted — it's a
    # safety property on a medical corpus — so it's kept, but narrowed to the
    # case where the sources genuinely don't address the question.
    prompt = (
        "You are a careful clinical assistant. Answer the question using ONLY the "
        "numbered sources below, citing each claim as [n]. Report what the sources "
        "DO establish, directly and without preamble about their limitations. Only "
        "if the sources genuinely do not address the question at all, say so plainly.\n\n"
        f"Sources:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    body = json.dumps({
        "model": config.LOCAL_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": config.MAX_TOKENS},
    }).encode()
    req = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.loads(r.read())
    return {"answer": out.get("response", "").strip(), "citations": [],
            "model": config.LOCAL_MODEL, "backend": "local"}
