"""Citation graph over the full-text corpus — the "graph database" half of hybrid retrieval.

The vector store answers "what's semantically similar?". A citation graph answers a
different question: "how are papers CONNECTED — which cites which, and what is the
field built on?" We get it for free from the reference lists captured in the PMC
full text (pmc.py): every article's references carry the cited paper's PMID, so:

    corpus article  --cites-->  referenced paper

Our corpus is ~500 articles, but their references point at thousands of OTHER
papers (mostly outside the corpus). So the graph surfaces FOUNDATIONAL works
(highly cited by our corpus) even when those works aren't in the corpus itself —
something semantic search cannot do.

    python citation_graph.py        # build + print stats and the most-cited papers
"""
from functools import lru_cache

import networkx as nx

import pmc


@lru_cache(maxsize=1)
def build_graph() -> nx.DiGraph:
    """Directed citation network from the full-text reference lists (cached per process)."""
    graph = nx.DiGraph()
    for doc in pmc.load_fulltext():
        src = doc["pmid"]
        graph.add_node(src, title=doc.get("title", ""), in_corpus=True, year=doc.get("year", ""))
        for ref in doc.get("references", []):
            dst = ref.get("pmid")
            if not dst:
                continue
            if dst not in graph:
                graph.add_node(dst, title=ref.get("title", ""), in_corpus=False)
            elif not graph.nodes[dst].get("title") and ref.get("title"):
                graph.nodes[dst]["title"] = ref["title"]   # backfill a title if we learn one
            graph.add_edge(src, dst)
    return graph


def _node(graph, pmid: str) -> dict:
    n = graph.nodes[pmid]
    return {"pmid": pmid, "title": n.get("title", ""), "in_corpus": n.get("in_corpus", False)}


def cited_by(pmid: str) -> list[dict]:
    """Papers that `pmid` cites (its outgoing references)."""
    graph = build_graph()
    if pmid not in graph:
        return []
    return [_node(graph, n) for n in graph.successors(pmid)]


def who_cites(pmid: str) -> list[dict]:
    """Corpus papers that cite `pmid` (incoming edges) — who builds on this work."""
    graph = build_graph()
    if pmid not in graph:
        return []
    return [_node(graph, n) for n in graph.predecessors(pmid)]


def influential(n: int = 15) -> list[dict]:
    """Most-cited papers in the network (in-degree = how many corpus papers cite it).

    A simple, interpretable 'foundational works' ranking: the papers the corpus
    leans on most. Includes external references (not in our corpus), which is the
    point — the field's cornerstones aren't necessarily in any one collection.
    """
    graph = build_graph()
    ranked = sorted(graph.nodes, key=lambda x: graph.in_degree(x), reverse=True)
    out = []
    for node in ranked[:n]:
        info = _node(graph, node)
        info["citations"] = graph.in_degree(node)
        out.append(info)
    return out


def main() -> None:
    graph = build_graph()
    corpus = [n for n in graph if graph.nodes[n].get("in_corpus")]
    external = graph.number_of_nodes() - len(corpus)
    print(f"Citation graph: {graph.number_of_nodes():,} papers, {graph.number_of_edges():,} citation edges")
    print(f"  {len(corpus)} in corpus · {external:,} referenced (external)\n")
    print("Most-cited papers in the network (foundational works the corpus leans on):")
    for p in influential(15):
        loc = "in-corpus" if p["in_corpus"] else "external "
        print(f"  {p['citations']:>3}x  PMID {p['pmid']:<9} [{loc}]  {p['title'][:66]}")


if __name__ == "__main__":
    main()
