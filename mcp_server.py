"""MCP server — expose the access-controlled retriever to ANY MCP client.

MCP (Model Context Protocol) is the open standard for connecting AI apps to
external tools. `agent.py` wired the retrieval tools into our OWN loop; this makes
the same tools usable by any MCP client — Claude Desktop, Claude Code, Cursor —
so the model in those apps can call our orthopedic retriever without a bespoke
integration. Write the tool once here; every MCP-compatible agent can use it.

    RAG_MCP_USER=clinician python mcp_server.py     # runs over stdio

Register it with an MCP client (e.g. Claude Desktop's config):

    {
      "mcpServers": {
        "orthopedic-rag": {
          "command": "/abs/path/.venv/bin/python",
          "args": ["/abs/path/mcp_server.py"],
          "env": { "RAG_MCP_USER": "clinician" }
        }
      }
    }

SECURITY (same boundary as agent.py, one layer out): the principal is bound HERE,
at server launch, from RAG_MCP_USER — it is NOT a tool parameter. The client's
model only ever sees `search_corpus(query, k)`, so it cannot set or raise its own
clearance. Defaults to the least-privileged `public` if unset (fail closed).
Every tool call is written to the audit log.
"""
import os

from mcp.server.fastmcp import FastMCP

import access
import audit
import config
import federated
from agent import _format_hits          # reuse the same citeable formatting
from query import get_collection, retrieve

# Bind the principal at launch. Unknown / unset -> least privilege.
USER = access.USERS.get(os.environ.get("RAG_MCP_USER", "public"), access.USERS["public"])

mcp = FastMCP("orthopedic-rag")

_collection = None


def _col():
    global _collection
    if _collection is None:
        _collection = get_collection(config.COLLECTION_ORTHO)
    return _collection


@mcp.tool()
def search_corpus(query: str, k: int = 5) -> str:
    """Search the orthopedic literature for passages relevant to a query.

    Returns the best-matching abstract chunks with their PMIDs. Results are
    automatically restricted to what this server's bound user is authorized to
    see — you cannot widen that access from here."""
    hits = retrieve(_col(), query, k=min(int(k), 10), where=access.build_where(USER))
    audit.record_retrieval(USER, query, hits, {"tool": "mcp.search_corpus"}, backend="mcp")
    return _format_hits(hits)


@mcp.tool()
def federated_search(query: str, k: int = 5) -> str:
    """Search across every independent data silo the bound user is cleared to
    query, then merge the results (each tagged with its source silo)."""
    hits, _report = federated.federated_retrieve(query, USER, k=min(int(k), 10))
    audit.record_retrieval(USER, query, hits, {"tool": "mcp.federated_search"}, backend="mcp")
    return _format_hits(hits)


@mcp.tool()
def access_context() -> str:
    """Report the clearance and need-to-know this server is bound to, so the
    client knows the scope of what search results can contain."""
    return USER.describe()


if __name__ == "__main__":
    mcp.run()  # stdio transport
