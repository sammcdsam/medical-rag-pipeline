"""The security-signal tests: the model must never be able to set its own access.

Both the agent and the MCP server bind the user's clearance in the harness. These
assert that the tool schemas Claude sees expose ONLY a query + count — no way to
name a user, clearance, or classification — and that the MCP server fails closed
to the least-privileged principal when none is specified.
"""
import importlib

import agent


def test_agent_tools_do_not_expose_access_controls():
    # Benign tool params only (query text, result count, a PMID to look up) — and
    # crucially NONE that would let the model set or widen its own access.
    allowed = {"query", "k", "pmid"}
    forbidden = {"user", "clearance", "classification", "compartment", "principal", "role", "silo"}
    for tool in agent.TOOLS:
        props = set(tool["input_schema"]["properties"])
        assert props <= allowed, f"{tool['name']} exposes unexpected params: {props - allowed}"
        assert not (props & forbidden), f"{tool['name']} exposes an access control to the model"


def test_mcp_server_fails_closed_to_public(monkeypatch):
    # With no RAG_MCP_USER set, the bound principal must be the least-privileged one.
    monkeypatch.delenv("RAG_MCP_USER", raising=False)
    import mcp_server
    importlib.reload(mcp_server)
    assert mcp_server.USER.name == "public"


def test_mcp_server_binds_the_requested_principal(monkeypatch):
    monkeypatch.setenv("RAG_MCP_USER", "clinician")
    import mcp_server
    importlib.reload(mcp_server)
    assert mcp_server.USER.name == "clinician"
