"""Hermes agent-trader bridge.

Exposes a *read-only-for-now* slice of Auramaur's plumbing (portfolio,
accounting) as an MCP server that the Hermes agent consumes over stdio. This is
a separate paradigm experiment — a persistent LLM agent day-trader running
against the SAME plumbing as Auramaur, on its OWN isolated ``agent.db``, so the
two books can be compared head-to-head (strategy_source="agent_hermes").

Safety: this process never holds live credentials and forces paper mode
(``AURAMAUR_LIVE=false``); the live-order gates in CLAUDE.md are therefore
unreachable from here by construction, not by policy. See ``server.py``.
"""
