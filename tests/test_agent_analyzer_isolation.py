"""Depth-agent CLI isolation: neutral cwd + restricted tools.

From the repo root, `claude -p` loads CLAUDE.md and the project auto-memory —
and the auto-memory channel is WRITABLE, so a trading call can append
unreviewed beliefs to the context that seeds every future session and its own
future calls (observed 2026-07-09: an overnight depth-agent cycle wrote its
market analysis into the operator's memory index). The subprocess must run
from the system temp dir with only WebSearch/WebFetch enabled; the agent's
sanctioned state is world_model.json, passed explicitly in the prompt.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

import pytest

import auramaur.strategy.agent_analyzer as aa
from auramaur.strategy.agent_analyzer import AgentAnalyzer


@pytest.mark.asyncio
async def test_cli_call_neutral_cwd_and_restricted_tools(monkeypatch):
    captured = {}

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b"ok", b""

    async def fake_exec(*cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(aa.asyncio, "create_subprocess_exec", fake_exec)

    settings = MagicMock()
    settings.nlp.model = "claude-opus-4-8"
    settings.nlp.effort_tool_use = "medium"
    settings.nlp.daily_claude_call_budget = 0  # unlimited: skip budget gate
    analyzer = AgentAnalyzer(settings=settings, db=MagicMock())

    out = await analyzer._call_claude_agent("prompt")
    assert out == "ok"
    cmd = captured["cmd"]
    assert captured["kwargs"]["cwd"] == tempfile.gettempdir()
    assert "--allowedTools" in cmd
    assert cmd[cmd.index("--allowedTools") + 1] == "WebSearch,WebFetch"
