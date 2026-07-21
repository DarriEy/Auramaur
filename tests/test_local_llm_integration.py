"""Optional round-trip against a real local Ollama.

Skipped unless AURAMAUR_OLLAMA_TEST is set (never runs in CI). Requires
Ollama serving the configured model on 127.0.0.1:11434.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from auramaur.nlp.local_llm import LocalLLMClient
from config.settings import LocalLLMConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get("AURAMAUR_OLLAMA_TEST"),
    reason="set AURAMAUR_OLLAMA_TEST=1 with Ollama serving to run")


async def test_real_round_trip():
    settings = SimpleNamespace(local_llm=LocalLLMConfig(enabled=True))
    client = LocalLLMClient(settings, None)
    try:
        assert await client.health()
        result = await client.generate_json(
            'Return {"answer": 4} for 2+2.',
            schema={"type": "object",
                    "properties": {"answer": {"type": "number"}},
                    "required": ["answer"]},
            purpose="health",
            max_tokens=50)
        assert result is not None
        assert result.get("answer") == 4
    finally:
        await client.close()
