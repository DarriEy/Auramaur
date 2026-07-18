"""OpenAI Responses adapter and ETF intelligence-cell isolation."""

import json
from types import SimpleNamespace

import pytest

from auramaur.db.database import Database
from auramaur.nlp.openai_etf import OpenAIETFAnalyzer


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def json(self):
        body = {"probability": 0.64, "confidence": "MEDIUM_HIGH",
                "thesis": "breadth is improving", "key_risks": ["inflation"]}
        return {"id": "resp_test", "usage": {"input_tokens": 100,
                "output_tokens": 20, "total_tokens": 120},
                "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": json.dumps(body)}]}]}


class FakeSession:
    def __init__(self):
        self.payload = None

    def post(self, url, json):
        self.payload = json
        return FakeResponse(json)


@pytest.mark.asyncio
async def test_responses_request_is_structured_and_model_specific():
    analyzer = OpenAIETFAnalyzer("test-key", "gpt-5.6-terra", "medium")
    session = FakeSession()
    analyzer._get_session = lambda: _async_value(session)
    market = SimpleNamespace(question="Will SPY rise?", description="Five-day horizon")
    result = await analyzer.analyze(market, [])

    assert result.probability == 0.64
    assert result.confidence == "MEDIUM_HIGH"
    assert session.payload["model"] == "gpt-5.6-terra"
    assert session.payload["reasoning"] == {"effort": "medium"}
    assert session.payload["text"]["format"]["type"] == "json_schema"
    assert session.payload["text"]["format"]["strict"] is True
    assert session.payload["store"] is False


async def _async_value(value):
    return value


@pytest.mark.asyncio
async def test_missing_key_fails_closed_without_creating_session():
    analyzer = OpenAIETFAnalyzer("", "gpt-5.6-luna", "low")
    market = SimpleNamespace(question="?", description="")
    assert await analyzer.analyze(market, []) is None


@pytest.mark.asyncio
async def test_every_api_attempt_records_usage_and_status():
    db = Database(":memory:")
    await db.connect()
    analyzer = OpenAIETFAnalyzer(
        "test-key", "gpt-5.6-sol", "high", db=db, model_alias="sol",
        input_cost_per_million=2.0, output_cost_per_million=10.0)
    session = FakeSession()
    analyzer._get_session = lambda: _async_value(session)
    market = SimpleNamespace(question="Will SPY rise?", description="Five days")
    assert await analyzer.analyze(market, []) is not None
    row = await db.fetchone("SELECT * FROM ibkr_etf_openai_attempts")
    assert row["model_alias"] == "sol"
    assert row["status"] == "completed"
    assert row["response_id"] == "resp_test"
    assert row["total_tokens"] == 120
    assert row["cost_usd"] == pytest.approx(0.0004)
    assert row["finished_at"] is not None
    fee = await db.fetchone("SELECT kind, pnl FROM ibkr_etf_ledger")
    assert dict(fee) == {"kind": "intelligence", "pnl": pytest.approx(-0.0004)}
    await db.close()
