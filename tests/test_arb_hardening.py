"""Tests for the 2026-06-12 arb hardening.

- arbitrage.enabled actually gates the arb + correlation tasks (it was a
  dead flag; disabling it via config did nothing while the book locked a
  -3% cross-attempt pair).
- _execute_internal_arb skips markets where we already hold inventory: a
  partially-filled prior attempt must not be "completed" at the next
  cycle's worse prices (observed live: 0.82 + 0.21 = $1.03 for a $1.00
  payout).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from auramaur.components import Components
import pytest

from auramaur.bot import AuramaurBot
from auramaur.exchange.models import Market


def _bare_bot(*, held_row, is_live=True):
    bot = AuramaurBot.__new__(AuramaurBot)
    db = AsyncMock()
    db.fetchone = AsyncMock(return_value=held_row)
    bot._components = Components({"db": db, "alerts": AsyncMock()})
    bot.settings = SimpleNamespace(is_live=is_live)
    return bot, db


def _opp(market_id="m1"):
    market = Market(id=market_id, question="Will X happen?",
                    outcome_yes_price=0.17, outcome_no_price=0.82)
    return SimpleNamespace(market_a=market, exchange_a="polymarket",
                           expected_profit_pct=1.0, spread=0.01,
                           price_a=0.17, price_b=0.82)


@pytest.mark.asyncio
async def test_internal_arb_skips_market_with_existing_inventory():
    bot, db = _bare_bot(held_row={"1": 1})
    engine = SimpleNamespace(_get_available_cash=AsyncMock(return_value=100.0))

    await bot._execute_internal_arb(_opp(), risk_manager=AsyncMock(),
                                    engines={"polymarket": engine})

    # Early-returned on the inventory guard: cash/risk were never consulted.
    engine._get_available_cash.assert_not_awaited()
    sql, params = db.fetchone.call_args.args
    assert "FROM portfolio" in sql and params == ("m1", 0)  # live -> is_paper=0


@pytest.mark.asyncio
async def test_internal_arb_inventory_guard_respects_paper_mode():
    bot, db = _bare_bot(held_row={"1": 1}, is_live=False)
    engine = SimpleNamespace(_get_available_cash=AsyncMock(return_value=100.0))
    await bot._execute_internal_arb(_opp(), risk_manager=AsyncMock(),
                                    engines={"polymarket": engine})
    _, params = db.fetchone.call_args.args
    assert params == ("m1", 1)  # paper -> is_paper=1


def test_arbitrage_enabled_gates_arb_tasks_in_source():
    """Structural check: both arb task creations sit behind the flag.

    (Task creation happens deep in start(); rather than booting a bot, pin
    the gating in source so a refactor can't silently drop it again.)"""
    import inspect
    import auramaur.bot as botmod
    src = inspect.getsource(botmod)
    arb_idx = src.index('name="arb_scanner"')
    corr_idx = src.index('name="correlation"')
    for idx in (arb_idx, corr_idx):
        window = src[max(0, idx - 400):idx]
        assert "self.settings.arbitrage.enabled" in window
