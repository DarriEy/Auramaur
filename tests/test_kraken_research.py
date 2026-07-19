import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from auramaur.components import Components
from auramaur.db.database import Database
from auramaur.exchange.kraken import KrakenSpotClient
from auramaur.exchange.models import OrderSide
from auramaur.research.kraken_eval import (
    Bar, CostModel, Decision, Metrics, PortfolioEvaluator,
    fit_residual_betas, graduation, residual_mean_reversion_signal,
    volatility_breakout_signal, walk_forward,
)
from auramaur.treasury.kraken_pillar import KrakenPillar


def bars(prices, start=0):
    return [Bar(start+i*3600, p, p*1.01, p*.99, p, 100) for i, p in enumerate(prices)]


@pytest.mark.asyncio
async def test_ticker_price_and_bid_ask_parse_distinct_fields():
    client = KrakenSpotClient(SimpleNamespace())
    client._public = AsyncMock(return_value={"XXBTZUSD": {
        "c": ["101.0", "1"], "b": ["100.0", "1", "1"], "a": ["102.0", "1", "1"]}})
    assert await client.get_price("XBTUSD") == 101.0
    assert await client.get_bid_ask("XBTUSD") == (100.0, 102.0)


def test_balance_cli_runs_coroutine_once():
    import sys
    import types
    if "fcntl" not in sys.modules:
        fcntl = types.ModuleType("fcntl")
        fcntl.LOCK_EX, fcntl.LOCK_NB, fcntl.LOCK_UN = 2, 4, 8
        fcntl.flock = lambda *args: None
        sys.modules["fcntl"] = fcntl
    from auramaur.cli.kraken import kraken
    client = MagicMock()
    client.get_balance = AsyncMock(return_value={"USDC": 12.0})
    client.close = AsyncMock()
    with patch("auramaur.exchange.kraken.KrakenSpotClient", return_value=client):
        result = CliRunner().invoke(kraken, ["balance"])
    assert result.exit_code == 0, result.output
    client.get_balance.assert_awaited_once()
    client.close.assert_awaited_once()


def test_research_cli_accepts_event_and_orderbook_fields(tmp_path):
    from auramaur.cli.kraken import kraken
    rows = [[i, 100+i, 101+i, 99+i, 100+i, 10, 99.5+i, 100.5+i,
             .8 if i == 2 else 0, .3] for i in range(5)]
    path = tmp_path / "ohlc.json"
    path.write_text(json.dumps({"XBTUSDC": rows, "ETHUSDC": rows}), encoding="utf-8")
    result = CliRunner().invoke(kraken, ["research", "--input", str(path)])
    assert result.exit_code == 0, result.output
    assert "confirmed_events" in result.output


def test_forced_close_costs_and_equal_weight_benchmark():
    data = {"CHEAP": bars([1, 2]), "EXPENSIVE": bars([1000, 1000])}
    def signal(pair, xs, i, position):
        return Decision(enter=i == 0 and position is None)
    result = PortfolioEvaluator(costs=CostModel()).run("long", data, signal)
    assert result.trades == 2
    assert 45 < result.benchmark_return_pct < 50


def test_warmup_is_excluded_from_benchmark_and_sharpe_curve():
    data = {"A": bars([1, 100, 100])}
    result = PortfolioEvaluator().run("warm", data, lambda *args: Decision(),
                                      trade_start_ts=data["A"][1].ts)
    assert -2 < result.benchmark_return_pct < 0
    assert result.return_pct == 0


def test_shared_budget_limits_concurrent_entries():
    data = {"A": bars([10, 11]), "B": bars([10, 11]), "C": bars([10, 11])}
    def signal(pair, xs, i, position):
        return Decision(enter=i == 0 and position is None, score=1)
    assert PortfolioEvaluator(60, 30).run("slots", data, signal).trades == 2


def test_residual_signal_removes_market_factor():
    market = bars([100, 110, 121, 133.1])
    clone = bars([50, 55, 60.5, 66.55])
    weak = bars([50, 55, 54, 53])
    data = {"XBTUSDC": market, "CLONE": clone, "WEAK": weak}
    signal = residual_mean_reversion_signal(data, fit_residual_betas(data),
                                            lookback=1, entry=.03)
    assert signal("CLONE", clone, 3, None).enter is False
    assert signal("WEAK", weak, 3, None).enter is True


def test_atr_trail_uses_post_entry_peak():
    xs = [Bar(i, p, h, p-1, p) for i, (p, h) in enumerate([
        (10, 11), (11, 12), (13, 14), (15, 16), (12, 13)])]
    signal = volatility_breakout_signal(lookback=2, atr_n=2, stop_atr=1)
    assert signal("A", xs, 4, SimpleNamespace(peak=16)).exit is True


def test_walk_forward_has_warmup_but_never_trades_it():
    data = {"A": bars(list(range(10, 30))), "B": bars(list(range(20, 40)))}
    train_ends, first_seen = [], []
    def factory(train, context):
        train_ends.append(train["A"][-1].ts)
        seen = False
        def signal(pair, xs, i, position):
            nonlocal seen
            if not seen:
                first_seen.append(xs[i].ts)
                seen = True
            return Decision(enter=position is None)
        return signal
    folds = walk_forward(data, factory, 8, 4, PortfolioEvaluator(), "wf", warmup_bars=3)
    assert len(folds) == 3
    assert all(first > end for first, end in zip(first_seen, train_ends))


def metric(name="m", pnl=10, excess=2):
    return Metrics(name, 60, 60, pnl, 20, .2, 1.5, 10, 1, 100, 50,
                   18, excess, -1, {"A": {"pnl": 6}, "B": {"pnl": 4}})


def test_graduation_requires_all_evidence_and_then_passes():
    base = metric()
    assert graduation(base, 100)["eligible"] is False
    folds = [metric(str(i)) for i in range(3)]
    gate = graduation(base, 100, folds=folds, stressed=metric("stress"),
                      regimes={"bull": metric(), "bear": metric()}, holdout=metric("holdout"))
    assert gate["eligible"] is True


@pytest.mark.asyncio
async def test_paper_basis_peak_restart_and_delete_are_isolated():
    db = Database(":memory:")
    await db.connect()
    cfg = SimpleNamespace(directional_llm_enabled=True, directional_llm_paper=True,
                          directional_pairs=["XBTUSDC"])
    settings = SimpleNamespace(kraken=cfg, is_live=True)
    client = SimpleNamespace(get_price=AsyncMock(return_value=110.0))
    bot = SimpleNamespace(_components=Components({"db": db}))
    pillar = KrakenPillar(settings, client, bot=bot)
    await pillar._save_paper_position("XBTUSDC", .25, 102.5)
    assert await pillar._update_and_get_peak("XBTUSDC", 7.3) == 7.3
    held = await KrakenPillar(settings, client, bot=bot)._reconcile_paper_positions(["XBTUSDC"])
    assert held["XBTUSDC"] == (102.5, 110.0, .25)
    assert await db.fetchone("SELECT * FROM position_peaks WHERE market_id='XBTUSDC'") is None
    await pillar._delete_paper_position("XBTUSDC")
    assert await db.fetchone("SELECT * FROM kraken_paper_positions") is None
    await db.close()


@pytest.mark.asyncio
async def test_two_paper_buys_have_independent_fill_ids_and_basis():
    db = Database(":memory:")
    await db.connect()
    cfg = SimpleNamespace(directional_llm_enabled=True, directional_llm_paper=True,
                          directional_fee_pct=.26, directional_paper_slippage_bps=5)
    settings = SimpleNamespace(kraken=cfg, is_live=True)
    client = SimpleNamespace(get_bid_ask=AsyncMock(side_effect=[(99, 101), (199, 201)]))
    pillar = KrakenPillar(settings, client,
        bot=SimpleNamespace(_components=Components({"db": db})))
    a = await pillar._record_directional_fill("XBTUSDC", OrderSide.BUY,
        SimpleNamespace(order_id="paper-a"), 100, .1, paper=True)
    b = await pillar._record_directional_fill("ETHUSDC", OrderSide.BUY,
        SimpleNamespace(order_id="paper-b"), 200, .1, paper=True)
    assert a == pytest.approx(101 * 1.0005)
    assert b == pytest.approx(201 * 1.0005)
    rows = await db.fetchall("SELECT DISTINCT order_id FROM fills")
    assert {r["order_id"] for r in rows} == {"paper-a", "paper-b"}
    await db.close()
