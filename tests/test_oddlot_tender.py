"""Tests for the odd-lot tender pillar, EDGAR parsing, and IBKR ledger wiring.

Locks in:
  1. EDGAR search-hit parsing (defensive: malformed rows skipped) + ticker
     extraction + doc URL construction.
  2. Verdict gating: no odd-lot priority / record-date requirement / low
     confidence never alert or trade; verdicts are permanent (one LLM call
     per accession, ever).
  3. Detection-only mode when IBKR is disabled (alerts fire, no orders).
  4. Entry path: 99-share cap, thin-premium skip, paper-forcing, and the
     standard rails (markets row exchange='ibkr', signals/trades/fills/
     portfolio) so the ledger and graduation can score 'oddlot_tender'.
  5. Ledger fallback: a bare US ticker fill resolves venue='ibkr',
     category='ibkr_equity'.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from auramaur.broker.pnl import PnLTracker
from auramaur.data_sources.edgar import TenderFiling, parse_search_hits
from auramaur.db.database import Database
from auramaur.exchange.models import OrderResult
from auramaur.strategy.oddlot_tender import OddLotTenderPillar
from config.settings import Settings


def _filing(accession="0001-26-000001", ticker="ACME"):
    return TenderFiling(
        accession=accession, cik="1234567",
        company=f"Acme Corp ({ticker}) (CIK 1234567)",
        form="SC TO-I", filed_at="2026-06-09", primary_doc="scto.htm",
    )


def _settings(**overrides) -> Settings:
    s = Settings()
    s.oddlot_tender.enabled = True
    s.oddlot_tender.paper = True
    s.oddlot_tender.min_premium_pct = 2.0
    s.oddlot_tender.llm_min_confidence = 0.8
    for k, v in overrides.items():
        setattr(s.oddlot_tender, k, v)
    return s


def _edgar(filings, text="This tender offer includes odd lot priority ..."):
    e = MagicMock()
    e.recent_tender_filings = AsyncMock(return_value=filings)
    e.fetch_document = AsyncMock(return_value=text)
    return e


def _analyzer(priority=True, record_date=False, price=20.0, high=20.0,
              conf=0.95):
    a = MagicMock()
    a._call_llm = AsyncMock(return_value=(
        f'{{"odd_lot_priority": {str(priority).lower()}, '
        f'"requires_record_date_holding": {str(record_date).lower()}, '
        f'"tender_price": {price}, "tender_price_high": {high}, '
        f'"expiration": "2026-07-15", "conditions": "none material", '
        f'"confidence": {conf}}}'))
    return a


def _equity(price=19.0):
    eq = MagicMock()
    eq.get_price = AsyncMock(return_value=price)
    eq.place_share_order = AsyncMock(side_effect=lambda sym, side, qty, limit_price, dry_run: OrderResult(
        order_id="PAPER", market_id=sym, status="paper",
        filled_size=float(qty), filled_price=limit_price, is_paper=True))
    return eq


def _pillar(db, settings, filings, analyzer=None, equity=None, alerts=None,
            ibkr_enabled=True):
    settings.ibkr.enabled = ibkr_enabled
    return OddLotTenderPillar(
        db=db, settings=settings, edgar=_edgar(filings),
        analyzer=analyzer if analyzer is not None else _analyzer(),
        alerts=alerts, equity_client=equity,
        pnl_tracker=PnLTracker(db, settings),
    )


def test_parse_search_hits_defensive():
    payload = {"hits": {"hits": [
        {"_id": "0001-26-000001:scto.htm",
         "_source": {"ciks": ["1234567"],
                     "display_names": ["Acme Corp (ACME) (CIK 1234567)"],
                     "file_type": "SC TO-I", "file_date": "2026-06-09"}},
        {"_id": "broken-no-source"},          # skipped
        {"_id": ":nofile", "_source": {}},    # skipped (no accession/ciks)
    ]}}
    filings = parse_search_hits(payload)
    assert len(filings) == 1
    f = filings[0]
    assert f.ticker == "ACME"
    assert f.doc_url == ("https://www.sec.gov/Archives/edgar/data/1234567/"
                         "000126000001/scto.htm")


def test_opportunity_enters_with_full_rails():
    async def run():
        db = Database(":memory:")
        await db.connect()
        eq = _equity(price=19.0)  # tender $20 vs $19 = 5.3% premium
        alerts = MagicMock()
        alerts.send = AsyncMock()
        pillar = _pillar(db, _settings(), [_filing()], equity=eq, alerts=alerts)
        assert await pillar.run_once() == 1

        alerts.send.assert_awaited_once()
        # 99-share order, paper-forced
        call = eq.place_share_order.await_args
        assert call.args[2] == 99 and call.kwargs["dry_run"] is True

        row = await db.fetchone("SELECT * FROM oddlot_filings")
        assert row["odd_lot_priority"] == 1 and row["status"] == "entered"
        m = await db.fetchone("SELECT exchange, category FROM markets WHERE id='ACME'")
        assert (m["exchange"], m["category"]) == ("ibkr", "ibkr_equity")
        assert await db.fetchone(
            "SELECT 1 FROM trades WHERE strategy_source='oddlot_tender'")
        pos = await db.fetchone("SELECT size, is_paper FROM portfolio WHERE market_id='ACME'")
        assert pos["size"] == 99.0 and pos["is_paper"] == 1
        # Ledger context resolves via the markets row.
        from auramaur.broker.ledger import _market_context
        venue, category, strategy = await _market_context(db, "ACME")
        assert (venue, category, strategy) == ("ibkr", "ibkr_equity", "oddlot_tender")

        # Verdict is permanent: second cycle re-sees the filing, no re-entry.
        assert await pillar.run_once() == 0
        await db.close()

    asyncio.run(run())


def test_gating_blocks_bad_verdicts():
    async def run():
        db = Database(":memory:")
        await db.connect()
        cases = [
            _analyzer(priority=False),                # no priority
            _analyzer(priority=True, record_date=True),  # record-date kill
            _analyzer(priority=True, conf=0.5),       # low confidence
        ]
        for i, analyzer in enumerate(cases):
            eq = _equity()
            pillar = _pillar(db, _settings(), [_filing(accession=f"a-{i}")],
                             analyzer=analyzer, equity=eq)
            assert await pillar.run_once() == 0
            eq.place_share_order.assert_not_awaited()
        await db.close()

    asyncio.run(run())


def test_detection_only_without_ibkr():
    async def run():
        db = Database(":memory:")
        await db.connect()
        alerts = MagicMock()
        alerts.send = AsyncMock()
        pillar = _pillar(db, _settings(), [_filing()], equity=None,
                         alerts=alerts, ibkr_enabled=False)
        assert await pillar.run_once() == 1  # opportunity detected + alerted
        alerts.send.assert_awaited_once()
        row = await db.fetchone("SELECT status FROM oddlot_filings")
        assert row["status"] == "alerted"
        assert await db.fetchone("SELECT 1 FROM trades") is None
        await db.close()

    asyncio.run(run())


def test_thin_premium_and_expensive_skipped():
    async def run():
        db = Database(":memory:")
        await db.connect()
        # Premium 0.5% < 2% floor -> skip.
        eq = _equity(price=19.9)
        pillar = _pillar(db, _settings(), [_filing(accession="thin")], equity=eq)
        await pillar.run_once()
        eq.place_share_order.assert_not_awaited()
        row = await db.fetchone(
            "SELECT status FROM oddlot_filings WHERE accession='thin'")
        assert row["status"] == "premium_too_thin"

        # Price above max_position_usd for even 1 share -> skip.
        eq2 = _equity(price=30.0)
        pillar2 = _pillar(db, _settings(max_position_usd=25.0),
                          [_filing(accession="rich")],
                          analyzer=_analyzer(price=40.0), equity=eq2)
        await pillar2.run_once()
        eq2.place_share_order.assert_not_awaited()
        await db.close()

    asyncio.run(run())


def test_position_cap_shrinks_qty_below_99():
    async def run():
        db = Database(":memory:")
        await db.connect()
        # $2500 cap at $50/share -> 50 shares, not 99.
        eq = _equity(price=50.0)
        pillar = _pillar(db, _settings(max_position_usd=2500.0),
                         [_filing(accession="big")],
                         analyzer=_analyzer(price=55.0), equity=eq)
        assert await pillar.run_once() == 1
        assert eq.place_share_order.await_args.args[2] == 50
        await db.close()

    asyncio.run(run())


def test_ledger_bare_ticker_fallback():
    async def run():
        from auramaur.broker.ledger import _market_context
        db = Database(":memory:")
        await db.connect()
        # No markets row at all: bare ticker falls back to ibkr context.
        venue, category, _ = await _market_context(db, "NVDA")
        assert (venue, category) == ("ibkr", "ibkr_equity")
        # Kraken pairs still win over the ticker heuristic.
        venue, category, strategy = await _market_context(db, "XBTUSDC")
        assert (venue, category, strategy) == ("kraken", "kraken_spot", "kraken_directional")
        await db.close()

    asyncio.run(run())


def test_safe_float_tolerates_non_numeric_tender_prices():
    """Regression: a NAV-linked tender returns a non-numeric price ('NAV') for
    the price field. _safe_float must coerce it to 0.0 (downstream rejects as
    'no usable fixed premium') instead of raising — a single bad field used to
    discard the entire filing audit, including a well-parsed odd_lot_priority."""
    from auramaur.strategy.oddlot_tender import _safe_float
    assert _safe_float("NAV") == 0.0
    assert _safe_float("N/A") == 0.0
    assert _safe_float(None) == 0.0
    assert _safe_float("") == 0.0
    assert _safe_float("12.50") == 12.5
    assert _safe_float(12.5) == 12.5
    assert _safe_float("bad", default=-1.0) == -1.0
