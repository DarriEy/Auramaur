"""Regression: settle_via_venue must reach holdings whose portfolio row is gone.

The venue sweep used to match resolved positions only against the `portfolio`
table. A live position drops its portfolio row the moment a leg sells/settles,
but the wallet keeps holding the (now worthless) loser token and cost_basis
keeps a non-zero size — so the venue kept reporting it, the sweep kept skipping
it (portfolio miss), and the cost_basis leg resurrected every cycle, never
booking and never draining. The fix unions cost_basis into the match set and
stops skipping already-booked legs before the drain.
"""

import pytest

from auramaur.broker import redeemer as redeemer_mod
from auramaur.broker.redeemer import RedeemablePosition
from auramaur.db.database import Database
from auramaur.strategy.resolution_tracker import ResolutionTracker


def _vp(asset_id: str, *, is_winner: bool, token_label: str = "No",
        size: float = 10.0, avg: float = 0.30) -> RedeemablePosition:
    return RedeemablePosition(
        condition_id="0xcond", asset_id=asset_id, title="Will X?",
        outcome=token_label, size=size, avg_price=avg,
        cur_price=1.0 if is_winner else 0.0,
        payout=size if is_winner else 0.0, is_winner=is_winner,
        redeemable_now=True, status="redeemable", neg_risk=False,
        mergeable=False, end_date="2026-05-01T00:00:00+00:00", slug="x",
    )


async def _tracker(tmp_path):
    db = Database(str(tmp_path / "auramaur.db"))
    await db.connect()
    await db.execute(
        "INSERT INTO markets (id, question, active, condition_id, last_updated) "
        "VALUES ('mkt1','Will X?',1,'0xcond','now')")
    await db.commit()
    return db, ResolutionTracker(db, None, {}, proxy_address="0xproxy")


async def _cost_basis(db, *, token, token_id, size, avg, market="mkt1"):
    await db.execute(
        "INSERT INTO cost_basis (market_id, token, token_id, size, avg_cost, "
        "total_cost, is_paper) VALUES (?,?,?,?,?,?,0)",
        (market, token, token_id, size, avg, avg * size))
    await db.commit()


@pytest.mark.asyncio
async def test_settles_holding_with_no_portfolio_row(tmp_path, monkeypatch):
    """A loser held only in cost_basis (no portfolio row) settles + drains."""
    db, tracker = await _tracker(tmp_path)
    try:
        await _cost_basis(db, token="No", token_id="TOK_NO", size=10.0, avg=0.30)

        async def fake_fetch(proxy, *a, **k):
            return [_vp("TOK_NO", is_winner=False, size=10.0, avg=0.30)]
        monkeypatch.setattr(redeemer_mod, "fetch_redeemable_positions", fake_fetch)

        settled = await tracker.settle_via_venue("0xproxy")

        assert len(settled) == 1
        # Loss booked: (0 - 0.30) * 10 = -3.00
        led = await db.fetchone(
            "SELECT pnl FROM pnl_ledger WHERE market_id='mkt1' AND is_paper=0")
        assert led is not None and led["pnl"] == pytest.approx(-3.0)
        # cost_basis drained, market marked inactive.
        cb = await db.fetchone("SELECT size FROM cost_basis WHERE market_id='mkt1'")
        assert cb["size"] == pytest.approx(0.0)
        active = await db.fetchone("SELECT active FROM markets WHERE id='mkt1'")
        assert active["active"] == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_already_booked_leg_still_drains_cost_basis(tmp_path, monkeypatch):
    """An already-booked settlement with a lingering cost_basis size must drain
    (the resurrection bug) without double-booking the P&L."""
    db, tracker = await _tracker(tmp_path)
    try:
        await _cost_basis(db, token="No", token_id="TOK_NO", size=10.0, avg=0.30)
        # Pre-book the settlement at the correct venue value (-3.00).
        await db.execute(
            "INSERT INTO pnl_ledger (market_id, kind, token, qty, pnl, is_paper, "
            "source_ref) VALUES ('mkt1','settlement','NO',10,-3.0,0,'settle:mkt1:NO:0')")
        await db.commit()

        async def fake_fetch(proxy, *a, **k):
            return [_vp("TOK_NO", is_winner=False, size=10.0, avg=0.30)]
        monkeypatch.setattr(redeemer_mod, "fetch_redeemable_positions", fake_fetch)

        await tracker.settle_via_venue("0xproxy")

        # Still exactly one ledger row, value unchanged (no double-book)...
        rows = await db.fetchall(
            "SELECT pnl FROM pnl_ledger WHERE source_ref='settle:mkt1:NO:0'")
        assert len(rows) == 1 and rows[0]["pnl"] == pytest.approx(-3.0)
        # ...but the lingering cost_basis leg is now drained.
        cb = await db.fetchone("SELECT size FROM cost_basis WHERE market_id='mkt1'")
        assert cb["size"] == pytest.approx(0.0)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fully_drained_position_is_not_reprocessed(tmp_path, monkeypatch):
    """Once cost_basis is zero and no portfolio row exists, the venue still lists
    the worthless token but the sweep must skip it (nothing to drain)."""
    db, tracker = await _tracker(tmp_path)
    try:
        await _cost_basis(db, token="No", token_id="TOK_NO", size=0.0, avg=0.30)

        async def fake_fetch(proxy, *a, **k):
            return [_vp("TOK_NO", is_winner=False)]
        monkeypatch.setattr(redeemer_mod, "fetch_redeemable_positions", fake_fetch)

        settled = await tracker.settle_via_venue("0xproxy")
        assert settled == []
    finally:
        await db.close()
