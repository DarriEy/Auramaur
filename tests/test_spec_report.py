"""Kraken spec performance readout."""

import asyncio
from types import SimpleNamespace

from auramaur.db.database import Database
from auramaur.monitoring.spec_report import (
    _trips_from_fills,
    gather_spec_performance,
    render_spec_performance,
)


def _settings(pairs, is_live=True):
    return SimpleNamespace(
        is_live=is_live,
        kraken=SimpleNamespace(directional_pairs=pairs, directional_enabled=True),
    )


def test_trips_from_fills_weighted_average():
    # Buy 10 @ 1.00, buy 10 @ 1.20 (avg 1.10), sell 20 @ 1.30 fee 0.05.
    # Realized = (1.30 - 1.10) * 20 - 0.05 = 4.00 - 0.05 = 3.95, one trip.
    rows = [
        {"market_id": "APEUSDC", "side": "BUY", "size": 10, "price": 1.00, "fee": 0},
        {"market_id": "APEUSDC", "side": "BUY", "size": 10, "price": 1.20, "fee": 0},
        {"market_id": "APEUSDC", "side": "SELL", "size": 20, "price": 1.30, "fee": 0.05},
    ]
    trips = _trips_from_fills(rows)
    assert len(trips) == 1
    assert abs(trips[0] - 3.95) < 1e-9


async def _seed(db):
    # Two pairs: APE a winner, SHIB a loser. Recorded as fills + cost_basis (live).
    fills = [
        ("APEUSDC", "BUY", 10, 1.00, 0.026), ("APEUSDC", "SELL", 10, 1.20, 0.031),
        ("SHIBUSDC", "BUY", 100, 0.10, 0.026), ("SHIBUSDC", "SELL", 100, 0.08, 0.020),
    ]
    for i, (mid, side, size, price, fee) in enumerate(fills):
        await db.execute(
            "INSERT INTO fills (order_id, market_id, side, token, size, price, fee, is_paper, timestamp) "
            "VALUES (?,?,?,?,?,?,?,0,?)",
            (f"tx{i}", mid, side, "YES", size, price, fee, f"2026-06-01T0{i}:00:00"),
        )
    # cost_basis realized: APE = (1.20-1.00)*10 - 0.031 = 1.969; SHIB = (0.08-0.10)*100 - 0.020 = -2.020
    await db.execute("INSERT INTO cost_basis (market_id, token, size, avg_cost, total_cost, realized_pnl, is_paper) "
                     "VALUES ('APEUSDC','YES',0,0,0,1.969,0)")
    await db.execute("INSERT INTO cost_basis (market_id, token, size, avg_cost, total_cost, realized_pnl, is_paper) "
                     "VALUES ('SHIBUSDC','YES',0,0,0,-2.020,0)")
    await db.commit()


def test_gather_spec_performance_net_and_winrate():
    async def run():
        db = Database(":memory:")
        await db.connect()
        await _seed(db)
        s = await gather_spec_performance(db, _settings(["APEUSDC", "SHIBUSDC"]))

        # Realized = 1.969 + (-2.020) = -0.051 (net of fees, the loser dominates).
        assert abs(s["realized"] - (-0.051)) < 1e-6
        assert s["trips"] == 2
        assert s["wins"] == 1 and s["losses"] == 1
        assert abs(s["win_rate"] - 50.0) < 1e-9
        # No open positions seeded -> unrealized 0, net == realized.
        assert s["open_count"] == 0
        assert abs(s["net"] - s["realized"]) < 1e-9
        # Fees summed across all 4 fills.
        assert abs(s["fees"] - (0.026 + 0.031 + 0.026 + 0.020)) < 1e-9
        # Per-pair present and sorted worst-first (SHIB before APE).
        assert [p["pair"] for p in s["per_pair"]] == ["SHIBUSDC", "APEUSDC"]
        # Renders without error.
        assert render_spec_performance(s) is not None
        await db.close()

    asyncio.run(run())


def test_gather_handles_no_pairs():
    async def run():
        db = Database(":memory:")
        await db.connect()
        s = await gather_spec_performance(db, _settings([]))
        assert s["trips"] == 0 and s["net"] == 0.0 and s["per_pair"] == []
        assert render_spec_performance(s) is not None
        await db.close()

    asyncio.run(run())


if __name__ == "__main__":
    test_trips_from_fills_weighted_average()
    test_gather_spec_performance_net_and_winrate()
    test_gather_handles_no_pairs()
    print("ok")
