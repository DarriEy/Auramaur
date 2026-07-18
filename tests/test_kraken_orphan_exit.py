"""Tests for Kraken directional orphan-position exit wiring.

Regression coverage for the gap where a held directional position whose pair was
pruned from the valid set (or removed from config) was never iterated by the
exit loop — leaving it stuck on Kraken, unsold. _directional now evaluates the
union of configured pairs and currently-held bases, force-closing orphans.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from auramaur.exchange.models import OrderSide
from auramaur.treasury.kraken_pillar import KrakenPillar
from config.settings import Settings


def run(coro):
    return asyncio.run(coro)


class FakeKraken:
    """Minimal async stub of KrakenSpotClient for the pillar's exit path."""

    def __init__(self, balances: dict, price: float = 150.0):
        self._bal = balances
        self._price = price
        self.sells: list[tuple[str, float]] = []
        self.buys: list[tuple[str, float]] = []

    async def get_free_balance(self):
        return dict(self._bal)

    async def get_price(self, pair):
        return self._price

    async def usd_notional(self, pair, volume, price=None):
        return volume * (price or self._price)

    async def size_for_usd(self, pair, usd, price=None):
        return usd / (price or self._price)

    async def get_pair_quote(self, pair):
        return "ZUSD"

    async def query_fill(self, txid):
        # A real market order fills; report the last order's executed fill so the
        # pillar can confirm the close (vol > 0 == filled).
        return getattr(self, "_last_fill", None)

    async def _public(self, endpoint, params=None):
        return {}

    async def place_spot_order(self, pair, side, **kw):
        vol = kw.get("volume")
        self._last_fill = {"price": self._price, "vol": vol or 0.0, "fee": 0.0}
        rec = (pair, vol)
        (self.sells if side == OrderSide.SELL else self.buys).append(rec)
        return SimpleNamespace(order_id="OK", status="ok", error_message="")


def _pillar(fake, **kraken_overrides) -> KrakenPillar:
    s = Settings()
    s.kraken.directional_enabled = True
    s.kraken.directional_pairs = []
    # These tests exercise the liquidation mechanism itself. The shipped paper
    # profile disables it operationally so a reduced allowlist cannot touch
    # legacy real holdings.
    s.kraken.directional_liquidate_orphans = True
    for k, v in kraken_overrides.items():
        setattr(s.kraken, k, v)
    p = KrakenPillar(settings=s, kraken_client=fake, bot=None)
    # Pre-resolve so _directional skips the catalog fetch.
    p._pair_base = {}
    p._valid_pairs = []
    p._base_to_pair = {"SOL": "SOLUSD", "XXBT": "BTCUSD"}
    p._base_pair_meta = {"SOLUSD": ("SOL", 0.0, 8), "BTCUSD": ("XXBT", 0.0, 8)}
    return p


# ---------------------------------------------------------------------------
# _is_cash_asset
# ---------------------------------------------------------------------------

class TestCashAssetFilter:
    def test_classification(self):
        p = KrakenPillar(settings=Settings(), kraken_client=None, bot=None)
        assert p._is_cash_asset("USDC")
        assert p._is_cash_asset("ZUSD")
        assert p._is_cash_asset("ZEUR")
        assert p._is_cash_asset("ETH2.S")   # staked variant
        assert not p._is_cash_asset("SOL")
        assert not p._is_cash_asset("XXBT")


# ---------------------------------------------------------------------------
# _managed_pairs — the union
# ---------------------------------------------------------------------------

class TestManagedPairs:
    def test_union_adds_orphans_skips_cash(self):
        fake = FakeKraken({})
        p = _pillar(fake)
        p._valid_pairs = ["BTCUSD"]
        p._pair_base = {"BTCUSD": "XXBT"}
        bal = {"XXBT": 0.5, "SOL": 1.0, "USDC": 50.0, "ZEUR": 10.0, "DOT.S": 3.0}
        managed = p._managed_pairs(bal)
        assert "BTCUSD" in managed       # configured/valid
        assert "SOLUSD" in managed       # orphan (held, not configured)
        assert len(managed) == 2         # cash + staked excluded; BTC not duplicated

    def test_unmappable_orphan_skipped(self):
        fake = FakeKraken({})
        p = _pillar(fake)
        p._base_to_pair = {}             # nothing maps
        managed = p._managed_pairs({"WIF": 100.0})
        assert managed == []


# ---------------------------------------------------------------------------
# reconcile keeps orphans (does not drop as "closed externally")
# ---------------------------------------------------------------------------

class TestReconcileKeepsOrphan:
    def test_orphan_not_dropped(self):
        fake = FakeKraken({"SOL": 1.0})
        p = _pillar(fake)
        p._dir_long = {"SOLUSD": 140.0}
        p._register_orphan_pair("SOLUSD")
        held = run(p._reconcile_positions({"SOL": 1.0}, ["SOLUSD"]))
        assert "SOLUSD" in held
        assert "SOLUSD" in p._dir_long   # NOT dropped


# ---------------------------------------------------------------------------
# _directional — orphan liquidation
# ---------------------------------------------------------------------------

class TestOrphanLiquidation:
    def test_orphan_is_sold(self):
        fake = FakeKraken({"SOL": 1.0, "USDC": 100.0})
        p = _pillar(fake)  # no valid pairs → SOL is an orphan
        run(p._directional())
        assert fake.sells == [("SOLUSD", 1.0)]
        assert "SOLUSD" not in p._dir_long  # tracking cleared after close

    def test_orphan_left_when_liquidation_disabled(self):
        fake = FakeKraken({"SOL": 1.0, "USDC": 100.0})
        p = _pillar(fake, directional_liquidate_orphans=False)
        run(p._directional())
        assert fake.sells == []

    def test_configured_pair_not_treated_as_orphan(self):
        # SOL is configured+valid: it should go through the normal exit path
        # (momentum), not the forced orphan liquidation. With flat momentum and
        # default stops it holds, so no sell.
        fake = FakeKraken({"SOL": 1.0, "USDC": 100.0})
        p = _pillar(fake)
        p._valid_pairs = ["SOLUSD"]
        p._pair_base = {"SOLUSD": "SOL"}
        p._pair_min = {"SOLUSD": 0.0}
        p._pair_lot_dec = {"SOLUSD": 8}
        p._dir_long = {"SOLUSD": 150.0}   # entry == price → flat
        # momentum returns 0 (flat) → no momentum exit, no stop hit
        async def _flat_mom(pair):
            return 0.0
        p._momentum = _flat_mom
        run(p._directional())
        assert fake.sells == []
