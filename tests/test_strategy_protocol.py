"""Conformance guard for the uniform Strategy pillar contract.

Every pillar declares ``name`` + ``execution_mode`` (strategy/protocols.py). The
point of the contract is to make the EXECUTE stage explicit and *checked*: a
pillar that should route through the single ExecutionGateway must not quietly
grow a direct ``exchange.place_order`` bypass, while the legitimate bypasses
(market maker resting quotes, concurrent arb legs, IBKR equities) are declared
and whitelisted with a reason. That turns "is this gateway-bypass intentional?"
from an unanswerable code-reading question into a test.
"""

from __future__ import annotations

import pathlib

import pytest

from auramaur.strategy.bias_harvest import BiasHarvestPillar
from auramaur.strategy.cross_venue_arb import CrossVenueArbPillar
from auramaur.strategy.econ_indicator import EconIndicatorPillar
from auramaur.strategy.entailment_arb import EntailmentArbPillar
from auramaur.strategy.market_maker import MarketMaker
from auramaur.strategy.momentum_coupling import MomentumCouplingPillar
from auramaur.strategy.oddlot_tender import OddLotTenderPillar
from auramaur.strategy.protocols import ExecutionMode, GATEWAY_PURE_MODES
from auramaur.strategy.resolution_lens import ResolutionLensPillar
from auramaur.strategy.arbitrage_scanner import ArbitrageScanner
from auramaur.strategy.weather_temp import WeatherTempPillar

_ROOT = pathlib.Path(__file__).resolve().parent.parent

# (pillar class, source module) — every pillar the bot drives.
PILLARS = [
    (BiasHarvestPillar, "auramaur/strategy/bias_harvest.py"),
    (ResolutionLensPillar, "auramaur/strategy/resolution_lens.py"),
    (WeatherTempPillar, "auramaur/strategy/weather_temp.py"),
    (EconIndicatorPillar, "auramaur/strategy/econ_indicator.py"),
    (EntailmentArbPillar, "auramaur/strategy/entailment_arb.py"),
    (CrossVenueArbPillar, "auramaur/strategy/cross_venue_arb.py"),
    (MomentumCouplingPillar, "auramaur/strategy/momentum_coupling.py"),
    (MarketMaker, "auramaur/strategy/market_maker.py"),
    (ArbitrageScanner, "auramaur/strategy/arbitrage_scanner.py"),
    (OddLotTenderPillar, "auramaur/strategy/oddlot_tender.py"),
]

# GATEWAY_PURE_MODES (single source of truth in protocols.py): the pillar must
# not call exchange.place_order at all. The other modes legitimately place
# directly (MM quotes; concurrent arb legs that then record_external_fill;
# equities) and are skipped with their declared reason below.


def test_every_pillar_declares_a_unique_named_contract():
    names: set[str] = set()
    for cls, _ in PILLARS:
        assert isinstance(cls.execution_mode, ExecutionMode), cls.__name__
        assert isinstance(cls.name, str) and cls.name, cls.__name__
        assert cls.name not in names, f"duplicate pillar name: {cls.name}"
        names.add(cls.name)


@pytest.mark.parametrize("cls,modfile", PILLARS, ids=[c.name for c, _ in PILLARS])
def test_gateway_pillars_do_not_place_orders_directly(cls, modfile):
    """A GATEWAY_SINGLE/PAIRED pillar must route through the ExecutionGateway —
    it must not call exchange.place_order in its own module. The direct-placement
    modes are skipped with their declared reason (this is the whitelist)."""
    if cls.execution_mode not in GATEWAY_PURE_MODES:
        pytest.skip(
            f"{cls.name} is {cls.execution_mode.value}: direct placement is its "
            f"declared, intentional path (not a gateway bypass)")
    src = (_ROOT / modfile).read_text()
    assert ".place_order(" not in src, (
        f"{cls.name} ({cls.execution_mode.value}) must submit through the "
        f"ExecutionGateway, not call place_order directly")
