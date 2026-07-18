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
from auramaur.strategy.informed_flow_pillar import InformedFlowPillar
from auramaur.strategy.ibkr_etf_paper import IBKRETFPaperPillar
from auramaur.strategy.long_horizon import LongHorizonPillar
from auramaur.strategy.cross_venue_arb import CrossVenueArbPillar
from auramaur.strategy.econ_indicator import EconIndicatorPillar
from auramaur.strategy.entailment_arb import EntailmentArbPillar
from auramaur.strategy.market_maker import MarketMaker
from auramaur.strategy.momentum_coupling import MomentumCouplingPillar
from auramaur.strategy.oddlot_tender import OddLotTenderPillar
from auramaur.strategy.protocols import ExecutionMode, NO_DIRECT_PLACE_MODES
from auramaur.strategy.resolution_lens import ResolutionLensPillar
from auramaur.strategy.arbitrage_scanner import ArbitrageScanner
from auramaur.strategy.weather_temp import WeatherTempPillar

_ROOT = pathlib.Path(__file__).resolve().parent.parent

# (pillar class, source module) — every pillar the bot drives.
PILLARS = [
    (BiasHarvestPillar, "auramaur/strategy/bias_harvest.py"),
    (LongHorizonPillar, "auramaur/strategy/long_horizon.py"),
    (InformedFlowPillar, "auramaur/strategy/informed_flow_pillar.py"),
    (ResolutionLensPillar, "auramaur/strategy/resolution_lens.py"),
    (WeatherTempPillar, "auramaur/strategy/weather_temp.py"),
    (EconIndicatorPillar, "auramaur/strategy/econ_indicator.py"),
    (EntailmentArbPillar, "auramaur/strategy/entailment_arb.py"),
    (CrossVenueArbPillar, "auramaur/strategy/cross_venue_arb.py"),
    (MomentumCouplingPillar, "auramaur/strategy/momentum_coupling.py"),
    (MarketMaker, "auramaur/strategy/market_maker.py"),
    (ArbitrageScanner, "auramaur/strategy/arbitrage_scanner.py"),
    (OddLotTenderPillar, "auramaur/strategy/oddlot_tender.py"),
    (IBKRETFPaperPillar, "auramaur/strategy/ibkr_etf_paper.py"),
]

# NO_DIRECT_PLACE_MODES (single source of truth in protocols.py): the pillar must
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
    if cls.execution_mode not in NO_DIRECT_PLACE_MODES:
        pytest.skip(
            f"{cls.name} is {cls.execution_mode.value}: direct placement is its "
            f"declared, intentional path (not a gateway bypass)")
    src = (_ROOT / modfile).read_text()
    assert ".place_order(" not in src, (
        f"{cls.name} ({cls.execution_mode.value}) must submit through the "
        f"ExecutionGateway, not call place_order directly")


def test_arb_mixin_does_not_place_orders_directly():
    """The arb execution mixin (bot_arb.py) places legs via the gateway's
    place_legs adapter, never exchange.place_order directly — extending the
    'only the gateway places' perimeter beyond the pillar modules."""
    src = (_ROOT / "auramaur/bot_arb.py").read_text()
    assert ".place_order(" not in src, (
        "bot_arb.py must place through ExecutionGateway.place_legs, not call "
        "exchange.place_order directly")


def test_exit_path_only_ibkr_places_directly():
    """Exit-path perimeter: the poly/kalshi exits route through the gateway's
    submit_exit; the ONLY direct exchange.place_order in bot_exits.py is the IBKR
    equity/options exit (the DIRECT_EQUITY exception — IBKR is off the
    prediction-market gateway, with its own accounting). This names that
    exception so a new direct placement on the gateway-venue exits fails."""
    import ast
    src = (_ROOT / "auramaur/bot_exits.py").read_text()
    tree = ast.parse(src)
    ibkr = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == "_execute_ibkr_exit")
    bad = [node.lineno for node in ast.walk(tree)
           if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
           and node.func.attr == "place_order"
           and not (ibkr.lineno <= node.lineno <= ibkr.end_lineno)]
    assert not bad, (
        f"direct exchange.place_order in bot_exits.py outside the named IBKR "
        f"equity exit (lines {bad}) — poly/kalshi exits must use gateway.submit_exit")
