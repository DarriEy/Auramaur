"""Pure formation of atomic two-leg arbitrage proposals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ArbitrageMarket:
    """The market facts needed to form a pair, detached from venue models."""

    market_id: str
    question: str
    yes_price: float
    exchange: str = "polymarket"
    yes_token: str = ""
    no_token: str = ""


@dataclass(frozen=True)
class ArbitrageLeg:
    market_id: str
    question: str
    side: str
    market_probability: float
    fair_probability: float
    edge_percent: float


@dataclass(frozen=True)
class PairedArbitrageProposal:
    """One indivisible research proposal; its legs are not standalone targets."""

    buy: ArbitrageLeg
    sell: ArbitrageLeg
    opportunity_type: str
    edge: float


def form_arbitrage_pair(
    opportunity: Mapping[str, Any],
    market_a: ArbitrageMarket,
    market_b: ArbitrageMarket,
) -> PairedArbitrageProposal | None:
    """Deterministically form a pair, failing closed on incomplete candidates."""
    try:
        if market_a.market_id != str(opportunity["market_a"]):
            return None
        if market_b.market_id != str(opportunity["market_b"]):
            return None
        kind = str(opportunity["type"])
        price_a = float(opportunity["price_a"])
        price_b = float(opportunity["price_b"])
    except (KeyError, TypeError, ValueError):
        return None

    prices = (price_a, price_b, market_a.yes_price, market_b.yes_price)
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in prices):
        return None

    if kind == "conditional_violation":
        buy, sell = market_b, market_a
        edge = price_a - price_b
    elif kind == "price_divergence":
        buy, sell = (
            (market_a, market_b) if price_a < price_b else (market_b, market_a)
        )
        try:
            edge = float(opportunity.get("divergence", abs(price_a - price_b)))
        except (TypeError, ValueError):
            return None
    else:
        return None

    if not math.isfinite(edge) or edge <= 0.0:
        return None
    if not _has_execution_metadata(buy, "BUY") or not _has_execution_metadata(sell, "SELL"):
        return None

    half_edge = edge / 2.0
    return PairedArbitrageProposal(
        buy=ArbitrageLeg(
            market_id=buy.market_id,
            question=buy.question,
            side="BUY",
            market_probability=buy.yes_price,
            fair_probability=buy.yes_price + half_edge,
            edge_percent=half_edge * 100.0,
        ),
        sell=ArbitrageLeg(
            market_id=sell.market_id,
            question=sell.question,
            side="SELL",
            market_probability=sell.yes_price,
            fair_probability=sell.yes_price - half_edge,
            edge_percent=half_edge * 100.0,
        ),
        opportunity_type=kind,
        edge=edge,
    )


def _has_execution_metadata(market: ArbitrageMarket, side: str) -> bool:
    if (market.exchange or "polymarket") != "polymarket":
        return True
    return bool(market.yes_token if side == "BUY" else market.no_token)
