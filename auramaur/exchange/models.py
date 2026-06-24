"""Pydantic models for markets, orders, and positions."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class MarketStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"


class Confidence(str, Enum):
    LOW = "LOW"
    MEDIUM_LOW = "MEDIUM_LOW"
    MEDIUM = "MEDIUM"
    MEDIUM_HIGH = "MEDIUM_HIGH"
    HIGH = "HIGH"

    @classmethod
    def _missing_(cls, value):
        """Case-insensitive lookup so an LLM returning e.g. ``'medium'`` (Gemini
        does) resolves to ``MEDIUM`` instead of crashing the cycle with
        ``'medium' is not a valid Confidence``. Truly-unknown values still raise.
        """
        if isinstance(value, str):
            norm = value.strip().upper().replace("-", "_").replace(" ", "_")
            for member in cls:
                if member.value == norm:
                    return member
        return None


class Market(BaseModel):
    id: str
    exchange: str = "polymarket"
    condition_id: str = ""
    ticker: str = ""
    question: str
    description: str = ""
    category: str = ""
    end_date: datetime | None = None
    active: bool = True
    # Polymarket's `active` flag lags resolution (a market stays active=True for
    # a while after it settles), but `closed` flips at venue closure — it is the
    # reliable "trading has ended" signal the resolution tracker keys off.
    closed: bool = False
    outcome_yes_price: float = 0.5
    outcome_no_price: float = 0.5
    volume: float = 0
    liquidity: float = 0
    spread: float = 0
    clob_token_yes: str = ""
    clob_token_no: str = ""
    # NegRisk grouping: Polymarket multi-outcome events (e.g. "who wins the
    # election") are a set of mutually-exclusive binary markets sharing one
    # neg_risk_market_id. Buying NO on every outcome for < (N-1) dollars is a
    # guaranteed-profit arb. neg_risk_market_id is the grouping key.
    neg_risk: bool = False
    neg_risk_market_id: str = ""
    # Venue-reported lifecycle. Kalshi exposes an explicit settlement status
    # ("settled"/"finalized") and a result side ("yes"/"no") — the resolution
    # tracker keyed off `market.status` while no Market ever carried the
    # field (getattr defaulted to None), so no Kalshi position was ever
    # settled into the ledger. Empty for venues that don't report these.
    status: str = ""
    result: str = ""
    # Polymarket UMA optimistic-oracle resolution state, surfaced by Gamma.
    # `uma_status` is the current stage ("" → no proposal yet, "proposed" →
    # outcome proposed and in the liveness window, "disputed" → actively
    # contested, "resolved" → final). `uma_statuses` is the full lifecycle
    # history (e.g. ["proposed","disputed","proposed","disputed","resolved"]).
    # A market mid-dispute can be price-pinned to the *proposed* outcome and
    # then flip, so the dispute-risk gate keys off these to avoid acting on /
    # settling a contested resolution. Empty for non-Polymarket venues.
    uma_status: str = ""
    uma_statuses: list[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def dispute_risk(self) -> str:
        """Resolution/dispute diagnostic from the venue's UMA state.

        Returns one of:
            "DO_NOT_ACT"            — an active, unresolved dispute: the outcome
                                      is genuinely contested, so neither enter
                                      nor settle off the (pinned-but-provisional)
                                      price.
            "INSUFFICIENT_EVIDENCE" — the venue reported a status we don't
                                      recognise; can't classify, fail closed.
            "READY"                 — no active dispute ("" not yet proposed,
                                      "proposed" in-window, or "resolved" final;
                                      a *resolved* market that was disputed in
                                      its history is still final, hence READY).

        Non-Polymarket venues (no UMA) report "" → READY.
        """
        status = (self.uma_status or "").strip().lower()
        if status == "disputed":
            return "DO_NOT_ACT"
        if status in ("", "proposed", "resolved"):
            return "READY"
        return "INSUFFICIENT_EVIDENCE"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class TokenType(str, Enum):
    YES = "YES"
    NO = "NO"

    @classmethod
    def from_str(cls, value: str | None) -> "TokenType":
        """Normalize an outcome string to a canonical TokenType.

        Case-insensitive so external strings — the reconciler/CLOB raw
        outcomes ("Yes"/"No"), Kalshi sides, legacy rows — all collapse to
        the same canonical ``.value`` ("YES"/"NO"). Anything that isn't a
        recognizable NO defaults to YES (the historical default).
        """
        return cls.NO if str(value or "").strip().upper() == "NO" else cls.YES


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class Order(BaseModel):
    market_id: str
    exchange: str = "polymarket"
    token_id: str = ""  # CLOB token ID for the outcome being traded
    side: OrderSide
    token: TokenType = TokenType.YES
    size: float
    price: float
    order_type: OrderType = OrderType.MARKET
    dry_run: bool = True  # Paper trade by default
    # Post-only: reject rather than cross the spread. Only meaningful for
    # LIMIT orders. Protects against turning a maker quote into a taker fill
    # when the book moves between order formation and submission, and keeps
    # us eligible for Polymarket's maker-reward tier.
    post_only: bool = False
    # Who placed this order ("market_maker", a strategy book like "llm" /
    # "news_speed", or "exit"). The order monitor uses it to render TTL
    # cancels honestly: an expired MM quote is routine churn, an expired
    # strategy entry is a must-see failure.
    source: str = ""
    # Stamped at construction (≈ submission time). The order monitor uses this
    # to TTL-cancel live limit orders that are still resting, since live GTC
    # orders never auto-expire on-chain and would otherwise lock collateral.
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def notional(self) -> float:
        return self.size * self.price


class OrderResult(BaseModel):
    order_id: str
    market_id: str
    status: Literal["filled", "partial", "pending", "rejected", "paper", "cancelled", "expired"]
    filled_size: float = 0
    filled_price: float = 0
    is_paper: bool = True
    error_message: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Position(BaseModel):
    market_id: str
    exchange: str = "polymarket"
    side: OrderSide
    size: float
    avg_price: float
    current_price: float = 0
    category: str = ""
    token: TokenType = TokenType.YES
    token_id: str = ""

    @property
    def unrealized_pnl(self) -> float:
        if self.side == OrderSide.BUY:
            return (self.current_price - self.avg_price) * self.size
        return (self.avg_price - self.current_price) * self.size


class OrderBookLevel(BaseModel):
    price: float
    size: float


class OrderBook(BaseModel):
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)

    # Compute, never index: the Polymarket CLOB /book endpoint returns levels
    # WORST-first (bids ascending, asks descending), so bids[0]/asks[0] were
    # the $0.01 stub bid and the $0.99 stub ask. Every consumer (router
    # crossing, exit pricing, market maker quoting) was reading a fictional
    # 98-cent-wide book. max/min is ordering-agnostic, so any venue's book
    # ordering is safe.
    @property
    def best_bid(self) -> float | None:
        return max(level.price for level in self.bids) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(level.price for level in self.asks) if self.asks else None

    @property
    def midpoint(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    def fill_to_size(self, shares: float, is_buy: bool) -> tuple[float, float, float]:
        """Walk the book as a taker and return ``(fillable_shares, vwap,
        marginal_price)``.

        A BUY lifts asks cheapest-first; a SELL hits bids highest-first.
        ``fillable_shares`` is what the book can actually absorb (<= the
        requested ``shares``); ``vwap`` is the volume-weighted price over them
        (the *effective* price the size pays/receives, not the top-of-book
        quote); ``marginal_price`` is the worst level touched — the minimum
        limit price that still fills the whole size. Returns ``(0.0, 0.0, 0.0)``
        when the relevant side is empty or ``shares`` is non-positive.
        """
        levels = self.asks if is_buy else self.bids
        if not levels or shares <= 0:
            return 0.0, 0.0, 0.0
        # Taker lifts the best price first: asks ascending, bids descending.
        ordered = sorted(levels, key=lambda lvl: lvl.price, reverse=not is_buy)
        remaining = shares
        cost = 0.0
        filled = 0.0
        marginal = 0.0
        for lvl in ordered:
            if remaining <= 1e-9:
                break
            take = min(remaining, lvl.size)
            cost += take * lvl.price
            filled += take
            marginal = lvl.price
            remaining -= take
        if filled <= 0:
            return 0.0, 0.0, 0.0
        return filled, cost / filled, marginal

    def depth_within(self, price_cap: float, is_buy: bool) -> float:
        """Total shares available no worse than ``price_cap`` on the taker side.

        For a BUY, sums ask sizes at price <= cap (the shares we could lift
        without paying more than the cap); for a SELL, bid sizes at price >= cap.
        """
        levels = self.asks if is_buy else self.bids
        if is_buy:
            return sum(lvl.size for lvl in levels if lvl.price <= price_cap + 1e-9)
        return sum(lvl.size for lvl in levels if lvl.price >= price_cap - 1e-9)


class ExitReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    PROFIT_TARGET = "PROFIT_TARGET"
    EDGE_EROSION = "EDGE_EROSION"
    TIME_DECAY = "TIME_DECAY"
    DUST_CLEANUP = "DUST_CLEANUP"
    CAPITAL_EFFICIENCY = "CAPITAL_EFFICIENCY"


class Fill(BaseModel):
    """A single execution fill from the CLOB or paper trader."""

    order_id: str
    market_id: str
    token_id: str = ""
    side: OrderSide
    token: TokenType = TokenType.YES
    size: float
    price: float
    fee: float = 0.0
    is_paper: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LivePosition(BaseModel):
    """Ground-truth position from on-chain / CLOB API."""

    market_id: str
    token_id: str = ""
    token: TokenType = TokenType.YES
    size: float
    avg_cost: float = 0.0
    current_price: float = 0.0
    market_question: str = ""
    category: str = ""

    @property
    def notional(self) -> float:
        return self.size * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.size * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.size


class Signal(BaseModel):
    market_id: str
    market_question: str = ""
    claude_prob: float
    claude_confidence: Confidence
    market_prob: float
    edge: float
    second_opinion_prob: float | None = None
    divergence: float | None = None
    evidence_summary: str = ""
    recommended_side: OrderSide | None = None
    recommended_size: float = 0
    # When set on a SELL signal, identifies which held token to exit.
    # Exchanges like Kalshi that support direct YES/NO sells read this to
    # build a proper SELL order instead of opening a new opposing position.
    exit_token: TokenType | None = None
    strategy_source: str = "llm"
    # Name-the-gap gate: WHY the market is mispriced (structural / behavioral /
    # informational mechanism). Empty = not yet audited; "none" = the model
    # could not name a mechanism (blocked when the gate is enabled).
    mispricing_reason: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
