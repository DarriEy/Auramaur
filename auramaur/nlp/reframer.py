"""Reframing parser — converts traditional market instruments into binary questions.

The core insight: an option's delta IS an implied probability.
A call option with delta 0.30 means the market prices P(stock > strike) at ~30%.

This module generates binary questions from stock/option data that our NLP pipeline
can process, then maps Claude's probability estimate back to a tradeable edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import structlog

from auramaur.exchange.models import Market

log = structlog.get_logger()


class ReframeType(str, Enum):
    """Types of binary questions we can construct from traditional instruments."""

    PRICE_ABOVE = "price_above"    # Call option: "Will X close above $Y?"
    PRICE_BELOW = "price_below"    # Put option: "Will X close below $Y?"
    EARNINGS_BEAT = "earnings_beat"  # "Will X beat earnings estimates?"
    MOVE_MORE_THAN = "move_pct"    # Straddle: "Will X move more than Z%?"


@dataclass
class OptionContract:
    """Minimal option data needed for reframing."""

    symbol: str                 # Underlying ticker (e.g. "AAPL")
    strike: float               # Strike price
    expiry: datetime            # Expiration date
    right: str                  # "C" (call) or "P" (put)
    delta: float                # Option delta (our implied probability)
    mid_price: float            # Mid-price of the option
    bid: float = 0.0
    ask: float = 0.0
    implied_vol: float = 0.0
    volume: int = 0
    open_interest: int = 0
    underlying_price: float = 0.0
    con_id: int = 0             # IB contract ID


@dataclass
class ReframedMarket:
    """A traditional instrument reframed as a binary question with mapping info."""

    market: Market              # The binary-question Market for our pipeline
    option: OptionContract      # The underlying option contract
    reframe_type: ReframeType   # How we generated the question
    trade_mapping: TradeMapping # How to convert binary decision → option order


@dataclass
class TradeMapping:
    """Maps a binary YES/NO decision back to an option trade.

    buy_yes_action: What to do when Claude says BUY YES (e.g. "buy_call")
    sell_yes_action: What to do when Claude says SELL YES (e.g. "buy_put")
    """

    buy_yes_action: str   # "buy_call" | "buy_put" | "sell_call" | "sell_put"
    sell_yes_action: str  # "buy_put" | "sell_call" | "buy_call" | "sell_put"
    contract_multiplier: int = 100  # Standard options = 100 shares per contract


def reframe_option_as_binary(option: OptionContract) -> ReframedMarket:
    """Convert an option contract into a binary-question Market.

    For calls: "Will {symbol} close above ${strike} by {expiry}?"
        - delta ≈ P(stock > strike)
        - BUY YES → buy the call
        - SELL YES → buy the put at same strike (or sell call if held)

    For puts: "Will {symbol} close below ${strike} by {expiry}?"
        - |delta| ≈ P(stock < strike)
        - BUY YES → buy the put
        - SELL YES → buy the call at same strike
    """
    expiry_str = option.expiry.strftime("%B %d, %Y")
    strike_str = f"${option.strike:,.2f}"

    if option.right == "C":
        reframe_type = ReframeType.PRICE_ABOVE
        question = (
            f"Will {option.symbol} close above {strike_str} by {expiry_str}?"
        )
        description = (
            f"Binary question derived from {option.symbol} call option at "
            f"{strike_str} strike, expiring {expiry_str}. "
            f"Current underlying price: ${option.underlying_price:,.2f}. "
            f"Implied volatility: {option.implied_vol * 100:.1f}%."
        )
        # Delta of call = P(stock > strike)
        implied_prob = abs(option.delta)
        trade_map = TradeMapping(
            buy_yes_action="buy_call",
            sell_yes_action="buy_put",
        )
    else:
        reframe_type = ReframeType.PRICE_BELOW
        question = (
            f"Will {option.symbol} close below {strike_str} by {expiry_str}?"
        )
        description = (
            f"Binary question derived from {option.symbol} put option at "
            f"{strike_str} strike, expiring {expiry_str}. "
            f"Current underlying price: ${option.underlying_price:,.2f}. "
            f"Implied volatility: {option.implied_vol * 100:.1f}%."
        )
        # |Delta of put| = P(stock < strike)
        implied_prob = abs(option.delta)
        trade_map = TradeMapping(
            buy_yes_action="buy_put",
            sell_yes_action="buy_call",
        )

    # Clamp probability to valid range
    implied_prob = max(0.01, min(0.99, implied_prob))

    # Build market ID: deterministic from contract details
    market_id = (
        f"IB:{option.symbol}:{option.strike}:{option.expiry.strftime('%Y%m%d')}"
        f":{option.right}"
    )

    # Spread from bid/ask as fraction of mid
    spread = 0.0
    if option.mid_price > 0 and option.bid > 0 and option.ask > 0:
        spread = (option.ask - option.bid) / option.mid_price

    market = Market(
        id=market_id,
        exchange="ibkr",
        ticker=option.symbol,
        question=question,
        description=description,
        category="options",
        end_date=option.expiry,
        active=True,
        outcome_yes_price=implied_prob,
        outcome_no_price=1.0 - implied_prob,
        volume=float(option.volume),
        liquidity=float(option.open_interest * option.mid_price * 100),
        spread=spread,
    )

    return ReframedMarket(
        market=market,
        option=option,
        reframe_type=reframe_type,
        trade_mapping=trade_map,
    )


def reframe_earnings_binary(
    symbol: str,
    earnings_date: datetime,
    underlying_price: float,
    implied_move_pct: float,
    atm_call: OptionContract | None = None,
) -> ReframedMarket | None:
    """Create a binary question about earnings outcome.

    "Will {symbol} beat earnings estimates on {date}?"

    Market probability is set to 0.50 (no strong prior) unless we can derive
    something from options skew.
    """
    if atm_call is None:
        return None

    question = f"Will {symbol} beat earnings estimates on {earnings_date.strftime('%B %d, %Y')}?"
    description = (
        f"Earnings binary for {symbol}. "
        f"Current price: ${underlying_price:,.2f}. "
        f"Options-implied move: ±{implied_move_pct:.1f}%. "
        f"Historical beat rate for S&P 500 companies: ~75%."
    )

    market_id = f"IB:{symbol}:EARN:{earnings_date.strftime('%Y%m%d')}"

    # Earnings beat rate baseline is ~75% for S&P 500
    # Skew between call/put IV can adjust this
    implied_prob = 0.50

    market = Market(
        id=market_id,
        exchange="ibkr",
        ticker=symbol,
        question=question,
        description=description,
        category="earnings",
        end_date=earnings_date,
        active=True,
        outcome_yes_price=implied_prob,
        outcome_no_price=1.0 - implied_prob,
        volume=float(atm_call.volume),
        liquidity=float(atm_call.open_interest * atm_call.mid_price * 100),
    )

    # For earnings, BUY YES (beat) → buy ATM call (benefits from up-move)
    trade_map = TradeMapping(
        buy_yes_action="buy_call",
        sell_yes_action="buy_put",
    )

    return ReframedMarket(
        market=market,
        option=atm_call,
        reframe_type=ReframeType.EARNINGS_BEAT,
        trade_mapping=trade_map,
    )


def select_interesting_strikes(
    options: list[OptionContract],
    underlying_price: float,
    max_contracts: int = 10,
) -> list[OptionContract]:
    """Select the most interesting option contracts for analysis.

    Prioritizes:
    1. Near-the-money options (delta 0.20 - 0.80) — most informative
    2. Higher volume/open interest — more liquid
    3. Reasonable expiry (7-90 days) — enough time for thesis to play out
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    scored: list[tuple[float, OptionContract]] = []

    for opt in options:
        # Filter: must have meaningful delta and pricing
        if abs(opt.delta) < 0.05 or abs(opt.delta) > 0.95:
            continue
        if opt.mid_price <= 0:
            continue

        # Days to expiry
        dte = (opt.expiry - now).days
        if dte < 7 or dte > 90:
            continue

        # Score components
        # 1. Delta proximity to 0.50 (ATM is most informative)
        delta_score = 1.0 - abs(abs(opt.delta) - 0.50) * 2  # peaks at 0.50

        # 2. Liquidity (log scale)
        import math
        liq_score = math.log1p(opt.volume + opt.open_interest) / 10.0

        # 3. Tight spread = better
        spread_score = 0.0
        if opt.mid_price > 0 and opt.ask > opt.bid:
            spread_pct = (opt.ask - opt.bid) / opt.mid_price
            spread_score = max(0, 1.0 - spread_pct * 5)  # Penalize wide spreads

        # 4. DTE sweet spot: 20-45 days
        dte_score = 1.0 - abs(dte - 30) / 60.0
        dte_score = max(0, dte_score)

        total = delta_score * 0.3 + liq_score * 0.3 + spread_score * 0.2 + dte_score * 0.2
        scored.append((total, opt))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [opt for _, opt in scored[:max_contracts]]
