"""Pure proposal contract for the structurally paper-only IBKR ETF book."""

from __future__ import annotations

import math
from dataclasses import dataclass

from auramaur.risk.ibkr_math import adverse_fill, risk_quantity, stop_distance


@dataclass(frozen=True)
class ETFExitProposal:
    symbol: str
    quantity: float
    price: float
    reason: str


@dataclass(frozen=True)
class ETFEntryProposal:
    symbol: str
    quantity: float
    price: float
    stop_price: float
    initial_risk_usd: float
    probability: float
    confidence: str


def exit_proposal(*, symbol: str, quantity: float, entry_price: float,
                  bid: float, ask: float, stored_stop: float, peak_gain_pct: float,
                  probability: float | None, stop_loss_pct: float,
                  take_profit_pct: float, trailing_stop_pct: float,
                  exit_probability: float, slippage_bps: float) -> ETFExitProposal | None:
    """Return a deterministic liquidation proposal, using production precedence."""
    values = (quantity, entry_price, bid, ask, stored_stop, peak_gain_pct,
              stop_loss_pct, take_profit_pct, trailing_stop_pct,
              exit_probability, slippage_bps)
    if (not symbol or any(not math.isfinite(v) for v in values)
            or quantity <= 0 or entry_price <= 0 or bid <= 0 or ask < bid):
        return None
    gain = (bid - entry_price) / entry_price * 100
    reason = None
    if (stored_stop > 0 and bid <= stored_stop) or gain <= -stop_loss_pct:
        reason = "stop_loss"
    elif gain >= take_profit_pct:
        reason = "take_profit"
    elif peak_gain_pct > 0 and peak_gain_pct - gain >= trailing_stop_pct:
        reason = "trailing_stop"
    elif (probability is not None and math.isfinite(probability)
          and probability < exit_probability):
        reason = "llm_bearish"
    if reason is None:
        return None
    return ETFExitProposal(symbol, quantity,
                           adverse_fill(bid, ask, "SELL", slippage_bps), reason)


def entry_proposal(*, symbol: str, bid: float, ask: float, probability: float | None,
                   confidence: str, confidence_rank: int, min_confidence_rank: int,
                   min_probability: float, annual_volatility: float | None,
                   momentum: float | None, remaining_deployment_usd: float,
                   remaining_class_usd: float, max_entry_usd: float,
                   fee_usd: float, slippage_bps: float, stop_vol_multiple: float,
                   min_stop_pct: float, risk_budget_usd: float,
                   open_risk_usd: float, max_portfolio_risk_usd: float,
                   controls_allow_entry: bool) -> ETFEntryProposal | None:
    """Propose paper exposure only after every upstream control has passed."""
    numeric = (bid, ask, min_probability, remaining_deployment_usd,
               remaining_class_usd, max_entry_usd, fee_usd, slippage_bps,
               stop_vol_multiple, min_stop_pct, risk_budget_usd,
               open_risk_usd, max_portfolio_risk_usd)
    if (not controls_allow_entry or not symbol or probability is None
            or annual_volatility is None or momentum is None
            or any(not math.isfinite(v) for v in numeric)
            or not math.isfinite(probability) or not math.isfinite(annual_volatility)
            or not math.isfinite(momentum) or bid <= 0 or ask < bid
            or confidence_rank < min_confidence_rank or probability < min_probability
            or annual_volatility < 0 or momentum <= 0):
        return None
    notional = min(max_entry_usd, remaining_deployment_usd, remaining_class_usd)
    if notional <= fee_usd:
        return None
    price = adverse_fill(bid, ask, "BUY", slippage_bps)
    distance = stop_distance(price, annual_volatility, stop_vol_multiple, min_stop_pct)
    quantity = min((notional - fee_usd) / price,
                   risk_quantity(risk_budget_usd, distance, 1, 1, fractional=True))
    if quantity <= 0:
        return None
    initial_risk = quantity * distance
    if open_risk_usd + initial_risk > max_portfolio_risk_usd:
        return None
    return ETFEntryProposal(symbol, quantity, price, price - distance,
                            initial_risk, probability, confidence)
