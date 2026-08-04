"""Quantitative context for the Kraken spot desk call.

Why this exists. The directional read was a news-fed LLM asked for a
probability, gated on a confidence floor it was instructed never to reach — so
it produced LOW/MEDIUM_LOW, the MEDIUM floor rejected 100% of signals, and
Kraken has not traded since 2026-06-11. Loosening the floor alone would not
help: at Kraken's 0.26% per side a 3-day view was worth ~44bps against ~62bps
of round-trip cost, so it loses on expectation even when right about direction.

The answer is not a looser threshold, it is a better-informed decision. These
are the things a text-only model cannot derive from headlines and that a spot
desk actually looks at: where price sits in its own volatility, whether the
regime is calm or stressed, which way the book leans, and — the number that
decides everything — how far price must move before the trade is worth doing.

Everything here is PURE: candles and a book in, numbers out. No network, no
model, no side effects, so the arithmetic is testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from statistics import median, pstdev

# Hourly candles: 24 * 365.
HOURS_PER_YEAR = 8_760


@dataclass(frozen=True)
class SpotFeatures:
    """One pair's quantitative state. Every field is model-readable."""
    pair: str
    price: float
    spread_bps: float
    breakeven_move_pct: float
    momentum_24h_sigma: float | None
    momentum_72h_sigma: float | None
    momentum_168h_sigma: float | None
    realized_vol_annual_pct: float | None
    vol_regime: str
    range_position_pct: float | None
    drawdown_from_high_pct: float | None
    book_imbalance: float | None
    expected_move_pct: float | None
    edge_needed_prob: float | None

    def as_prompt_block(self) -> str:
        """Compact, unambiguous rendering for the model."""
        def show(value, suffix="", nd=2):
            return "n/a" if value is None else f"{value:.{nd}f}{suffix}"
        return (
            f"{self.pair}: price={self.price:.6g} spread={self.spread_bps:.1f}bps\n"
            f"  momentum (vol-normalised): 24h={show(self.momentum_24h_sigma)}"
            f" 72h={show(self.momentum_72h_sigma)}"
            f" 168h={show(self.momentum_168h_sigma)}\n"
            f"  realised vol={show(self.realized_vol_annual_pct, '%', 1)} annual"
            f" ({self.vol_regime} regime)\n"
            f"  position in 7d range={show(self.range_position_pct, '%', 0)}"
            f", drawdown from 7d high={show(self.drawdown_from_high_pct, '%')}\n"
            f"  book imbalance={show(self.book_imbalance, '', 3)}"
            f" (positive = bid-heavy)\n"
            f"  COST TO TRADE: needs a {self.breakeven_move_pct:.2f}% move to break "
            f"even; expected move over the horizon is "
            f"{show(self.expected_move_pct, '%')}, so this is only worth doing at "
            f"P(up) >= {show(self.edge_needed_prob, '', 3)}"
        )


def _log_returns(closes: list[float]) -> list[float]:
    return [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]


def realized_vol_annual(closes: list[float]) -> float | None:
    """Annualised realised volatility from hourly closes, as a percentage."""
    returns = _log_returns(closes)
    if len(returns) < 24:
        return None
    vol = pstdev(returns) * math.sqrt(HOURS_PER_YEAR)
    return vol * 100 if math.isfinite(vol) and vol > 0 else None


def normalized_momentum_sigma(closes: list[float], hours: int) -> float | None:
    """Return over ``hours``, divided by what volatility alone would produce.

    Dimensionless, so BTC and SOL are directly comparable and a 3% BTC move is
    not confused with a 3% SOL move. This is the same construction the IBKR
    books use, on an hourly clock.
    """
    if len(closes) <= hours:
        return None
    returns = _log_returns(closes)
    if len(returns) < 24:
        return None
    sigma = pstdev(returns)
    if sigma <= 0:
        return None
    move = math.log(closes[-1] / closes[-1 - hours])
    return move / (sigma * math.sqrt(hours))


def vol_regime(closes: list[float], window: int = 168) -> str:
    """Is volatility elevated against this pair's own recent history?

    Named rather than numeric because the model reasons better about "stressed"
    than about 0.87 — and the comparison is to the pair's OWN median, so it
    carries no cross-asset assumption.
    """
    returns = _log_returns(closes)
    if len(returns) < window + 24:
        return "unknown"
    recent = pstdev(returns[-24:])
    hourly = [pstdev(returns[i:i + 24]) for i in range(0, len(returns) - 24, 24)]
    baseline = median([h for h in hourly if h > 0]) if hourly else 0.0
    if baseline <= 0 or recent <= 0:
        return "unknown"
    ratio = recent / baseline
    if ratio >= 1.75:
        return "stressed"
    if ratio >= 1.25:
        return "elevated"
    if ratio <= 0.6:
        return "quiet"
    return "normal"


def range_position(closes: list[float], window: int = 168) -> float | None:
    """Where price sits in its recent range: 0% at the low, 100% at the high."""
    recent = closes[-window:]
    if len(recent) < 24:
        return None
    low, high = min(recent), max(recent)
    if high <= low:
        return None
    return (closes[-1] - low) / (high - low) * 100


def drawdown_from_high(closes: list[float], window: int = 168) -> float | None:
    recent = closes[-window:]
    if len(recent) < 24:
        return None
    high = max(recent)
    return None if high <= 0 else (closes[-1] / high - 1) * 100


def book_imbalance(bids, asks, levels: int = 10) -> float | None:
    """(bid size - ask size) / total, over the top levels. Range [-1, 1].

    Positive means resting demand outweighs supply. A shallow, one-sided book is
    also the reason a "cheap" spread can still be expensive to trade through.
    """
    bid_size = sum(float(size) for _, size in list(bids)[:levels])
    ask_size = sum(float(size) for _, size in list(asks)[:levels])
    total = bid_size + ask_size
    return None if total <= 0 else (bid_size - ask_size) / total


def breakeven_move_pct(spread_bps: float, fee_pct_per_side: float,
                       slippage_bps: float = 0.0) -> float:
    """Percentage move needed just to cover the round trip.

    THE number the old design never told the model. Kraken charges 0.26% a side,
    so a round trip starts ~52bps underwater before the spread is crossed twice
    — which is why a 3-day view worth ~44bps was still a losing trade.
    """
    return 2 * fee_pct_per_side + (spread_bps + 2 * slippage_bps) / 100.0


def expected_move_pct(realized_vol_annual_pct: float | None,
                      horizon_hours: float) -> float | None:
    """E|move| over the horizon for a zero-mean normal, as a percentage."""
    if not realized_vol_annual_pct or horizon_hours <= 0:
        return None
    sigma = realized_vol_annual_pct * math.sqrt(horizon_hours / HOURS_PER_YEAR)
    return sigma * math.sqrt(2 / math.pi)


def edge_needed_prob(breakeven_pct: float, expected_pct: float | None,
                     margin: float = 1.0) -> float | None:
    """P(up) at which the trade stops losing to its own costs.

    Expected excess return is 2*(p - 0.5)*E|move|; set that equal to
    ``margin`` x breakeven and solve. Returns None when no probability
    suffices, which is itself the answer: at that horizon the trade cannot pay.
    """
    if not expected_pct or expected_pct <= 0:
        return None
    needed = 0.5 + margin * breakeven_pct / (2 * expected_pct)
    return None if needed >= 1.0 else needed


def build_features(pair: str, closes: list[float], *, bid: float, ask: float,
                   fee_pct_per_side: float, horizon_hours: float,
                   bids=(), asks=(), slippage_bps: float = 0.0,
                   margin: float = 1.0) -> SpotFeatures:
    """Assemble one pair's context. Missing inputs degrade to None, never guess."""
    price = (bid + ask) / 2 if bid > 0 and ask > 0 else (closes[-1] if closes else 0.0)
    spread_bps = ((ask - bid) / price * 10_000) if price > 0 and ask > bid else 0.0
    vol = realized_vol_annual(closes)
    breakeven = breakeven_move_pct(spread_bps, fee_pct_per_side, slippage_bps)
    expected = expected_move_pct(vol, horizon_hours)
    return SpotFeatures(
        pair=pair, price=price, spread_bps=spread_bps,
        breakeven_move_pct=breakeven,
        momentum_24h_sigma=normalized_momentum_sigma(closes, 24),
        momentum_72h_sigma=normalized_momentum_sigma(closes, 72),
        momentum_168h_sigma=normalized_momentum_sigma(closes, 168),
        realized_vol_annual_pct=vol,
        vol_regime=vol_regime(closes),
        range_position_pct=range_position(closes),
        drawdown_from_high_pct=drawdown_from_high(closes),
        book_imbalance=book_imbalance(bids, asks),
        expected_move_pct=expected,
        edge_needed_prob=edge_needed_prob(breakeven, expected, margin),
    )


def desk_prompt(features: list[SpotFeatures], *, horizon_hours: float,
                budget_usd: float, open_positions: list[dict],
                recent_pnl_usd: float) -> str:
    """The desk's situation report.

    Deliberately includes the BOOK — open positions and recent P&L — because a
    desk decides differently when it is already long than when it is flat, and
    the previous per-pair call could not see either.
    """
    lines = [
        f"Horizon: {horizon_hours:.0f} hours. Budget: ${budget_usd:,.2f}.",
        f"Realised P&L on this book so far: ${recent_pnl_usd:+,.2f}.",
    ]
    if open_positions:
        lines.append("Open positions:")
        for position in open_positions:
            lines.append(
                f"  {position.get('pair')}: {position.get('quantity')} @ "
                f"{position.get('entry')} (unrealised "
                f"${float(position.get('unrealized_usd') or 0):+,.2f})")
    else:
        lines.append("Open positions: none — the book is flat.")
    lines.append("")
    lines.append("Market state:")
    lines.extend(f.as_prompt_block() for f in features)
    return "\n".join(lines)


def feature_dict(features: SpotFeatures) -> dict:
    """Serialisable form, for logging what the model was actually shown."""
    return asdict(features)


__all__ = [
    "SpotFeatures", "build_features", "desk_prompt", "feature_dict",
    "realized_vol_annual", "normalized_momentum_sigma", "vol_regime",
    "range_position", "drawdown_from_high", "book_imbalance",
    "breakeven_move_pct", "expected_move_pct", "edge_needed_prob",
]
