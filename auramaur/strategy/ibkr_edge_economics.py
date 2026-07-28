"""Whether a directional forecast is worth acting on, in dollars.

The previous IBKR book gated entries on a bare probability threshold
(``min_prob = 0.62``). That number carries no information about the instrument
or the cost of trading it, and the measured consequence was a $968 loss over
463 replayed round trips of which **$926 was commission** — a book paying
~30bps to act on ~6bps of expected edge.

The gate here is economic instead. Expected edge over the benchmark is

    E[excess return] = 2 * (p - base_rate) * E|move|

Going long on a view ``p`` earns ``(2p-1)*E|move|`` against a zero-drift world;
the benchmark's own unconditional up-rate earns ``(2b-1)*E|move|``; the
difference is what the forecast actually adds. ``E|move|`` for a zero-mean
normal over ``h`` sessions is ``sigma * sqrt(h/252) * sqrt(2/pi)``.

Measured 2026-07-27 with the LLM arms' *typical* conviction of |p-0.5| = 0.02:

    TLT   9.3% vol ->  4.2bps edge      SPY  12.7% ->  5.7bps
    QQQ  19.0% vol ->  8.6bps edge      GLD  28.4% -> 12.8bps
    SLV  64.4% vol -> 29.0bps edge

against ~30bps round-trip cost at $1,100 of capital. Only the highest-
volatility names can carry a small probability edge past the fee floor, which
is why ``required_conviction`` exists: it turns "is this instrument worth
forecasting at all" into a number that can be compared with what the model
actually produces.

Nothing here decides direction. It answers only: given a view of this
strength, on an instrument of this volatility, at this size — does acting beat
not acting?
"""

from __future__ import annotations

import math

# E|X| for X ~ N(0, s) is s * sqrt(2/pi).
_HALF_NORMAL = math.sqrt(2 / math.pi)


def horizon_sigma(annual_vol: float, horizon_sessions: int,
                  periods: int = 252) -> float:
    """Standard deviation of the return over the holding horizon."""
    if annual_vol <= 0 or horizon_sessions <= 0:
        return 0.0
    return annual_vol * math.sqrt(horizon_sessions / periods)


def expected_absolute_move(annual_vol: float, horizon_sessions: int,
                           periods: int = 252) -> float:
    """E|return| over the horizon, as a fraction."""
    return horizon_sigma(annual_vol, horizon_sessions, periods) * _HALF_NORMAL


def round_trip_cost_bps(notional_usd: float, *, commission_usd: float,
                        spread_bps: float, slippage_bps: float) -> float:
    """Total cost of getting in and out, in bps of notional.

    Commission is counted TWICE because it is charged per fill, and at small
    size its fixed minimum is the dominant term: IBKR's equity schedule is
    ``max(1.0, min(notional*0.001, 10.0))``, so a $1,100 position pays the $1
    floor — 9bps a leg — while a $10,000 position pays the marginal 10bps rate
    on a base ten times larger. Crossing costs half the spread each way, plus
    modelled slippage.
    """
    if notional_usd <= 0:
        return float("inf")
    commission = 2 * commission_usd / notional_usd * 10_000
    return commission + spread_bps + 2 * slippage_bps


def expected_edge_bps(probability: float, base_rate: float, annual_vol: float,
                      horizon_sessions: int, periods: int = 252) -> float:
    """Expected excess return over the benchmark, in bps. Signed.

    Negative means the forecast is BELOW the instrument's own drift, i.e. going
    long is worse than doing nothing — which for a long-only book is a skip,
    not a short.
    """
    move = expected_absolute_move(annual_vol, horizon_sessions, periods)
    return 2 * (probability - base_rate) * move * 10_000


def required_conviction(annual_vol: float, horizon_sessions: int,
                        cost_bps: float, *, margin: float = 2.0,
                        periods: int = 252) -> float:
    """|p - base_rate| needed before trading beats not trading.

    The universe filter. An instrument whose required conviction exceeds what
    the model has ever produced cannot be traded profitably by that model, at
    that size, at any threshold — and belongs out of the universe rather than
    in it with a tighter gate. On 2026-07-27 numbers ($1,100, ~30bps, margin 2)
    SPY needs 0.21 and SLV needs 0.041, against an observed model maximum of
    0.06.
    """
    move = expected_absolute_move(annual_vol, horizon_sessions, periods)
    if move <= 0:
        return float("inf")
    return margin * cost_bps / (2 * move * 10_000)


def clears_costs(probability: float, base_rate: float, annual_vol: float,
                 horizon_sessions: int, cost_bps: float, *,
                 margin: float = 2.0, periods: int = 252) -> bool:
    """The entry test: does expected edge beat cost by the required margin?

    ``margin`` is not decoration. The edge estimate rests on a forecast whose
    calibration is itself uncertain, so demanding only 1.0x cost would trade on
    a coin flip between edge and fees. 2.0x is the default because it is the
    smallest multiple at which a 50% overestimate of the edge still breaks even.
    """
    edge = expected_edge_bps(probability, base_rate, annual_vol,
                             horizon_sessions, periods)
    return edge > 0 and edge >= margin * cost_bps


def viable_universe(vols_by_symbol: dict[str, float], *, horizon_sessions: int,
                    cost_bps: float, max_conviction: float,
                    margin: float = 2.0) -> list[tuple[str, float]]:
    """Symbols whose required conviction is within the model's demonstrated
    reach, most-reachable first.

    ``max_conviction`` should be measured from the model's own record, not
    hoped for. The ETF arms' observed maximum |p-0.5| is 0.06; assuming more
    would put instruments in the universe that the model can never trade
    profitably.
    """
    out = []
    for symbol, vol in vols_by_symbol.items():
        needed = required_conviction(vol, horizon_sessions, cost_bps,
                                     margin=margin)
        if needed <= max_conviction:
            out.append((symbol, needed))
    return sorted(out, key=lambda item: item[1])
