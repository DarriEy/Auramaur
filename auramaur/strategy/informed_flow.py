"""Informed-flow detection on Kalshi trades — the DATA layer for the
abnormal-trade-size (ATS) follower.

Research basis (Delvecchio, CMC thesis #4166, 2026; corroborated by Bartlett &
O'Hara, "Adverse Selection in Prediction Markets: Evidence from Kalshi", 41.6M
trades): abnormal trade size proxies non-liquidity-motivated (informed) order
flow, which predicts the resolution and strengthens near it. The follower
strategy mimics the informed side rather than forecasting.

This module is the DATA layer ONLY — a pure detector over a Kalshi trades list,
plus a thin tape wrapper that pulls trades via the Kalshi client. It does NOT
trade. The follower pillar that consumes these signals is a separate, strictly
PAPER-FORCED step (the evidence is a single in-sample thesis = medium confidence;
and adverse selection means the edge depends on reliably being on the informed,
not the picked-off, side — so it earns its own graduation cell before any live).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import structlog

log = structlog.get_logger()


@dataclass(frozen=True)
class InformedFlowSignal:
    """Result of scanning one market's recent trade tape."""

    has_signal: bool
    informed_side: str | None    # 'yes' | 'no' | None
    abnormal_count: int          # number of abnormally-large trades
    baseline_size: float         # median trade size (the normal-flow yardstick)
    signal_volume: float         # net abnormal contracts on the informed side
    sample: int                  # trades examined


_NO_SIGNAL = InformedFlowSignal(False, None, 0, 0.0, 0.0, 0)


def _size(t: dict) -> float:
    # Kalshi migrated the per-trade contract count `count` -> `count_fp` (a
    # fixed-point string, e.g. "225.25"), the same *_fp migration that hit market
    # parsing. Reading the dead `count` made every trade size 0, so the detector
    # ALWAYS returned no-signal — informed_flow could never fire. Prefer count_fp,
    # fall back to the legacy field.
    try:
        return float(t.get("count_fp", t.get("count", 0)) or 0)
    except (TypeError, ValueError):
        return 0.0


def _side(t: dict) -> str | None:
    s = str(t.get("taker_side", "")).lower()
    return s if s in ("yes", "no") else None


def detect_informed_flow(
    trades: list[dict],
    *,
    min_sample: int = 20,
    size_mult: float = 3.0,
    min_dominance: float = 0.6,
) -> InformedFlowSignal:
    """Flag abnormally-large trades vs the market's OWN recent size baseline and
    infer the informed direction. Pure — no I/O.

    - ``min_sample``: need at least this many sized trades for a stable baseline
      (thin tapes give no signal — a single big print in a 3-trade market is
      noise, not information).
    - ``size_mult``: a trade is abnormal when its size >= ``size_mult`` x the
      median trade size (the non-liquidity / ATS proxy).
    - ``min_dominance``: the informed side must carry at least this share of the
      abnormal volume, else the large flow is two-sided / inconclusive -> no
      signal (don't follow a wash).
    """
    sized = [(s, side) for t in trades
             if (s := _size(t)) > 0 and (side := _side(t)) is not None]
    if len(sized) < min_sample:
        return _NO_SIGNAL

    baseline = median(s for s, _ in sized)
    if baseline <= 0:
        return _NO_SIGNAL
    threshold = size_mult * baseline

    abn_yes = sum(s for s, side in sized if s >= threshold and side == "yes")
    abn_no = sum(s for s, side in sized if s >= threshold and side == "no")
    abn_count = sum(1 for s, _ in sized if s >= threshold)
    total_abn = abn_yes + abn_no
    if total_abn <= 0:
        return _NO_SIGNAL

    if abn_yes >= min_dominance * total_abn:
        side, net = "yes", abn_yes
    elif abn_no >= min_dominance * total_abn:
        side, net = "no", abn_no
    else:
        # Large flow on both sides — inconclusive, don't follow.
        return InformedFlowSignal(False, None, abn_count, baseline, 0.0, len(sized))

    return InformedFlowSignal(True, side, abn_count, baseline, net, len(sized))


class KalshiTradeTape:
    """Thin wrapper: pull a market's recent trades via the Kalshi client and run
    :func:`detect_informed_flow`. Keeps the I/O out of the pure detector so the
    detection logic stays unit-testable. Fail-soft — a fetch error yields no
    signal, never an exception into the scan loop."""

    def __init__(self, exchange) -> None:
        self._exchange = exchange  # a KalshiExchange exposing get_trades()

    async def informed_flow(self, ticker: str, *, limit: int = 200,
                            **kwargs) -> InformedFlowSignal:
        try:
            trades = await self._exchange.get_trades(ticker, limit=limit)
        except Exception as e:
            log.debug("informed_flow.fetch_failed", ticker=ticker, error=str(e))
            return _NO_SIGNAL
        return detect_informed_flow(trades or [], **kwargs)
