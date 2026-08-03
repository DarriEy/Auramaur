"""Stage-1 earnings-event replay for the event-driven IBKR book.

Implements EXACTLY the frozen specification in
``docs/ibkr-event-driven-design.md`` ("Stage-1 FROZEN specification,
2026-08-03") — sign-only long-only entries at the first tradeable close,
fixed $2,000 notional, N=5 session exits, the intl book's own cost
arithmetic, and a G2 verdict computed on the post-2024 test split only.
Parameters live in the spec, not here: this module has no tuning knobs
beyond what the spec pre-registered, and any rule change belongs in the
spec FIRST, with a date.

Like ``ibkr_replay``, this writes NOTHING. The forward-evidence tables
(`ibkr_paper_round_trips`, `ibkr_paper_daily_marks`) are the
pre-registered record the live gate reads; a replay that filled them
would manufacture the evidence the contract exists to demand.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

from auramaur.backtest.ibkr_replay import ReplayTrip, _equity_commission
from auramaur.risk.ibkr_math import adverse_fill

# Static FX per the frozen spec: keeps notional/commission arithmetic
# realistic without modelling FX P&L over the hold. LSE bars arrive in
# pence, hence the GBX scale. Frozen 2026-08-03 — not a market feed.
STATIC_FX_TO_USD: dict[str, float] = {
    "GBP": 0.0127,  # LSE quotes in GBX (pence)
    "EUR": 1.17,
    "USD": 1.0,
    "CAD": 0.71,
    "JPY": 0.0068,
    "HKD": 0.128,
    "AUD": 0.65,
}

NOTIONAL_USD = 2_000.0  # the cost-floor design input, frozen
HOLD_SESSIONS = 5       # primary rule; S2/S3 sensitivities are 10/21


@dataclass(frozen=True)
class EarningsEvent:
    home_listing: str
    report_date: str      # YYYY-MM-DD
    timing_class: str     # BMO | AMC | MID | UNKNOWN
    surprise_sign: int    # -1 | 0 | +1, from actual - estimate


def load_events(events_csv: str, timing_csv: str) -> list[EarningsEvent]:
    """Join the two frozen files into usable events.

    An event is usable iff estimate and actual are both present (the G1
    sign-only constraint needs both to form a sign); zero surprise is
    kept here with sign 0 so the caller's skip is visible in coverage
    counts rather than silent.
    """
    timing: dict[tuple[str, str], str] = {}
    with open(timing_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            timing[(row["home_listing"], row["event_datetime_et"])] = (
                row["timing_class"])
    events: list[EarningsEvent] = []
    with open(events_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            est, act = row["eps_estimate"], row["eps_actual"]
            if not est or not act or est == "nan" or act == "nan":
                continue
            try:
                diff = float(act) - float(est)
            except ValueError:
                continue
            sign = 1 if diff > 0 else -1 if diff < 0 else 0
            events.append(EarningsEvent(
                home_listing=row["home_listing"],
                report_date=row["event_datetime_et"][:10],
                timing_class=timing.get(
                    (row["home_listing"], row["event_datetime_et"]),
                    "UNKNOWN"),
                surprise_sign=sign,
            ))
    return events


def entry_session_index(dates: list[str], report_date: str,
                        timing_class: str) -> int | None:
    """Index into *dates* of the entry session per the frozen timing rule.

    BMO → the report date's own session if it exists, else the next one
    (a report on a home-market holiday trades the next session). Anything
    else → strictly the next session AFTER the report date. None when no
    such session exists in the data.
    """
    if timing_class == "BMO":
        for i, d in enumerate(dates):
            if d >= report_date:
                return i
        return None
    for i, d in enumerate(dates):
        if d > report_date:
            return i
    return None


def build_event_trips(
    events: list[EarningsEvent],
    bars_by_key: dict[str, list[tuple[str, float]]],
    currency_by_key: dict[str, str],
    *,
    assumed_spread_bps: float,
    slippage_bps: float,
    hold_sessions: int = HOLD_SESSIONS,
    long_only: bool = True,
    notional_usd: float = NOTIONAL_USD,
) -> tuple[list[ReplayTrip], dict[str, int]]:
    """Build one trip per eligible event. Returns (trips, coverage).

    Coverage counts every exclusion by reason so nothing is silently
    dropped — the spec's coverage-honesty clause.
    """
    half = assumed_spread_bps / 2 / 10_000
    coverage = {"eligible": 0, "no_bars": 0, "zero_surprise": 0,
                "negative_skipped": 0, "no_entry_session": 0,
                "no_exit_session": 0}
    trips: list[ReplayTrip] = []
    for ev in events:
        if ev.surprise_sign == 0:
            coverage["zero_surprise"] += 1
            continue
        if long_only and ev.surprise_sign < 0:
            coverage["negative_skipped"] += 1
            continue
        bars = bars_by_key.get(ev.home_listing)
        if not bars:
            coverage["no_bars"] += 1
            continue
        dates = [d for d, _ in bars]
        closes = [c for _, c in bars]
        i = entry_session_index(dates, ev.report_date, ev.timing_class)
        if i is None:
            coverage["no_entry_session"] += 1
            continue
        j = i + hold_sessions
        if j >= len(dates):
            coverage["no_exit_session"] += 1
            continue
        fx = STATIC_FX_TO_USD.get(currency_by_key.get(ev.home_listing, "USD"),
                                  1.0)
        side = 1 if ev.surprise_sign > 0 else -1
        entry_local = adverse_fill(
            closes[i] * (1 - half), closes[i] * (1 + half),
            "BUY" if side > 0 else "SELL", slippage_bps)
        exit_local = adverse_fill(
            closes[j] * (1 - half), closes[j] * (1 + half),
            "SELL" if side > 0 else "BUY", slippage_bps)
        quantity = notional_usd / (entry_local * fx)
        gross = side * (exit_local - entry_local) * quantity * fx
        commission = (_equity_commission(entry_local * quantity * fx)
                      + _equity_commission(exit_local * quantity * fx))
        coverage["eligible"] += 1
        trips.append(ReplayTrip(
            key=ev.home_listing, asset_class="earnings_event",
            entry_date=dates[i], exit_date=dates[j],
            entry_price=entry_local, exit_price=exit_local,
            quantity=quantity, gross_usd=gross, commission_usd=commission,
            reason=f"hold_{hold_sessions}"))
    return trips, coverage
