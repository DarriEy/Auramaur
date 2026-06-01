"""Path-B step 1 — basis-trade edge validation.

Backtests a delta-neutral cash-and-carry (long spot + short perpetual) using
REAL historical funding rates from Kraken Futures' public API. The edge is the
funding the short-perp leg collects; price exposure is hedged out by the spot
leg, so this models the funding harvest net of trading costs and the idle-margin
drag required to keep the short leg from being liquidated.

This is a research/validation tool — it places no orders and needs no keys.
Funding data is public.

    python scripts/research/basis_backtest.py

Outputs per asset and blended: gross vs after-cost annualized yield, % of
intervals funding was positive (short-perp collects), an annualized Sharpe of
the funding stream, and the worst peak-to-trough drawdown of cumulative funding.
Ends with a blunt go/no-go read.

Cost assumptions are conservative and stated up top — adjust to your real
Kraken fee tier and margin policy and re-run.
"""

from __future__ import annotations

import json
import math
import urllib.request
from datetime import datetime

# --- Cost & capital assumptions ----------------------------------------------
# Two scenarios so we don't strawman: conservative (taker, rotate quarterly) and
# optimistic (maker, rotate semi-annually). Each "turnover" = one full open+close
# of the notional on BOTH legs.
SCENARIOS = {
    "conservative (taker)": {"spot": 0.0025, "fut": 0.0005, "turnover": 4},
    "optimistic (maker)":   {"spot": 0.0016, "fut": 0.0002, "turnover": 2},
}
# Idle capital: spot notional (1.0x) plus margin posted on the short perp to
# survive up-moves without liquidation. 0.5x => total deployed = 1.5x notional.
MARGIN_BUFFER = 0.5

# USD-margined linear perps (the right contract for a USDC cash-and-carry).
SYMBOLS = ["PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD"]
FUNDING_URL = "https://futures.kraken.com/derivatives/api/v4/historicalfundingrates?symbol={}"


def _fetch_rates(symbol: str) -> list[tuple[datetime, float]]:
    """Return [(timestamp, relativeFundingRate)] for a symbol, or [] on failure."""
    try:
        with urllib.request.urlopen(FUNDING_URL.format(symbol), timeout=30) as r:
            data = json.load(r)
    except Exception as e:  # noqa: BLE001 — research tool, report and skip
        print(f"  {symbol}: fetch failed — {type(e).__name__}: {e}")
        return []
    out = []
    for row in data.get("rates", []):
        ts = row.get("timestamp")
        rel = row.get("relativeFundingRate")
        if ts is None or rel is None:
            continue
        out.append((datetime.fromisoformat(ts.replace("Z", "+00:00")), float(rel)))
    out.sort(key=lambda x: x[0])
    return out


def _intervals_per_year(rows: list[tuple[datetime, float]]) -> float:
    """Infer funding cadence from median spacing → annualization factor."""
    if len(rows) < 3:
        return 24 * 365
    gaps = sorted((rows[i + 1][0] - rows[i][0]).total_seconds() for i in range(len(rows) - 1))
    median_gap = gaps[len(gaps) // 2]
    if median_gap <= 0:
        return 24 * 365
    return (365 * 24 * 3600) / median_gap


def _analyze(symbol: str, rows: list[tuple[datetime, float]]) -> dict | None:
    if len(rows) < 100:
        return None
    rates = [r for _, r in rows]
    n = len(rates)
    ipy = _intervals_per_year(rows)
    span_days = (rows[-1][0] - rows[0][0]).total_seconds() / 86400

    mean_rate = sum(rates) / n
    gross_annual = mean_rate * ipy
    positives = [r for r in rates if r > 0]
    pct_positive = len(positives) / n
    # "Feast" intensity: annualized funding rate WHILE it's positive. Informs
    # whether a regime-gated overlay (deploy only when funding is rich) helps.
    pos_only_annual = (sum(positives) / len(positives)) * ipy if positives else 0.0

    # After-cost net on deployed capital, per fee scenario.
    net_by_scenario = {}
    for name, sc in SCENARIOS.items():
        annual_fee_drag = 2 * (sc["spot"] + sc["fut"]) * sc["turnover"]
        net_before_margin = gross_annual - annual_fee_drag
        net_by_scenario[name] = net_before_margin / (1 + MARGIN_BUFFER)

    # Carry Sharpe on DAILY-aggregated funding returns (annualized by sqrt(365)).
    # NOTE: this measures only funding-stream smoothness. It looks high because
    # carry is smooth right up until the liquidation tail — which it does NOT
    # capture. Treat as "how steady is the drip," not "how safe is the trade."
    per_day = max(1, round(ipy / 365))
    daily = [sum(rates[i:i + per_day]) for i in range(0, n, per_day)]
    if len(daily) > 2:
        dmean = sum(daily) / len(daily)
        dstd = math.sqrt(sum((d - dmean) ** 2 for d in daily) / (len(daily) - 1))
        carry_sharpe = (dmean / dstd) * math.sqrt(365) if dstd > 0 else float("nan")
    else:
        carry_sharpe = float("nan")

    # Worst drawdown of cumulative funding, in plain return units (fraction).
    # Captures negative-funding stretches where the short leg PAYS.
    cum = peak = max_dd = 0.0
    for r in rates:
        cum += r
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    return {
        "symbol": symbol,
        "span_days": span_days,
        "gross_annual": gross_annual,
        "pos_only_annual": pos_only_annual,
        "net_by_scenario": net_by_scenario,
        "pct_positive": pct_positive,
        "carry_sharpe": carry_sharpe,
        "worst_funding_dd": max_dd,  # fraction of notional, peak-to-trough
    }


def main() -> None:
    print("=" * 72)
    print("BASIS-TRADE EDGE VALIDATION — Kraken funding, delta-neutral cash & carry")
    print("=" * 72)
    sc_names = list(SCENARIOS)
    print(f"Capital multiplier: {1 + MARGIN_BUFFER:.1f}x (spot + margin buffer). Fee scenarios:")
    for name, sc in SCENARIOS.items():
        print(f"  {name:22} spot {sc['spot']:.2%} / fut {sc['fut']:.2%}, "
              f"{sc['turnover']}x turnover -> {2*(sc['spot']+sc['fut'])*sc['turnover']:.2%}/yr fee drag")
    print()

    results = []
    for sym in SYMBOLS:
        rows = _fetch_rates(sym)
        res = _analyze(sym, rows)
        if res:
            results.append(res)

    if not results:
        print("No data — aborting.")
        return

    hdr = (f"{'asset':10} {'days':>5} {'gross/yr':>9} "
           + " ".join(f"{n.split()[0][:9]:>10}" for n in sc_names)
           + f" {'%pos':>6} {'carryS':>7} {'wstDD':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        nets = " ".join(f"{r['net_by_scenario'][n]:9.1%} " for n in sc_names)
        print(f"{r['symbol']:10} {r['span_days']:5.0f} {r['gross_annual']:8.1%}  "
              f"{nets}{r['pct_positive']:5.0%} {r['carry_sharpe']:7.1f} {r['worst_funding_dd']:6.2%}")

    blended_gross = sum(r["gross_annual"] for r in results) / len(results)
    blended_pos = sum(r["pct_positive"] for r in results) / len(results)
    blended_net = {n: sum(r["net_by_scenario"][n] for r in results) / len(results) for n in sc_names}

    print("\n" + "=" * 72)
    print("GO / NO-GO READ  (trailing ~12 months — a known low-funding regime)")
    print("=" * 72)
    print(f"  Blended GROSS funding yield         : {blended_gross:.1%} / yr")
    for n in sc_names:
        print(f"  Net on capital, {n:22}: {blended_net[n]:+.1%} / yr")
    blended_pos_only = sum(r["pos_only_annual"] for r in results) / len(results)
    print(f"  Funding positive (short collects)   : {blended_pos:.0%} of the time")
    print(f"  Yield WHILE positive (feast rate)   : {blended_pos_only:.1%} / yr annualized")
    best_net = max(blended_net.values())
    verdict = "GO" if (best_net > 0.08 and blended_pos > 0.6) else "MARGINAL / NO-GO"
    print(f"\n  VERDICT: {verdict}")
    print("  (bar: >8%/yr net on capital in the best fee scenario AND funding +ve >60%)")


if __name__ == "__main__":
    main()
