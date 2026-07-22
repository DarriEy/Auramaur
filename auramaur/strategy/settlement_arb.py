"""Settlement-lag / known-outcome arb — FRED-first.

The disciplined generalization of the one edge that graduated (resolution_lens ×
weather). That edge is NOT category-specific — it is STRUCTURAL: read a PRECISE
numeric criterion against an AUTHORITATIVE data feed, and trade only the
CONVERGENCE, never the forecast. Weather had it (temp bin × Open-Meteo); econ
releases have it too (CPI/unemployment/payrolls × FRED).

This pillar trades a Polymarket econ market ONLY when:
  1. the referenced indicator's print for the market's reference period is
     ALREADY PUBLISHED by FRED (the official source), AND
  2. that published value DETERMINISTICALLY satisfies or fails the market's
     numeric criterion, AND
  3. the market has NOT yet repriced to ~0/1 (the settlement LAG is the edge).

No forecasting: if the print isn't out yet, or the criterion is compound /
ambiguous, it SKIPS. The LLM only EXTRACTS the predicate (indicator, operator,
threshold, period) — once, cached, adversarially verified; the RESOLVE step is a
pure deterministic compare against FRED. PAPER-FORCED, its own graduation cell.

WHERE THE EDGE LIVES (NBER w34702, 2026): liquid macro contracts reprice
intraday and are well-calibrated — so the settlement LAG survives mostly in
ILLIQUID, low-volume tail/bin contracts with stale prices. Because this pillar
holds to resolution (known-outcome convergence, no exit), illiquidity does not
block an exit, so the candidate scan keeps only a low DUST floor (min_liquidity)
to admit those tail bins rather than demanding liquid headline contracts. The
min_edge gate still drops anything that has already converged, so a liquid
un-repriced market (the easy fill) is never wrongly excluded.
"""

from __future__ import annotations

from auramaur.strategy.protocols import ExecutionMode

import json
import re

import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.exchange.models import Confidence, OrderSide, Signal
from auramaur.strategy.econ_pricing import (
    ECON_SERIES,
    EconSpec,
    kalshi_macro_predicate,
    spec_for_series,
)

log = structlog.get_logger()

# Indicators this pillar understands, by the registry key (see econ_pricing).
# Lexical hints route a market to a candidate series before the LLM extracts the
# precise predicate; the series the LLM names must be one of these.
_INDICATOR_HINTS: dict[str, tuple[str, ...]] = {
    "KXCPIYOY": ("cpi", "inflation", "consumer price"),
    "KXU3": ("unemployment", "jobless rate", "u-3", "unemployment rate"),
    "KXPAYROLLS": ("payroll", "nonfarm", "jobs added", "jobs report"),
}


def has_econ_trigger(question: str) -> bool:
    q = (question or "").lower()
    return any(h in q for hints in _INDICATOR_HINTS.values() for h in hints)


EXTRACT_PROMPT = """You convert a prediction-market question into a MACHINE-CHECKABLE econ predicate, or reject it. Be strict: only single, unambiguous numeric thresholds against ONE official indicator. Compound/qualified/multi-leg criteria -> reject.

Question: "{question}"
Resolution criteria: {description}
Today is {today} (interpret 2-digit years like '26 = 2026 relative to today).

The indicator MUST be exactly one of:
  KXCPIYOY    — US CPI year-over-year inflation rate, in percent
  KXU3        — US unemployment rate (U-3), in percent
  KXPAYROLLS  — US nonfarm payrolls month-over-month change, in THOUSANDS of jobs

Give the operator the YES outcome needs ('>=','>','<=','<','=='), the numeric
threshold (in the indicator's units above), and the REFERENCE PERIOD the print
is for as YYYY-MM (the month the data describes, NOT the announcement date). If
the question is not a single clean threshold on one of these indicators, set
indicator to "" (reject).

Respond with ONLY this JSON:
{{"indicator": "KXCPIYOY|KXU3|KXPAYROLLS|", "operator": ">=|>|<=|<|==", "threshold": <number>, "reference_period": "YYYY-MM", "confidence": <float 0-1>}}"""

VERIFY_PROMPT = """Adversarially check an extracted econ predicate against the market. Default to REFUTED unless the predicate EXACTLY captures the market's YES condition under a literal reading.

Question: "{question}"
Resolution criteria: {description}
Extracted predicate: indicator={indicator}, YES if value {operator} {threshold}, reference period {reference_period}.

Refute if: the indicator is wrong, the operator/threshold is off, the period is wrong, the units don't match (e.g. payrolls in thousands vs absolute), OR the market is actually compound/qualified so a single threshold can't decide it.

Respond with ONLY this JSON:
{{"verdict": "confirmed|refuted", "confidence": <float 0-1>, "why": "<one sentence>"}}"""


# ----------------------------------------------------------------------
# Deterministic resolve — pure functions (the well-tested core)
# ----------------------------------------------------------------------


def indicator_at_period(
    obs: list[tuple], spec: EconSpec, period: str
) -> float | None:
    """The indicator value FOR a reference period (YYYY-MM), or None if FRED has
    not published the data needed yet (the print isn't out -> undetermined).

    ``obs`` is the FRED ``get_observations`` output: (date, value) oldest-first,
    dated by the observation period. Pure — no I/O.
    """
    by_month: dict[str, float] = {}
    ordered: list[tuple[str, float]] = []
    for date, value in obs:
        key = f"{date.year:04d}-{date.month:02d}"
        by_month[key] = float(value)
        ordered.append((key, float(value)))

    if period not in by_month:
        return None  # the print for this period isn't out yet

    if spec.transform == "level":
        return by_month[period]

    if spec.transform == "yoy":
        y, m = period.split("-")
        prior = f"{int(y) - 1:04d}-{m}"
        base = by_month.get(prior)
        if base in (None, 0):
            return None
        return (by_month[period] / base - 1.0) * 100.0

    if spec.transform == "mom_change":
        idx = next((i for i, (k, _) in enumerate(ordered) if k == period), None)
        if idx is None or idx == 0:
            return None
        return (ordered[idx][1] - ordered[idx - 1][1]) * spec.scale

    return None


def _bin_half_width(threshold: float) -> float:
    """Half-width of the point-bin centered on ``threshold``, derived from the
    threshold's own decimal precision (Kalshi econ point-bins are one grid step
    wide: 3.8 -> the [3.75, 3.85) bin, half-width 0.05). Needed because the
    indicator is computed CONTINUOUSLY from FRED (e.g. CPI YoY = 3.7841%), so an
    exact ``== 3.8`` test never matches and every point-bin would price fair=0.
    """
    s = f"{threshold:.10f}".rstrip("0")
    decimals = len(s.split(".")[1]) if "." in s else 0
    return 0.5 * (10.0 ** -decimals)


def is_satisfied(value: float, operator: str, threshold: float) -> bool:
    """Does the published value satisfy the YES criterion?"""
    if operator == ">=":
        return value >= threshold
    if operator == ">":
        return value > threshold
    if operator == "<=":
        return value <= threshold
    if operator == "<":
        return value < threshold
    if operator == "==":
        # Point-bin: YES iff the continuous indicator rounds INTO this bin, i.e.
        # falls within half a grid step of the threshold — not exact equality.
        return abs(value - threshold) < _bin_half_width(threshold) + 1e-12
    return False


class SettlementArbPillar:

    # Uniform Strategy contract (see strategy/protocols.py).
    name = "settlement_arb"
    execution_mode = ExecutionMode.GATEWAY_SINGLE

    def __init__(self, db, settings, discovery, exchange, risk_manager,
                 pnl_tracker, fred_source, analyzer,
                 kalshi_discovery=None, kalshi_exchange=None) -> None:
        self._db = db
        self._settings = settings
        self._discovery = discovery
        self._exchange = exchange
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._fred = fred_source
        self._analyzer = analyzer
        self._gateway = ExecutionGateway(
            router=None, exchange=exchange, exchange_name="polymarket",
            settings=settings, db=db, pnl_tracker=pnl_tracker,
        )
        # Kalshi monthly macro bins (KXCPIYOY/KXU3/KXPAYROLLS) are where the
        # settlement lag actually lives — the Poly econ universe is all annual /
        # compound markets. Wired when the Kalshi venue is composed; its
        # predicate comes from structured strike fields (no LLM) and execution
        # routes through a Kalshi-named gateway.
        self._kalshi = kalshi_discovery
        self._kalshi_gateway = (
            ExecutionGateway(
                router=None, exchange=kalshi_exchange, exchange_name="kalshi",
                settings=settings, db=db, pnl_tracker=pnl_tracker,
            )
            if kalshi_exchange is not None else None
        )
        self._schema_ready = False
        # Per-cycle funnel-stage counters (where predicated markets died) —
        # reset each run_once, emitted on the cycle log line.
        self._stages: dict[str, int] = {}

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS settlement_extractions (
                   market_id TEXT PRIMARY KEY,
                   indicator TEXT NOT NULL DEFAULT '',
                   operator TEXT NOT NULL DEFAULT '',
                   threshold REAL,
                   reference_period TEXT NOT NULL DEFAULT '',
                   verified INTEGER NOT NULL DEFAULT 0,
                   checked_at TEXT NOT NULL DEFAULT (datetime('now')))"""
        )
        await self._db.commit()
        self._schema_ready = True

    async def run_once(self) -> int:
        cfg = self._settings.settlement_arb
        if not cfg.enabled or self._fred is None:
            return 0
        await self._ensure_schema()
        # Per-cycle FRED observation cache. Kalshi scanning yields ~250 candidate
        # bins but only ~3 distinct FRED series (CPIAUCNS/UNRATE/PAYEMS); without
        # this each candidate re-fetches in _maybe_enter, bursting ~250 calls/cycle
        # past FRED's rate limit (fred_observations_failed spike). Keyed by series.
        self._fred_cycle_cache = {}
        self._stages = {}
        markets = await self._candidates()
        entered = 0
        with_pred = 0
        for m in markets:
            if entered >= cfg.max_entries_per_cycle:
                break
            try:
                pred = await self._predicate(m)
                if pred is None:
                    continue
                with_pred += 1
                if await self._maybe_enter(m, pred):
                    entered += 1
            except Exception as e:
                log.debug("settlement_arb.market_error", market_id=m.id, error=str(e))
        # Always log the cycle — a scan that finds candidates but enters nothing
        # (the steady state while the Poly econ universe is all not-yet-settled
        # annual / compound-range markets) was previously SILENT, making the
        # pillar indistinguishable from dead. scanned/with_predicate/entered tell
        # the three stories apart: no candidates vs none determinable vs converged.
        log.info("settlement_arb.cycle", scanned=len(markets),
                 with_predicate=with_pred, entered=entered,
                 stages=dict(self._stages))
        return entered

    def _bump(self, stage: str) -> None:
        self._stages[stage] = self._stages.get(stage, 0) + 1

    async def _candidates(self) -> list:
        """Active Poly econ markets with a numeric-threshold shape, future-dated."""
        rows = await self._db.fetchall(
            """SELECT id, question, description, outcome_yes_price, outcome_no_price,
                      liquidity, category, clob_token_yes, clob_token_no, end_date
               FROM markets
               WHERE active = 1 AND exchange = 'polymarket'
                 AND (end_date IS NULL OR end_date >= strftime('%Y-%m-%dT%H:%M:%SZ','now'))""",
        )
        from auramaur.exchange.models import Market
        out: list[Market] = []
        for r in rows or []:
            if not has_econ_trigger(r["question"]):
                continue
            if not (0.0 < (r["outcome_yes_price"] or 0.0) < 1.0):
                continue
            if (r["liquidity"] or 0.0) < self._settings.settlement_arb.min_liquidity:
                continue
            out.append(Market(
                id=r["id"], exchange="polymarket", question=r["question"] or "",
                description=r["description"] or "",
                outcome_yes_price=r["outcome_yes_price"] or 0.0,
                outcome_no_price=r["outcome_no_price"] or 0.0,
                liquidity=r["liquidity"] or 0.0, volume=r["liquidity"] or 0.0,
                category=r["category"] or "economics",
                clob_token_yes=r["clob_token_yes"] or "",
                clob_token_no=r["clob_token_no"] or "",
            ))
        out.extend(await self._kalshi_candidates())
        return out

    async def _kalshi_candidates(self) -> list:
        """Kalshi monthly macro-bin markets for the registered econ series.

        Fetched live per-series (the bins aren't in the local market cache) and
        kept only when they carry a real two-sided price — a no-bid far-dated bin
        (yes_price collapses to 0/1) has no lag to capture and nothing to fill.
        No min_liquidity gate: the thesis is that the lag survives in the thin,
        low-volume bins, and the pillar holds to resolution so illiquidity never
        strands an exit. Predicate/operator come from strike fields downstream.
        """
        if self._kalshi is None or not hasattr(self._kalshi, "get_markets_by_series"):
            return []
        out: list = []
        for series in ECON_SERIES:
            try:
                markets = await self._kalshi.get_markets_by_series(series, limit=200)
            except Exception as e:
                log.debug("settlement_arb.kalshi_fetch_error", series=series, error=str(e))
                continue
            for km in markets or []:
                if 0.0 < (km.outcome_yes_price or 0.0) < 1.0:
                    out.append(km)
        return out

    async def _predicate(self, m) -> dict | None:
        """Resolve predicate for a market.

        Kalshi macro bins carry the predicate in structured strike fields, so it
        is parsed DETERMINISTICALLY (no LLM extract/verify, no cost, no parse
        error). Polymarket markets fall back to the cached LLM extract+verify.
        """
        if (getattr(m, "exchange", "") or "") == "kalshi":
            return kalshi_macro_predicate(m)
        row = await self._db.fetchone(
            "SELECT indicator, operator, threshold, reference_period, verified "
            "FROM settlement_extractions WHERE market_id = ?", (m.id,))
        if row is not None:
            if not row["indicator"] or not row["verified"]:
                return None
            return {"indicator": row["indicator"], "operator": row["operator"],
                    "threshold": row["threshold"],
                    "reference_period": row["reference_period"]}

        pred = await self._extract(m)
        verified = pred is not None and await self._verify(m, pred)
        await self._db.execute(
            """INSERT OR REPLACE INTO settlement_extractions
               (market_id, indicator, operator, threshold, reference_period, verified)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (m.id, (pred or {}).get("indicator", ""), (pred or {}).get("operator", ""),
             (pred or {}).get("threshold"), (pred or {}).get("reference_period", ""),
             1 if verified else 0))
        await self._db.commit()
        return pred if verified else None

    async def _extract(self, m) -> dict | None:
        if self._analyzer is None:
            return None
        from datetime import datetime, timezone
        raw = await self._analyzer._call_llm(EXTRACT_PROMPT.format(
            question=m.question, description=(m.description or "")[:2000],
            today=datetime.now(timezone.utc).strftime("%Y-%m-%d")))
        try:
            p = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            return None
        ind = str(p.get("indicator", "")).upper()
        if ind not in ECON_SERIES:
            return None
        if not re.fullmatch(r"\d{4}-\d{2}", str(p.get("reference_period", ""))):
            return None
        if p.get("operator") not in (">=", ">", "<=", "<", "=="):
            return None
        try:
            thr = float(p.get("threshold"))
        except (TypeError, ValueError):
            return None
        if float(p.get("confidence", 0.0) or 0.0) < self._settings.settlement_arb.min_extract_confidence:
            return None
        return {"indicator": ind, "operator": p["operator"], "threshold": thr,
                "reference_period": p["reference_period"]}

    async def _verify(self, m, pred: dict) -> bool:
        if self._analyzer is None:
            return False
        raw = await self._analyzer._call_llm(VERIFY_PROMPT.format(
            question=m.question, description=(m.description or "")[:2000],
            indicator=pred["indicator"], operator=pred["operator"],
            threshold=pred["threshold"], reference_period=pred["reference_period"]))
        try:
            v = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            return False
        return (str(v.get("verdict")) == "confirmed"
                and float(v.get("confidence", 0.0) or 0.0)
                >= self._settings.settlement_arb.verify_min_confidence)

    async def _fred_observations(self, series: str, history_n: int):
        """FRED observations for a series, memoized for the current cycle so a
        fan-out of same-series candidate bins makes ONE API call, not hundreds
        (the cache is reset at the top of each run_once)."""
        cache = getattr(self, "_fred_cycle_cache", None)
        if cache is None:
            return await self._fred.get_observations(series, n=history_n)
        if series not in cache:
            cache[series] = await self._fred.get_observations(series, n=history_n)
        return cache[series]

    async def _maybe_enter(self, m, pred: dict) -> bool:
        """Deterministic resolve + settlement-lag gate."""
        cfg = self._settings.settlement_arb
        spec = spec_for_series(pred["indicator"])
        if spec is None:
            self._bump("no_spec")
            return False
        obs = await self._fred_observations(spec.fred_series, cfg.history_n)
        if not obs:
            self._bump("no_obs")
            return False
        value = indicator_at_period(obs, spec, pred["reference_period"])
        if value is None:
            self._bump("print_pending")
            return False  # print not out yet -> undetermined, never forecast
        # Resolve on the figure as the agency REPORTS it (CPI YoY/U3 to 0.1pp,
        # payrolls to 1,000) — the bins settle on the rounded print, not the
        # continuous FRED-derived value, so a hair-off-boundary value would
        # otherwise resolve the wrong side of a strike.
        if spec.report_round_to:
            value = round(value / spec.report_round_to) * spec.report_round_to

        yes_locked = is_satisfied(value, pred["operator"], pred["threshold"])
        yes_price = m.outcome_yes_price
        # Fair is a HARD 0/1 — the print already decided it. The lag is the edge:
        # only trade the un-converged distance. Same edge-sign convention as the
        # graduated resolution_lens: edge>0 -> long YES (BUY); edge<0 -> short YES
        # = long NO (SELL). |edge| < min_edge means the market already converged.
        fair = 1.0 if yes_locked else 0.0
        edge = fair - yes_price
        if abs(edge) < cfg.min_edge:
            self._bump("converged")
            return False  # already converged — no lag to capture
        side = OrderSide.BUY if edge > 0 else OrderSide.SELL

        signal = Signal(
            market_id=m.id, claude_prob=fair,
            claude_confidence=Confidence.HIGH, market_prob=yes_price,
            edge=abs(edge) * 100.0,
            evidence_summary=(f"settlement_arb: {pred['indicator']} "
                              f"{pred['reference_period']} = {value:.2f} "
                              f"{pred['operator']} {pred['threshold']} -> "
                              f"{'YES' if yes_locked else 'NO'} locked"),
            recommended_side=side,
            strategy_source="settlement_arb",
            mispricing_reason="structural: official print already determines the criterion",
        )
        decision = await self._risk.evaluate(signal, m)
        if not decision.approved or decision.position_size <= 0:
            self._bump("risk_rejected")
            log.info("settlement_arb.risk_rejected", market_id=m.id, reason=decision.reason)
            return False
        size = min(decision.position_size, cfg.stake_usd)
        force_paper = cfg.paper or getattr(decision, "force_paper", False)
        gateway = (self._kalshi_gateway
                   if (getattr(m, "exchange", "") or "") == "kalshi"
                   and self._kalshi_gateway is not None
                   else self._gateway)
        res = await gateway.submit(TradeIntent(
            signal=signal, market=m, size_dollars=size, force_paper=force_paper))
        ok = res.status in ("filled", "paper", "partial", "pending")
        if not ok:
            self._bump("submit_failed")
        if ok:
            log.info("settlement_arb.entered", market_id=m.id, side=side.value,
                     locked="YES" if yes_locked else "NO", value=round(value, 2),
                     market_price=round(yes_price, 3), paper=getattr(res.result, "is_paper", True))
        return ok
