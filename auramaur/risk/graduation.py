"""Graduation ladder — capital earned per (strategy × category) cell.

Phase 3 of the edge-first redesign. The static edge map (blocked_categories,
per-strategy paper flags) is hand-maintained and goes stale; this replaces it
with a mechanism: every (strategy_source × category) cell EARNS live capital
from its measured record in the pnl_ledger, and loses it again on decay.

The ladder (mode=enforce):

  * observations are aggregated by market before evaluation
  * cell has >= min_markets independent LIVE markets in the window and its
    one-sided mean-P&L lower confidence bound is positive -> LIVE, full size
  * else the same evidence contract is applied to PAPER markets -> probation
  * else -> PAPER (unproven; exploration happens on paper)

mode=observe computes and logs the same decision but never changes behavior —
the rollout default, so the operator can read `auramaur graduation` and flip
to enforce deliberately. mode=off skips entirely.

Safety properties:
  * ENTRIES ONLY. Exits never pass through RiskManager.evaluate (they flow
    through PortfolioTracker.check_exits -> direct SELL orders), so a
    demotion can never strand an open position — verified 2026-06-09.
  * Graduation only ever RESTRICTS (paper-force or shrink); it never loosens
    a risk check and never upsizes beyond the risk manager's Kelly size.
  * Exempt strategies (arbitrage, market_maker by default) bypass the
    ladder: they are structural/two-sided, not directional conviction.

LEDGER CONVENTION, and it is the whole reason the SQL below reads SUM(pnl)
rather than SUM(pnl - fees). ``pnl_ledger.pnl`` is ALREADY NET OF FEES at
every one of its six writers — pnl.py `(price - avg_cost) * size - fill.fee`,
ledger.py the same, kalshi_settlements `payout - cost - fee`,
instrument_booking `pnl=-fee_usd, fees=0.0`, manual_trades and
resolution_tracker `fees=0.0`. The `fees` column is the informational
breakdown of what has already been deducted, not a second charge —
readiness.check_pnl_after_fees says so outright, commenting `# already net of
fees` on SUM(pnl) and then recovering gross as `net + fees`.

Subtracting it again charged every fee twice on the paper->live PROMOTION
path. The error is conservative (it understates P&L, so it holds a cell back
rather than promoting one that has not earned it), which is why it survived
unnoticed, but a promotion gate that judges cells on a number that is not
their P&L is judging the wrong thing. Corrected 2026-08-06 alongside the same
defect in ledger_report's benchmark panel; measured on the live DB the change
moves exactly one of 127 cells (kraken_directional/kraken_spot, -$43.98 ->
-$36.53, negative under both conventions and carrying no decision_snapshots),
so no verdict flips today.
"""

from __future__ import annotations
from datetime import datetime
from statistics import NormalDist

import math
import time
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


@dataclass(frozen=True)
class CellDecision:
    force_paper: bool
    size_multiplier: float
    status: str          # live | probation | demoted | paper_negative | unproven | exempt | observe:<...>
    reason: str


_LIVE_FULL = CellDecision(False, 1.0, "live", "live evidence lower bound positive")
_EXEMPT = CellDecision(False, 1.0, "exempt", "strategy exempt from graduation")


class GraduationLadder:
    """Computes and caches per-cell graduation decisions from pnl_ledger."""

    def __init__(self, db, settings) -> None:
        self._db = db
        self._settings = settings
        self._cache: dict[tuple[str, str], tuple[float, CellDecision]] = {}
        self._breadth: tuple[float, int] | None = None  # (monotonic_ts, count)

    # ------------------------------------------------------------------

    async def decide(self, strategy_source: str, category: str) -> CellDecision:
        cfg = self._settings.graduation
        if cfg.mode == "off":
            return _LIVE_FULL
        strategy = strategy_source or "llm"
        if strategy in set(cfg.exempt_strategies):
            return _EXEMPT
        category = category or ""

        key = (strategy, category)
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit and now - hit[0] < cfg.cache_seconds:
            return hit[1]

        decision = await self._compute(strategy, category)
        if cfg.mode == "observe":
            log.info("graduation.observe", strategy=strategy, category=category,
                     would_force_paper=decision.force_paper,
                     would_multiply=decision.size_multiplier,
                     status=decision.status, reason=decision.reason)
            decision = CellDecision(
                False, 1.0, f"observe:{decision.status}", decision.reason)
        self._cache[key] = (now, decision)
        return decision

    # ------------------------------------------------------------------

    async def _cell_stats(self, strategy: str, category: str) -> dict:
        # Strategy-grain election: aggregate the whole strategy's record
        # (category ignored) — every cell of the strategy then shares one
        # decision. See GraduationConfig.strategy_level_strategies.
        if strategy in self._settings.graduation.strategy_level_strategies:
            rows = await self._db.fetchall(
                # SUM(pnl), not SUM(pnl - fees): pnl is already net at every
                # writer. See LEDGER CONVENTION in the module docstring.
                """SELECT market_id, is_paper, SUM(pnl) AS pnl
                   FROM pnl_ledger
                   WHERE strategy_source = ?
                     AND realized_at >= datetime('now', ?)
                   GROUP BY market_id, is_paper""",
                (strategy,
                 f"-{int(self._settings.graduation.window_days)} days"),
            )
        else:
            rows = await self._db.fetchall(
                # SUM(pnl), not SUM(pnl - fees) — see the module docstring.
                """SELECT market_id, is_paper, SUM(pnl) AS pnl
                   FROM pnl_ledger
                   WHERE strategy_source = ? AND category = ?
                     AND realized_at >= datetime('now', ?)
                   GROUP BY market_id, is_paper""",
                (strategy, category,
                 f"-{int(self._settings.graduation.window_days)} days"),
            )
        live = [float(r["pnl"] or 0.0) for r in rows or [] if not r["is_paper"]]
        paper = [float(r["pnl"] or 0.0) for r in rows or [] if r["is_paper"]]

        def lower_bound(values: list[float]) -> float:
            if len(values) < 2:
                # Sample variance is undefined below two independent markets;
                # -inf prevents a single outcome from claiming evidence.
                return float("-inf")
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            return mean - self._settings.graduation.confidence_z * math.sqrt(
                variance / len(values))

        return {
            "live_n": len(live), "live_pnl": sum(live),
            "live_lcb": lower_bound(live),
            "paper_n": len(paper), "paper_pnl": sum(paper),
            "paper_lcb": lower_bound(paper),
        }

    async def _prospective_stats(self, strategy: str, category: str) -> dict:
        """Independent-family, executable, locked-holdout evidence."""
        cfg = self._settings.graduation
        evidence = list(cfg.credible_fill_evidence)
        marks = ", ".join("?" for _ in evidence)
        filters = ""
        tail: list[object] = []
        if cfg.require_executable_fills:
            filters += f" AND d.filled = 1 AND d.fill_evidence IN ({marks})"
            tail.extend(evidence)
        if strategy not in cfg.strategy_level_strategies:
            filters += " AND COALESCE(m.category, '') = ?"
            tail.append(category)
        rows = await self._db.fetchall(
            f"""WITH latest AS (
                    SELECT strategy_version FROM strategy_experiments
                    WHERE strategy_source = ? ORDER BY registered_at DESC LIMIT 1
                ), pnl AS (
                    -- SUM(pnl): already net of fees at every writer. See the
                    -- module docstring. This is the promotion path.
                    SELECT market_id, is_paper, SUM(pnl) AS net_pnl
                    FROM pnl_ledger WHERE strategy_source = ?
                    GROUP BY market_id, is_paper
                )
                SELECT d.market_id, d.is_paper, d.observed_at,
                       o.resolved_at, d.requested_size,
                       COALESCE(NULLIF(d.event_family, ''), d.market_id) AS family,
                       ((d.reference_price-o.outcome)*(d.reference_price-o.outcome)
                        -(d.fair_probability-o.outcome)*(d.fair_probability-o.outcome)) AS brier_edge,
                       COALESCE(p.net_pnl, 0) AS net_pnl
                FROM decision_snapshots d
                JOIN latest x ON x.strategy_version = d.strategy_version
                JOIN market_outcomes o ON o.event_key=lower(d.venue)||':'||d.market_id
                LEFT JOIN markets m ON m.id=d.market_id
                LEFT JOIN pnl p ON p.market_id=d.market_id AND p.is_paper=d.is_paper
                WHERE d.strategy_source=? AND d.is_holdout=1
                  AND d.observed_at >= datetime('now', ?) {filters}
                ORDER BY d.observed_at, d.id""",
            (strategy, strategy, strategy, f"-{int(cfg.window_days)} days", *tail),
        )
        grouped: dict[int, dict[str, dict]] = {0: {}, 1: {}}
        for row in rows or []:
            grouped[int(bool(row["is_paper"]))].setdefault(row["family"], row)
        alpha = cfg.familywise_alpha / max(1, cfg.max_hypotheses*cfg.sequential_looks_per_window)
        z = max(cfg.confidence_z, NormalDist().inv_cdf(1.0-alpha))

        def summarize(items: list[dict]) -> dict:
            pnl = []
            for row in items:
                net_pnl = float(row["net_pnl"] or 0)
                if cfg.require_cash_benchmark:
                    opened = datetime.fromisoformat(
                        str(row["observed_at"]).replace("Z", "+00:00"))
                    resolved = datetime.fromisoformat(
                        str(row["resolved_at"]).replace("Z", "+00:00"))
                    hold_years = max(
                        0.0, (resolved - opened).total_seconds()
                    ) / (365.25 * 86400)
                    net_pnl -= (
                        max(0.0, float(row["requested_size"] or 0))
                        * self._settings.benchmark.risk_free_annual_rate
                        * hold_years
                    )
                pnl.append(net_pnl)
            brier = [float(r["brier_edge"] or 0) for r in items]
            def lcb(values: list[float]) -> float:
                if len(values) < 2:
                    return float("-inf")
                mean = sum(values)/len(values)
                variance = sum((x-mean)**2 for x in values)/(len(values)-1)
                return mean-z*math.sqrt(variance/len(values))
            dates = [datetime.fromisoformat(str(r["observed_at"]).replace("Z", "+00:00")) for r in items]
            return {"n": len(items), "pnl": sum(pnl), "lcb": lcb(pnl),
                    "brier_lcb": lcb(brier),
                    "days": (max(dates)-min(dates)).days if len(dates)>1 else 0,
                    "regimes": len({(d.year, d.month) for d in dates})}
        live = summarize(list(grouped[0].values()))
        paper = summarize(list(grouped[1].values()))
        return ({f"live_{k}": v for k, v in live.items()} |
                {f"paper_{k}": v for k, v in paper.items()})
    async def _paper_breadth(self) -> int:
        """Concurrent open PAPER/exploratory positions — the spray breadth. Cached
        for cache_seconds (a soft cap doesn't need an exact live count)."""

        now = time.monotonic()
        if self._breadth and now - self._breadth[0] < self._settings.graduation.cache_seconds:
            return self._breadth[1]
        try:
            row = await self._db.fetchone(
                "SELECT COUNT(*) AS n FROM portfolio WHERE is_paper = 1 AND size > 0")
            n = int(row["n"] or 0) if row else 0
        except Exception:
            n = 0  # fail-open: a count failure must not block trading
        self._breadth = (now, n)
        return n

    async def _compute(self, strategy: str, category: str) -> CellDecision:
        cfg = self._settings.graduation
        s = (await self._prospective_stats(strategy, category)
             if cfg.prospective_only else await self._cell_stats(strategy, category))
        min_markets = cfg.min_markets_overrides.get(strategy, cfg.min_markets)
        min_evidence = cfg.min_paired_forecasts or min_markets

        def qualifies(mode: str) -> bool:
            return (s[f"{mode}_n"] >= min_evidence
                    and s.get(f"{mode}_days", 0) >= cfg.min_calendar_days
                    and s.get(f"{mode}_regimes", 1) >= cfg.min_regime_months
                    and s[f"{mode}_lcb"] > cfg.min_mean_pnl_lower_bound
                    and (not cfg.require_market_brier_edge
                         or s.get(f"{mode}_brier_lcb", float("-inf")) > 0))

        def evidence_reason(mode: str) -> str:
            return (f"{s[f'{mode}_n']} independent families; "
                    f"{s.get(f'{mode}_days', 0)}d/"
                    f"{s.get(f'{mode}_regimes', 1)} regimes; "
                    f"P&L LCB ${s[f'{mode}_lcb']:+.3f}; "
                    f"Brier-edge LCB "
                    f"{s.get(f'{mode}_brier_lcb', float('-inf')):+.4f}")

        if s["live_n"] >= min_evidence:
            if qualifies("live"):
                return _LIVE_FULL
            return CellDecision(True, 1.0, "demoted",
                                f"live evidence insufficient ({evidence_reason('live')})")
        if s["paper_n"] >= min_evidence:
            if qualifies("paper"):
                return CellDecision(False, cfg.probation_multiplier, "probation",
                                    f"graduated from paper ({evidence_reason('paper')})")
            return CellDecision(True, 1.0, "paper_negative",
                                f"paper evidence insufficient ({evidence_reason('paper')})")
        cap = cfg.max_unproven_positions
        if cap > 0 and await self._paper_breadth() >= cap:
            return CellDecision(True, 0.0, "unproven_capped",
                                f"unproven spray cap hit (>= {cap} open paper positions) — "
                                "concentrating; skip new unproven entry")
        return CellDecision(True, 1.0, "unproven",
                            f"insufficient record (live {s['live_n']}, paper {s['paper_n']} "
                            f"< {min_evidence} independent families in {cfg.window_days}d)")

    # ------------------------------------------------------------------
    # Reporting (CLI)
    # ------------------------------------------------------------------

    async def report(self) -> list[dict]:
        """Every cell with ledger history in the window + its decision."""
        cfg = self._settings.graduation
        rows = await self._db.fetchall(
            """SELECT strategy_source AS strategy, category,
                 SUM(CASE WHEN is_paper = 0 THEN 1 ELSE 0 END) AS live_n,
                 -- pnl only; already net of fees (module docstring).
                 COALESCE(SUM(CASE WHEN is_paper = 0 THEN pnl ELSE 0 END), 0) AS live_pnl,
                 SUM(CASE WHEN is_paper = 1 THEN 1 ELSE 0 END) AS paper_n,
                 COALESCE(SUM(CASE WHEN is_paper = 1 THEN pnl ELSE 0 END), 0) AS paper_pnl
               FROM pnl_ledger
               WHERE realized_at >= datetime('now', ?)
               GROUP BY 1, 2 ORDER BY 1, 2""",
            (f"-{int(cfg.window_days)} days",),
        )
        out = []
        for r in rows or []:
            d = await self._compute(r["strategy"] or "llm", r["category"] or "")
            if (r["strategy"] or "llm") in set(cfg.exempt_strategies):
                d = _EXEMPT
            out.append({
                "strategy": r["strategy"] or "(none)",
                "category": r["category"] or "(none)",
                "live_n": int(r["live_n"] or 0),
                "live_pnl": float(r["live_pnl"] or 0.0),
                "paper_n": int(r["paper_n"] or 0),
                "paper_pnl": float(r["paper_pnl"] or 0.0),
                "status": d.status,
                "force_paper": d.force_paper,
                "multiplier": d.size_multiplier,
                "reason": d.reason,
            })
        return out
