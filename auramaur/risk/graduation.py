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
"""

from __future__ import annotations

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
        rows = await self._db.fetchall(
            """SELECT market_id, is_paper, SUM(pnl) AS pnl
               FROM pnl_ledger
               WHERE strategy_source = ? AND category = ?
                 AND realized_at >= datetime('now', ?)
               GROUP BY market_id, is_paper""",
            (strategy, category, f"-{int(self._settings.graduation.window_days)} days"),
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
        s = await self._cell_stats(strategy, category)

        if s["live_n"] >= cfg.min_markets:
            if s["live_lcb"] > cfg.min_mean_pnl_lower_bound:
                return _LIVE_FULL
            return CellDecision(
                True, 1.0, "demoted",
                f"live evidence insufficient (mean-P&L LCB ${s['live_lcb']:+.3f}; "
                f"${s['live_pnl']:+.2f} over {s['live_n']} independent markets)")
        if s["paper_n"] >= cfg.min_markets:
            if s["paper_lcb"] > cfg.min_mean_pnl_lower_bound:
                return CellDecision(
                    False, cfg.probation_multiplier, "probation",
                    f"graduated from paper (mean-P&L LCB ${s['paper_lcb']:+.3f}; "
                    f"${s['paper_pnl']:+.2f} over {s['paper_n']} independent markets)")
            return CellDecision(
                True, 1.0, "paper_negative",
                f"paper evidence insufficient (mean-P&L LCB ${s['paper_lcb']:+.3f}; "
                f"${s['paper_pnl']:+.2f} over {s['paper_n']} independent markets)")
        # Unproven (still exploring). Restrict the spray: if the open paper book is
        # already at the breadth cap, skip NEW unproven entries (size x0) so
        # exploration concentrates instead of spraying. Restriction only — proven/
        # probation/exempt cells and exits are never affected.
        cap = cfg.max_unproven_positions
        if cap > 0 and await self._paper_breadth() >= cap:
            return CellDecision(
                True, 0.0, "unproven_capped",
                f"unproven spray cap hit (>= {cap} open paper positions) — "
                f"concentrating; skip new unproven entry")
        return CellDecision(
            True, 1.0, "unproven",
            f"insufficient record (live {s['live_n']}, paper {s['paper_n']} "
            f"< {cfg.min_markets} markets in {cfg.window_days}d)")

    # ------------------------------------------------------------------
    # Reporting (CLI)
    # ------------------------------------------------------------------

    async def report(self) -> list[dict]:
        """Every cell with ledger history in the window + its decision."""
        cfg = self._settings.graduation
        rows = await self._db.fetchall(
            """SELECT strategy_source AS strategy, category,
                 SUM(CASE WHEN is_paper = 0 THEN 1 ELSE 0 END) AS live_n,
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
