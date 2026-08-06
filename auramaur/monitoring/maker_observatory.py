"""Shadow-only market-maker microstructure observations and fill markouts."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone

from auramaur.exchange.models import OrderBook

FEATURE_SCHEMA = "maker-observatory-v2"
CREDIBLE_FILL_EVIDENCE = frozenset({"venue_fill", "book_cross", "trade_through"})


# The resolver's two hot statements, named so tests can EXPLAIN QUERY PLAN the
# exact SQL that ships. Both compare a timestamp column against a bound
# computed in Python.
#
# The MAX(observed_at) term is a liveness guard, not a filter on the evidence.
# A mark can only ever come from an observation, so a fill whose market has not
# been observed until at least its EARLIEST horizon cannot be marked at all
# yet — and because the batch below is oldest-first, fills stranded by a market
# that left the maker's five would otherwise hold the head of the queue until
# retention pruned them 45 days later, starving every newer mark behind them.
# Excluding them changes nothing about what gets concluded: the moment that
# market is observed again the fill re-enters the scan and takes its (late,
# is_valid=0) mark exactly as before. It is a per-candidate-row filter, not the
# driving predicate — `filled_at<=?` still seeks the index — and the subquery
# is an index-max lookup on (market_id, observed_at).
#
# The predicate this replaced,
# ``unixepoch(?) - unixepoch(f.filled_at) >= ?``, wrapped the column in a
# function: SQLite could seek to the market but then had to walk EVERY retained
# fill in it and probe the markouts index once per row
# (``SEARCH f USING INDEX idx_maker_fills_market_time (market_id=?)`` +
# ``CORRELATED SCALAR SUBQUERY``), which is where the 4.6 s went.
DUE_FILL_SCAN_SQL = """SELECT f.id,f.market_id,f.side,f.fill_price,f.filled_at
     FROM maker_observatory_fills f
    WHERE f.marks_pending=1 AND f.filled_at<=?
      AND unixepoch((SELECT MAX(o.observed_at) FROM maker_observations o
                      WHERE o.market_id=f.market_id))
          - unixepoch(f.filled_at) >= ?
    ORDER BY f.filled_at LIMIT ?"""
MARK_BOOK_SQL = """SELECT observed_at,midpoint FROM maker_observations
    WHERE market_id=? AND observed_at>=?
    ORDER BY observed_at LIMIT 1"""


def _stamp(moment: datetime) -> str:
    """The one timestamp spelling this module writes and compares.

    Every ``observed_at``/``filled_at`` is written through here, so the columns
    are fixed-width UTC ISO-8601 and lexicographic order IS chronological
    order. That is what lets the resolver's range predicates use an index
    instead of wrapping the column in ``unixepoch()``.
    """
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class MakerFeatures:
    best_bid: float
    best_ask: float
    midpoint: float
    microprice: float
    spread: float
    bid_depth: float
    ask_depth: float
    imbalance: float


def compute_maker_features(book: OrderBook, *, levels: int = 5) -> MakerFeatures:
    """Compute ordering-agnostic top-of-book microstructure features."""
    bids = sorted(book.bids, key=lambda level: level.price, reverse=True)[:levels]
    asks = sorted(book.asks, key=lambda level: level.price)[:levels]
    if not bids or not asks:
        raise ValueError("maker observatory requires a two-sided book")
    best_bid, best_ask = bids[0].price, asks[0].price
    if best_ask <= best_bid:
        raise ValueError("maker observatory requires a positive spread")
    bid_depth = sum(level.size for level in bids)
    ask_depth = sum(level.size for level in asks)
    total_depth = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / total_depth if total_depth else 0.0
    top_total = bids[0].size + asks[0].size
    midpoint = (best_bid + best_ask) / 2.0
    microprice = (
        (best_ask * bids[0].size + best_bid * asks[0].size) / top_total
        if top_total else midpoint
    )
    return MakerFeatures(
        best_bid=best_bid, best_ask=best_ask, midpoint=midpoint,
        microprice=microprice, spread=best_ask - best_bid,
        bid_depth=bid_depth, ask_depth=ask_depth, imbalance=imbalance,
    )


class MakerObservatory:
    """Persist measurements only; this class has no order-placement capability.

    The two halves run on different paths on purpose. ``observe()`` is on the
    market maker's quoting path and stays cheap and constant-cost:
    ``compute_maker_features`` plus a bounded, indexed volatility window plus
    one INSERT. ``resolve_markouts()`` — deciding whether a fill from four
    minutes ago moved against us — is on its own timer and touches nothing the
    current quote depends on. See docs/maker-observatory.md.
    """

    def __init__(self, db, *, flow_tracker=None, horizons=(30, 60, 300),
                 retention_days: int = 45, max_mark_lateness_seconds: int = 45,
                 holdout_days: int = 7, resolve_batch_fills: int = 500) -> None:
        self.db = db
        self.flow_tracker = flow_tracker
        self.horizons = tuple(sorted(set(int(value) for value in horizons)))
        self.retention_days = retention_days
        self.max_mark_lateness_seconds = max_mark_lateness_seconds
        self.holdout_days = holdout_days
        # Bounds one resolver pass. A backlog (long outage, clock jump) must not
        # let a single pass hold the shared Database serializer for minutes;
        # the remainder stays pending and the next pass takes it.
        self.resolve_batch_fills = max(1, int(resolve_batch_fills))
        frozen = json.dumps({
            "feature_schema": FEATURE_SCHEMA,
            "horizons": self.horizons,
            "max_mark_lateness_seconds": max_mark_lateness_seconds,
        }, sort_keys=True, separators=(",", ":"))
        self.config_json = frozen
        self.strategy_version = hashlib.sha256(frozen.encode()).hexdigest()
        self._registered = False

    async def _register(self) -> None:
        # Registration is idempotent and constant for this instance's frozen
        # config, but observe() runs on the market maker's quoting path, where
        # every statement takes the process-wide Database serializer. Re-issuing
        # 1 + len(horizons) INSERT OR IGNOREs on every refresh spends four
        # shared-lock acquisitions per observation to re-learn a fact that
        # cannot change. Do it once per process.
        if self._registered:
            return
        await self.db.execute(
            """INSERT OR IGNORE INTO strategy_experiments
                   (strategy_version,strategy_source,config_json,holdout_starts_at)
               VALUES (?,'maker_observatory',?,datetime('now', ?))""",
            (self.strategy_version, self.config_json, f"+{self.holdout_days} days"),
        )
        for horizon in self.horizons:
            await self.db.execute(
                """INSERT OR IGNORE INTO maker_observatory_horizons
                       (strategy_version,horizon_seconds) VALUES (?,?)""",
                (self.strategy_version, horizon),
            )
        self._registered = True

    async def observe(self, market, book: OrderBook, *, quote=None, active_quote=None,
                      observed_at: datetime | None = None) -> int:
        """Record one microstructure sample. Runs on the quoting path.

        Everything here is O(1) in retained history. Markout resolution is
        deliberately absent: it belongs to ``resolve_markouts()``, because an
        instrument that added seconds to the quote path would cause the adverse
        selection it exists to measure.
        """
        now = (observed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        await self._register()
        features = compute_maker_features(book)
        # NULL, not 0.0, when no feed ever reached this market. See
        # OrderFlowTracker.signed_flow: the two are different facts and the
        # column has to be able to say which.
        signed_flow = None
        if self.flow_tracker is not None:
            signed_flow = self.flow_tracker.signed_flow(
                (market.id, getattr(market, "condition_id", "") or "",
                 market.clob_token_yes), now=now)
        quote_age = (
            max(0.0, (now - active_quote.placed_at).total_seconds())
            if active_quote is not None else None
        )
        quote_changed = None
        if quote is not None and active_quote is not None:
            quote_changed = int(
                quote.bid_price != active_quote.bid_price
                or quote.ask_price != active_quote.ask_price
                or quote.size != active_quote.size
            )
        short_vol = await self._realized_vol(market.id, features.midpoint, now, 300)
        long_vol = await self._realized_vol(market.id, features.midpoint, now, 1800)
        cursor = await self.db.execute(
            """INSERT INTO maker_observations
                   (market_id,condition_id,token_id,strategy_version,is_holdout,
                    observed_at,best_bid,best_ask,midpoint,
                    microprice,spread,bid_depth,ask_depth,depth_imbalance,
                    signed_flow,short_vol,long_vol,quote_bid,quote_ask,
                    quote_changed,quote_age_seconds,quote_active)
               VALUES (?,?,?,?,CASE WHEN datetime(?) >=
                    (SELECT holdout_starts_at FROM strategy_experiments
                     WHERE strategy_version=?) THEN 1 ELSE 0 END,
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (market.id, getattr(market, "condition_id", "") or "",
             market.clob_token_yes, self.strategy_version,
             _stamp(now), self.strategy_version, _stamp(now),
             features.best_bid, features.best_ask, features.midpoint,
             features.microprice, features.spread, features.bid_depth,
             features.ask_depth, features.imbalance, signed_flow, short_vol, long_vol,
             getattr(quote, "bid_price", None), getattr(quote, "ask_price", None),
             quote_changed, quote_age, int(active_quote is not None)),
        )
        return int(cursor.lastrowid)

    async def _realized_vol(self, market_id: str, midpoint: float,
                            now: datetime, window_seconds: int) -> float:
        cutoff = _stamp(datetime.fromtimestamp(
            now.timestamp() - window_seconds, tz=timezone.utc))
        rows = await self.db.fetchall(
            """SELECT midpoint FROM maker_observations
               WHERE market_id=? AND observed_at>=? AND observed_at<?
               ORDER BY observed_at""",
            (market_id, cutoff, _stamp(now)),
        )
        prices = [float(row["midpoint"]) for row in rows] + [midpoint]
        changes = [prices[index] - prices[index - 1]
                   for index in range(1, len(prices))]
        return statistics.pstdev(changes) if len(changes) >= 2 else 0.0

    async def record_observed_fill(self, *, observation_id: int | None, order_id: str,
                                   market_id: str, side: str, price: float, size: float,
                                   is_paper: bool, fill_evidence: str,
                                   filled_at: datetime | None = None) -> None:
        """Note that a fill happened. NOT ``record_fill``, deliberately.

        ``record_fill`` is in exposure_registry.SENSITIVE_METHODS, and that
        registry is scanned by attribute name — so calling this method
        ``record_fill`` forced a pure measurement write to be declared as a
        ``prediction_quoting`` exposure-mutation callsite, which it is not.
        This class books no position, moves no cash and touches no venue.
        """
        if not order_id or price <= 0 or size <= 0:
            return
        stamp = (filled_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        await self.db.execute(
            """INSERT OR IGNORE INTO maker_observatory_fills
                   (order_id,observation_id,market_id,side,fill_price,fill_size,
                    is_paper,fill_evidence,filled_at) VALUES (?,?,?,?,?,?,?,?,?)""",
            (order_id, observation_id, market_id, side, price, size, int(is_paper),
             fill_evidence, _stamp(stamp)),
        )

    async def resolve_markouts(self, *, now: datetime | None = None) -> int:
        """Mark out every fill whose horizon has come due. Runs OFF the quote path.

        This used to run inline in ``observe()``, inside the market maker's
        per-market ``op_timeout`` window, holding the process-wide
        ``Database`` serializer that ~30 pillar tasks share. It was 93% of
        ``observe()``'s cost and it grew with retained history (4.6 s per
        5-market cycle at 45-day retention), so the instrument would have
        slowed the quotes it measures into exactly the adverse selection it
        exists to detect.

        Two properties make the move safe:

        * **The mark is a pure function of the record, not of when this runs.**
          A fill's mark is taken from the FIRST observation at or after
          ``filled_at + horizon`` — which is precisely the book the inline scan
          used, because the inline scan ran on the observation that first found
          the fill due. Horizons, lateness and the validity rule are unchanged,
          and a mark taken from a late book is still stored with
          ``is_valid=0``. Running once an hour and running every cycle
          therefore produce identical rows; only the wall-clock moment of the
          INSERT differs.
        * **An interrupted pass leaves due marks unresolved, never fabricated.**
          ``marks_pending`` is only cleared in the same transaction that writes
          a fill's last mark, and every insert is ``INSERT OR IGNORE`` against
          the ``(fill_id,horizon_seconds)`` primary key, so a re-run resumes
          rather than double-counting. A fill whose market has not been
          observed since its horizon simply stays pending.

        ``marks_pending`` is also what keeps this cheap: the partial index
        ``idx_maker_fills_pending`` contains only unresolved fills, so the scan
        is proportional to the backlog, not to the 259k retained fills per
        market that the old ``unixepoch()`` predicate had to re-read.
        """
        instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        due_bound = _stamp(datetime.fromtimestamp(
            instant.timestamp() - min(self.horizons), tz=timezone.utc))
        pending = await self.db.fetchall(
            DUE_FILL_SCAN_SQL,
            (due_bound, min(self.horizons), self.resolve_batch_fills))
        resolved = 0
        for fill in pending:
            resolved += await self._resolve_one(fill, instant)
        return resolved

    async def _resolve_one(self, fill, instant: datetime) -> int:
        filled_at = datetime.fromisoformat(str(fill["filled_at"]))
        marked = {int(row["horizon_seconds"]) for row in await self.db.fetchall(
            "SELECT horizon_seconds FROM maker_observatory_markouts WHERE fill_id=?",
            (fill["id"],))}
        # Reads first, writes second: the transaction below must not span the
        # per-horizon lookups, or the resolver would hold the write lock for
        # the whole fill instead of for its inserts.
        rows: list[tuple] = []
        for horizon in self.horizons:
            if horizon in marked:
                continue
            target_at = datetime.fromtimestamp(
                filled_at.timestamp() + horizon, tz=timezone.utc)
            if target_at > instant:
                continue
            mark = await self.db.fetchone(
                MARK_BOOK_SQL, (fill["market_id"], _stamp(target_at)))
            if mark is None:
                continue  # no book at or after the horizon yet: stay pending
            marked_at = datetime.fromisoformat(str(mark["observed_at"]))
            midpoint = float(mark["midpoint"])
            lateness = max(0.0, (marked_at - target_at).total_seconds())
            # Positive means price moved in the maker fill's favour.
            markout = (midpoint - fill["fill_price"] if fill["side"] == "bid"
                       else fill["fill_price"] - midpoint)
            rows.append((fill["id"], horizon, _stamp(target_at), midpoint, markout,
                         _stamp(marked_at), lateness,
                         int(lateness <= self.max_mark_lateness_seconds)))
            marked.add(horizon)
        if not rows:
            return 0
        complete = marked.issuperset(self.horizons)
        async with self.db.transaction(owner="maker_observatory.resolve_markouts"):
            for row in rows:
                await self.db.execute(
                    """INSERT OR IGNORE INTO maker_observatory_markouts
                           (fill_id,horizon_seconds,target_at,midpoint,markout,
                            marked_at,lateness_seconds,is_valid)
                       VALUES (?,?,?,?,?,?,?,?)""", row)
            if complete:
                await self.db.execute(
                    "UPDATE maker_observatory_fills SET marks_pending=0 WHERE id=?",
                    (fill["id"],))
        return len(rows)

    async def prune(self) -> None:
        """Bound raw storage while retaining rows needed by surviving fills."""
        modifier = f"-{self.retention_days} days"
        await self.db.execute(
            "DELETE FROM maker_observatory_markouts WHERE fill_id IN "
            "(SELECT id FROM maker_observatory_fills "
            "WHERE datetime(filled_at) < datetime('now', ?))",
            (modifier,),
        )
        await self.db.execute(
            "DELETE FROM maker_observatory_fills "
            "WHERE datetime(filled_at) < datetime('now', ?)",
            (modifier,),
        )
        await self.db.execute(
            "DELETE FROM maker_observations "
            "WHERE datetime(observed_at) < datetime('now', ?)",
            (modifier,),
        )


async def maker_observatory_summary(db, *, days: int = 21) -> list[dict]:
    """Return honest markout evidence, including invalid/missing mark diagnostics."""
    rows = await db.fetchall(
        """SELECT h.horizon_seconds,f.is_paper,f.market_id,f.fill_size,
                  f.fill_evidence,m.markout,m.is_valid,m.lateness_seconds,
                  o.is_holdout,
                  CASE WHEN unixepoch('now')-unixepoch(f.filled_at)
                         >= h.horizon_seconds THEN 1 ELSE 0 END AS is_due
           FROM maker_observatory_fills f
           JOIN maker_observations o ON o.id=f.observation_id
           JOIN maker_observatory_horizons h
             ON h.strategy_version=o.strategy_version
           LEFT JOIN maker_observatory_markouts m
             ON m.fill_id=f.id AND m.horizon_seconds=h.horizon_seconds
           WHERE datetime(f.filled_at) >= datetime('now', ?)
           ORDER BY f.is_paper,h.horizon_seconds,f.market_id""",
        (f"-{days} days",),
    )
    grouped: dict[tuple[int, int], list] = {}
    for row in rows:
        grouped.setdefault((row["horizon_seconds"], row["is_paper"]), []).append(row)
    output = []
    for (horizon, is_paper), values in grouped.items():
        due = [row for row in values if row["is_due"] == 1]
        valid = [row for row in due if row["is_valid"] == 1]
        credible = [row for row in valid
                    if row["fill_evidence"] in CREDIBLE_FILL_EVIDENCE
                    and row["is_holdout"] == 1]
        cluster_values = [(str(row["market_id"]), float(row["markout"]))
                          for row in credible]
        marks = [value for _, value in cluster_values]
        low, high = _clustered_mean_ci(cluster_values)
        weighted_denominator = sum(float(row["fill_size"]) for row in credible)
        output.append({
            "horizon_seconds": horizon, "is_paper": is_paper,
            "fills": len(values), "due_marks": len(due),
            "pending_marks": len(values) - len(due),
            "valid_marks": len(valid),
            "credible_holdout_marks": len(credible),
            "markets": len({key for key, _ in cluster_values}),
            "mean_markout": statistics.fmean(marks) if marks else None,
            "size_weighted_markout": (
                sum(float(row["markout"]) * float(row["fill_size"])
                    for row in credible) / weighted_denominator
                if weighted_denominator else None),
            "ci_low": low, "ci_high": high,
            "toxic_rate": (sum(value < 0 for value in marks) / len(marks)
                           if marks else None),
            "completeness": len(valid) / len(due) if due else 0.0,
            "late_marks": sum(row["is_valid"] == 0 and row["markout"] is not None
                              for row in values),
        })
    return output


async def maker_observatory_feature_report(db, *, days: int = 21) -> list[dict]:
    """Score frozen warmup thresholds only on credible holdout fill markouts.

    Every row carries its own ``covered_n``/``coverage``: the share of the
    horizon's marks for which this feature was actually recorded. A feature
    can be NULL for reasons that have nothing to do with the market —
    ``signed_flow`` is NULL whenever no trade feed reached the market — and a
    "no effect" verdict computed over a feature that was never measured is not
    a negative result, it is an absent one. NULLs are excluded from the warmup
    median and from both holdout buckets; the coverage number is what stops a
    future reader from reading that exclusion as balance.
    """
    rows = await db.fetchall(
        """SELECT m.horizon_seconds,f.market_id,f.fill_evidence,f.filled_at,
                  o.is_holdout,o.microprice-o.midpoint AS microprice_skew,
                  o.depth_imbalance,o.signed_flow,o.short_vol,o.long_vol,
                  o.quote_age_seconds,m.markout
             FROM maker_observatory_fills f
             JOIN maker_observations o ON o.id=f.observation_id
             JOIN maker_observatory_markouts m ON m.fill_id=f.id
            WHERE datetime(f.filled_at) >= datetime('now', ?)
              AND f.is_paper=0 AND m.is_valid=1
            ORDER BY m.horizon_seconds,f.market_id""",
        (f"-{days} days",),
    )
    features = ("microprice_skew", "depth_imbalance", "signed_flow",
                "short_vol", "long_vol", "quote_age_seconds")
    output = []
    horizons = sorted({int(row["horizon_seconds"]) for row in rows})
    for horizon in horizons:
        horizon_rows = [row for row in rows if row["horizon_seconds"] == horizon]
        for feature in features:
            warmup = [float(row[feature]) for row in horizon_rows
                      if row["is_holdout"] == 0 and row[feature] is not None]
            holdout = [row for row in horizon_rows
                       if row["is_holdout"] == 1
                       and row["fill_evidence"] in CREDIBLE_FILL_EVIDENCE
                       and row[feature] is not None]
            threshold = statistics.median(warmup) if warmup else None
            high = [] if threshold is None else [
                (str(row["market_id"]), float(row["markout"]))
                for row in holdout if float(row[feature]) >= threshold]
            low = [] if threshold is None else [
                (str(row["market_id"]), float(row["markout"]))
                for row in holdout if float(row[feature]) < threshold]
            effect = (statistics.fmean(value for _, value in high)
                      - statistics.fmean(value for _, value in low)
                      if high and low else None)
            covered = sum(row[feature] is not None for row in horizon_rows)
            output.append({
                "horizon_seconds": horizon, "feature": feature,
                "warmup_n": len(warmup), "holdout_n": len(holdout),
                "threshold": threshold, "high_n": len(high), "low_n": len(low),
                "effect": effect,
                "markets": len({key for key, _ in high + low}),
                "marks_n": len(horizon_rows), "covered_n": covered,
                "coverage": covered / len(horizon_rows) if horizon_rows else None,
            })
    return output


async def maker_quote_coverage(db, *, days: int = 21) -> dict:
    """Report cadence-weighted quote presence without pretending it is uptime.

    Also reports trade-feed coverage. ``COUNT(signed_flow)`` skips NULLs by
    definition, so ``flow_samples`` is literally "how many observations had
    flow data at all" — the number that separates a market whose flow was
    balanced from one no feed ever reached.
    """
    row = await db.fetchone(
        """SELECT COUNT(*) AS samples,
                  SUM(quote_active=1) AS active,
                  SUM(quote_changed=1) AS changed,
                  SUM(is_holdout=1) AS holdout,
                  COUNT(signed_flow) AS flow
             FROM maker_observations
            WHERE datetime(observed_at) >= datetime('now', ?)""",
        (f"-{days} days",),
    )
    samples = int(row["samples"] or 0)
    return {
        "samples": samples,
        "active_samples": int(row["active"] or 0),
        "changed_samples": int(row["changed"] or 0),
        "holdout_samples": int(row["holdout"] or 0),
        "sampled_quote_coverage": (
            int(row["active"] or 0) / samples if samples else None),
        "flow_samples": int(row["flow"] or 0),
        "flow_coverage": int(row["flow"] or 0) / samples if samples else None,
    }


def maker_promotion_blockers(rows: list[dict], *, min_fills: int = 100,
                             min_markets: int = 5,
                             min_completeness: float = .95) -> list[str]:
    """Return explicit reasons the live observatory evidence cannot be promoted."""
    blockers = []
    live = [row for row in rows if not row["is_paper"]]
    if not live:
        return ["no live fills"]
    for row in live:
        label = f"{row['horizon_seconds']}s"
        if row["credible_holdout_marks"] < min_fills:
            blockers.append(
                f"{label}: {row['credible_holdout_marks']}/{min_fills} "
                "credible holdout marks")
        if row["markets"] < min_markets:
            blockers.append(f"{label}: {row['markets']}/{min_markets} markets")
        if row["completeness"] < min_completeness:
            blockers.append(
                f"{label}: {row['completeness']:.1%}/{min_completeness:.1%} "
                "valid-mark completeness")
        if row["ci_low"] is None or row["ci_low"] <= 0:
            blockers.append(f"{label}: mean markout lower CI is not positive")
    return blockers


def _clustered_mean_ci(values: list[tuple[str, float]], *, samples: int = 2000,
                       seed: int = 20260805) -> tuple[float | None, float | None]:
    clusters: dict[str, list[float]] = {}
    for market_id, value in values:
        clusters.setdefault(market_id, []).append(value)
    keys = sorted(clusters)
    if len(keys) < 2:
        return None, None
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        chosen = [rng.choice(keys) for _ in keys]
        draw = [value for key in chosen for value in clusters[key]]
        draws.append(statistics.fmean(draw))
    draws.sort()
    return draws[int(samples * .025)], draws[int(samples * .975)]
