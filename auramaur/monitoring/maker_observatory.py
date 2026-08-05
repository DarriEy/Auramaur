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
    """Persist measurements only; this class has no order-placement capability."""

    def __init__(self, db, *, flow_tracker=None, horizons=(30, 60, 300),
                 retention_days: int = 45, max_mark_lateness_seconds: int = 45,
                 holdout_days: int = 7) -> None:
        self.db = db
        self.flow_tracker = flow_tracker
        self.horizons = tuple(sorted(set(int(value) for value in horizons)))
        self.retention_days = retention_days
        self.max_mark_lateness_seconds = max_mark_lateness_seconds
        self.holdout_days = holdout_days
        frozen = json.dumps({
            "feature_schema": FEATURE_SCHEMA,
            "horizons": self.horizons,
            "max_mark_lateness_seconds": max_mark_lateness_seconds,
        }, sort_keys=True, separators=(",", ":"))
        self.config_json = frozen
        self.strategy_version = hashlib.sha256(frozen.encode()).hexdigest()

    async def _register(self) -> None:
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

    async def observe(self, market, book: OrderBook, *, quote=None, active_quote=None,
                      observed_at: datetime | None = None) -> int:
        now = (observed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        await self._register()
        features = compute_maker_features(book)
        signed_flow = 0.0
        if self.flow_tracker is not None:
            signed_flow = self.flow_tracker.signed_flow(
                (market.id, market.condition_id, market.clob_token_yes), now=now)
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
             now.isoformat(timespec="microseconds"), self.strategy_version,
             now.isoformat(timespec="microseconds"),
             features.best_bid, features.best_ask, features.midpoint,
             features.microprice, features.spread, features.bid_depth,
             features.ask_depth, features.imbalance, signed_flow, short_vol, long_vol,
             getattr(quote, "bid_price", None), getattr(quote, "ask_price", None),
             quote_changed, quote_age, int(active_quote is not None)),
        )
        await self._mark_due(market.id, features.midpoint, now)
        return int(cursor.lastrowid)

    async def _realized_vol(self, market_id: str, midpoint: float,
                            now: datetime, window_seconds: int) -> float:
        cutoff = datetime.fromtimestamp(
            now.timestamp() - window_seconds, tz=timezone.utc
        ).isoformat(timespec="microseconds")
        rows = await self.db.fetchall(
            """SELECT midpoint FROM maker_observations
               WHERE market_id=? AND observed_at>=? AND observed_at<?
               ORDER BY observed_at""",
            (market_id, cutoff, now.isoformat(timespec="microseconds")),
        )
        prices = [float(row["midpoint"]) for row in rows] + [midpoint]
        changes = [prices[index] - prices[index - 1]
                   for index in range(1, len(prices))]
        return statistics.pstdev(changes) if len(changes) >= 2 else 0.0

    async def record_fill(self, *, observation_id: int | None, order_id: str,
                          market_id: str, side: str, price: float, size: float,
                          is_paper: bool, fill_evidence: str,
                          filled_at: datetime | None = None) -> None:
        if not order_id or price <= 0 or size <= 0:
            return
        stamp = (filled_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        await self.db.execute(
            """INSERT OR IGNORE INTO maker_observatory_fills
                   (order_id,observation_id,market_id,side,fill_price,fill_size,
                    is_paper,fill_evidence,filled_at) VALUES (?,?,?,?,?,?,?,?,?)""",
            (order_id, observation_id, market_id, side, price, size, int(is_paper),
             fill_evidence, stamp.isoformat(timespec="microseconds")),
        )

    async def _mark_due(self, market_id: str, midpoint: float, now: datetime) -> None:
        for horizon in self.horizons:
            rows = await self.db.fetchall(
                """SELECT f.id,f.side,f.fill_price,f.filled_at
                   FROM maker_observatory_fills f
                   WHERE f.market_id=?
                     AND unixepoch(?) - unixepoch(f.filled_at) >= ?
                     AND NOT EXISTS (SELECT 1 FROM maker_observatory_markouts m
                         WHERE m.fill_id=f.id AND m.horizon_seconds=?)
                   """,
                (market_id, now.isoformat(timespec="microseconds"), horizon, horizon),
            )
            for row in rows:
                filled_at = datetime.fromisoformat(str(row["filled_at"]))
                target_at = datetime.fromtimestamp(
                    filled_at.timestamp() + horizon, tz=timezone.utc)
                lateness = max(0.0, (now - target_at).total_seconds())
                # Positive means price moved in the maker fill's favour.
                markout = (midpoint - row["fill_price"] if row["side"] == "bid"
                           else row["fill_price"] - midpoint)
                await self.db.execute(
                    """INSERT OR IGNORE INTO maker_observatory_markouts
                           (fill_id,horizon_seconds,target_at,midpoint,markout,
                            marked_at,lateness_seconds,is_valid)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (row["id"], horizon, target_at.isoformat(timespec="microseconds"),
                     midpoint, markout, now.isoformat(timespec="microseconds"),
                     lateness, int(lateness <= self.max_mark_lateness_seconds)),
                )

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
    """Score frozen warmup thresholds only on credible holdout fill markouts."""
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
            output.append({
                "horizon_seconds": horizon, "feature": feature,
                "warmup_n": len(warmup), "holdout_n": len(holdout),
                "threshold": threshold, "high_n": len(high), "low_n": len(low),
                "effect": effect,
                "markets": len({key for key, _ in high + low}),
            })
    return output


async def maker_quote_coverage(db, *, days: int = 21) -> dict:
    """Report cadence-weighted quote presence without pretending it is uptime."""
    row = await db.fetchone(
        """SELECT COUNT(*) AS samples,
                  SUM(quote_active=1) AS active,
                  SUM(quote_changed=1) AS changed,
                  SUM(is_holdout=1) AS holdout
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
