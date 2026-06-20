"""Favorite-longshot bias harvest — forecast-free base-load strategy.

Backtest basis (scripts/research/favorite_longshot_backtest.py, 2026-06-09,
414 resolved Polymarket markets with ex-ante signal prices): buying the
favored side when it trades in the 0.60-0.97 band and holding to resolution
returned +3.3% per $1 at 2c slippage (88% win rate); the deep 0.90-0.97 band
won 99.3% of 151 markets vs a 95.5% breakeven. The mirror control (buying
the longshot side at the same moments) lost -87% per $1 in the deep band —
the classic favorite-longshot bias, measured on the bot's own universe. The
edge survives removing the Iran cluster and sports/junk (improves to +4.9%)
but DIES at 4.4c slippage, so entries are passive limit orders at the
observed price, never crossed.

Design constraints:
  * PAPER-FORCED by default (``bias_harvest.paper: true``): orders carry
    dry_run=True regardless of global live mode until the paper ledger
    proves the edge (read it with ``auramaur pnl --paper``).
  * One entry per market, ever — the backtest's first-band-crossing rule.
  * Never enters a market the bot already holds (any strategy, any mode):
    avoids doubled exposure and settlement ambiguity.
  * Rides the existing rails: signals + trades rows (attribution), full
    RiskManager gate (all 15 checks — never bypassed), calibration
    prediction (resolution scoring), PnLTracker fills (pnl_ledger), and a
    portfolio row so the resolution tracker settles it.
  * claude_prob is the band's empirically measured resolution rate proxy
    (favored price + ``edge_uplift``, default 0.04 — under the measured
    band gaps of +4..+6pts). With uplift < 0.05 the divergence-band risk
    filter (adverse band starts at 0.05) does not bite; raise it past
    0.05 and entries will need HIGH confidence, which this strategy
    deliberately does not claim.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from auramaur.strategy.classifier import ensure_category
from auramaur.exchange.models import (
    Confidence,
    Fill,
    Market,
    OrderSide,
    Signal,
)

log = structlog.get_logger()


class BiasHarvestPillar:
    """Periodic favored-side scanner for Polymarket."""

    def __init__(self, db, settings, discovery, exchange, risk_manager,
                 pnl_tracker, calibration) -> None:
        self._db = db
        self._settings = settings
        self._discovery = discovery
        self._exchange = exchange
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._calibration = calibration

    # ------------------------------------------------------------------
    # Scan cycle
    # ------------------------------------------------------------------

    async def run_once(self) -> int:
        """One scan; returns the number of entries placed."""
        cfg = self._settings.bias_harvest
        if not cfg.enabled:
            return 0

        open_count = await self._open_position_count()
        if open_count >= cfg.max_open:
            log.debug("bias_harvest.book_full", open=open_count, cap=cfg.max_open)
            return 0

        markets = await self._discovery.get_markets(limit=cfg.scan_limit)
        entered = 0
        for market in markets:
            if entered >= cfg.max_entries_per_cycle:
                break
            if open_count + entered >= cfg.max_open:
                break
            try:
                if await self._try_enter(market):
                    entered += 1
            except Exception as e:
                log.error("bias_harvest.entry_error", market_id=market.id, error=str(e))
        if entered:
            log.info("bias_harvest.cycle_done", entered=entered)
        return entered

    # ------------------------------------------------------------------
    # Entry pipeline
    # ------------------------------------------------------------------

    def _band_check(self, market: Market) -> tuple[float, bool] | None:
        """Return (favored_price, favored_is_yes) if in band, else None."""
        cfg = self._settings.bias_harvest
        p_yes = market.outcome_yes_price
        if not (0.0 < p_yes < 1.0):
            return None
        p_fav = max(p_yes, 1.0 - p_yes)
        if not (cfg.band_lo <= p_fav < cfg.band_hi):
            return None
        return p_fav, p_yes >= 0.5

    def _eligible(self, market: Market) -> bool:
        cfg = self._settings.bias_harvest
        if not market.active:
            return False
        if (market.exchange or "polymarket") != "polymarket":
            return False  # backtest validated Polymarket only (0% fees)
        # Tail-filter: don't harvest a favorite whose resolution is actively
        # disputed — the pinned price can flip when the dispute clears, which is
        # exactly the fat-tail loss the paper track surfaced. Fails open (only a
        # confirmed DO_NOT_ACT is skipped; markets with no UMA data still enter).
        if cfg.skip_disputed and market.dispute_risk == "DO_NOT_ACT":
            log.debug("bias_harvest.skip_disputed", market_id=market.id)
            return False
        if market.liquidity < cfg.min_liquidity:
            return False
        # Classify before the block: discovery often hands us an empty/mislabeled
        # category, which slips through a raw `category in blocked` test and only
        # gets corrected after entry — that bypass let sports/politics_us markets
        # (both already blocked) into the paper book. Resolve the category the
        # same way persistence does, then reject it against the global block AND
        # the bias-harvest no-edge list (weather/sports/politics_us).
        cat = ensure_category(market.question, market.description, market.category)
        excluded = set(self._settings.risk.blocked_categories) | set(cfg.exclude_categories)
        if cat in excluded:
            log.debug("bias_harvest.skip_category", market_id=market.id, category=cat)
            return False
        if market.end_date is None:
            return False
        end = market.end_date
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        hours_left = (end - datetime.now(timezone.utc)).total_seconds() / 3600.0
        if hours_left < cfg.min_hours_to_resolution:
            return False
        if hours_left > cfg.max_days_to_resolution * 24.0:
            return False  # capital lock-up cap
        return True

    async def _already_entered_or_held(self, market_id: str) -> bool:
        row = await self._db.fetchone(
            "SELECT 1 FROM signals WHERE market_id = ? AND strategy_source = 'bias_harvest' LIMIT 1",
            (market_id,),
        )
        if row is not None:
            return True  # one shot per market, ever
        row = await self._db.fetchone(
            "SELECT 1 FROM portfolio WHERE market_id = ? LIMIT 1", (market_id,),
        )
        return row is not None  # held by any strategy/mode — stay out

    async def _open_position_count(self) -> int:
        row = await self._db.fetchone(
            """SELECT COUNT(*) AS n FROM portfolio p
               WHERE EXISTS (SELECT 1 FROM signals s
                             WHERE s.market_id = p.market_id
                               AND s.strategy_source = 'bias_harvest')""",
        )
        return int(row["n"]) if row else 0

    def _build_signal(self, market: Market, p_fav: float, fav_is_yes: bool) -> Signal:
        cfg = self._settings.bias_harvest
        uplift = cfg.edge_uplift
        # claude_prob is P(YES): shift toward the favored side by the
        # measured-bias uplift, clamped inside the book.
        if fav_is_yes:
            claude_prob = min(0.99, market.outcome_yes_price + uplift)
        else:
            claude_prob = max(0.01, market.outcome_yes_price - uplift)
        return Signal(
            market_id=market.id,
            market_question=market.question,
            claude_prob=claude_prob,
            claude_confidence=Confidence.MEDIUM,
            market_prob=market.outcome_yes_price,
            edge=uplift * 100.0,
            evidence_summary=(
                f"Bias harvest: favored side at {p_fav:.2f} in band "
                f"[{cfg.band_lo:.2f}, {cfg.band_hi:.2f}); measured "
                f"favorite-longshot bias, no forecast claimed."
            ),
            recommended_side=OrderSide.BUY if fav_is_yes else OrderSide.SELL,
            strategy_source="bias_harvest",
        )

    async def _try_enter(self, market: Market) -> bool:
        cfg = self._settings.bias_harvest
        band = self._band_check(market)
        if band is None or not self._eligible(market):
            return False
        if await self._already_entered_or_held(market.id):
            return False
        p_fav, fav_is_yes = band

        signal = self._build_signal(market, p_fav, fav_is_yes)
        await self._persist_signal(signal, market)

        # Full risk gate — all checks apply; never bypassed.
        decision = await self._risk.evaluate(signal, market)
        if not decision.approved or decision.position_size <= 0:
            log.info("bias_harvest.risk_rejected", market_id=market.id,
                     reason=decision.reason)
            return False
        size = min(decision.position_size, cfg.stake_usd)

        # Passive limit at the observed price (prepare_order uses the seen
        # token price, tick-rounded — no improvement, no crossing). Paper-
        # forced entries pass is_live=False so the order carries dry_run=True
        # regardless of the global live gates. The graduation ladder
        # (decision.force_paper) can also paper-force, never un-paper.
        is_live_order = (self._settings.is_live and not cfg.paper
                         and not getattr(decision, "force_paper", False))
        order = self._exchange.prepare_order(signal, market, size, is_live_order)
        if order is None:
            log.warning("bias_harvest.order_build_failed", market_id=market.id)
            return False
        result = await self._exchange.place_order(order)
        if result.status not in ("filled", "paper", "partial", "pending"):
            log.warning("bias_harvest.order_rejected", market_id=market.id,
                        status=result.status, error=result.error_message)
            return False

        await self._record_entry(signal, market, order, result)
        log.info("bias_harvest.entered", market_id=market.id,
                 token=order.token.value, price=order.price,
                 size=order.size, paper=result.is_paper)
        return True

    # ------------------------------------------------------------------
    # Bookkeeping — same rails as the engine
    # ------------------------------------------------------------------

    async def _persist_signal(self, signal: Signal, market: Market) -> None:
        await self._db.execute(
            """INSERT OR IGNORE INTO markets (id, exchange, condition_id, question,
               description, category, active, outcome_yes_price, outcome_no_price,
               volume, liquidity, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, datetime('now'))""",
            (market.id, market.exchange or "polymarket", market.condition_id,
             market.question, (market.description or "")[:500],
             ensure_category(market.question, market.description, market.category),
             market.outcome_yes_price, market.outcome_no_price,
             market.volume, market.liquidity),
        )
        await self._db.execute(
            """INSERT INTO signals (market_id, claude_prob, claude_confidence,
               market_prob, edge, evidence_summary, action, strategy_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'bias_harvest')""",
            (signal.market_id, signal.claude_prob, signal.claude_confidence.value,
             signal.market_prob, signal.edge, signal.evidence_summary,
             signal.recommended_side.value),
        )
        await self._db.commit()

    async def _record_entry(self, signal: Signal, market: Market,
                            order, result) -> None:
        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price
        is_paper = bool(result.is_paper)

        # Fills -> cost_basis (+ pnl_ledger at realization)
        if result.status in ("filled", "paper", "partial") and fill_size > 0:
            await self._pnl.record_fill(Fill(
                order_id=result.order_id,
                market_id=order.market_id,
                token_id=order.token_id,
                side=order.side,
                token=order.token,
                size=fill_size,
                price=fill_price,
                is_paper=is_paper,
            ))

        # trades mirror (entry-strategy attribution + dedup belt)
        await self._db.execute(
            """INSERT INTO trades (market_id, timestamp, side, size, price,
               is_paper, order_id, status, strategy_source, exchange)
               VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, 'bias_harvest',
                       'polymarket')""",
            (order.market_id, order.side.value, fill_size, fill_price,
             1 if is_paper else 0, result.order_id,
             "filled" if result.status in ("filled", "paper") else result.status),
        )

        # Portfolio row so the resolution tracker settles the position. The
        # mode-scoped position sync won't maintain paper rows in a live bot,
        # so the pillar owns this write (additive upsert, same shape as
        # sync._merge_new_positions).
        await self._db.execute(
            """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
               current_price, unrealized_pnl, category, token, token_id,
               is_paper, updated_at)
               VALUES (?, 'polymarket', 'BUY', ?, ?, ?, 0, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                   size = excluded.size,
                   avg_price = excluded.avg_price,
                   current_price = excluded.current_price,
                   updated_at = excluded.updated_at""",
            (order.market_id, fill_size, fill_price, fill_price,
             market.category or "", order.token.value, order.token_id,
             1 if is_paper else 0),
        )
        await self._db.commit()

        # Calibration prediction so resolution scoring covers the strategy.
        try:
            await self._calibration.record_prediction(
                order.market_id, signal.claude_prob, market.category or "")
        except Exception as e:
            log.debug("bias_harvest.calibration_error", error=str(e))
