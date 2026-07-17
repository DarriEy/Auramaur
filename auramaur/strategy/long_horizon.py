"""Long-horizon favorite underpricing — structural underconfidence at long tenor.

Research basis (arXiv 2602.19520, 292M trades across Kalshi+Polymarket):
prediction-market prices are systematically UNDERCONFIDENT at long horizons — the
calibration slope rises from ~0.99 within an hour of resolution to ~1.32 beyond a
month, i.e. long-dated favorites are underpriced (a 70c contract resolves YES
~75% of the time). This is a STRUCTURAL, cross-venue property, not a forecast: we
apply the published ``slope`` to the market's OWN price (logit space) to get a
fair, and trade only the favored side's underpricing — never a view of our own.

Distinct from bias_harvest, which harvests the NEAR-resolution favorite-longshot
bias in the deep band (0.90-0.97, <=45 days). This pillar trades the LONG-horizon
(> min_days) MODERATE-favorite band where the slope correction is materially
positive and capital lock-up is the cost.

CAVEAT (why it's restricted + paper-forced): the paper's effect is STRONGEST in
political markets — exactly the bot's documented no-edge zone (politics_us /
politics_intl). Politics is therefore EXCLUDED: this cell exists to test whether
the long-horizon underconfidence GENERALIZES to tech/crypto/macro net of cost.
``slope`` defaults BELOW the paper's 1.32 because it rests on a single
non-peer-reviewed preprint. PAPER-FORCED, its own graduation cell. v1 enters at
the observed price; maker entry (see bias_harvest) is the obvious execution
follow-up once gross edge holds.

Rides the same rails as bias_harvest: signals + trades (attribution), the full
RiskManager gate (all 15 checks, never bypassed), a calibration prediction, and a
portfolio row the resolution tracker settles. One entry per market, ever; never
enters a market already held by any strategy/mode.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.exchange.models import (
    Confidence,
    Market,
    Order,
    OrderSide,
    OrderType,
    Signal,
    TokenType,
)
from auramaur.strategy.classifier import blocked_category_hit, ensure_category
from auramaur.strategy.protocols import ExecutionMode

log = structlog.get_logger()


def calibrated_fair(price: float, slope: float) -> float:
    """Slope-corrected fair probability for a price, in logit space:
    ``true_logit = slope * logit(price)`` (arXiv 2602.19520). For a favorite
    (price > 0.5) and slope > 1 this returns a value ABOVE the price — the
    documented long-horizon underpricing. Pure function (the tested core)."""
    p = min(max(price, 1e-6), 1.0 - 1e-6)
    logit = math.log(p / (1.0 - p))
    return 1.0 / (1.0 + math.exp(-slope * logit))


class LongHorizonPillar:
    """Periodic long-horizon favored-side scanner.

    Runs per venue: the classic Polymarket instance, and (when
    ``kalshi_enabled``) a second instance over Kalshi attributed to
    ``long_horizon_kalshi`` — its own graduation cell, so the venue earns
    live status on its own ledger (the resolution_lens_kalshi idiom). The
    Kalshi cell also ADMITS politics_intl: the live evidence for the slope
    edge on Kalshi is precisely long-tenor geopolitical persistence (the
    2026-04→07 live book: fear-fade ladders and status-quo favorites carried
    the whole gain), and the research the pillar is built on found the slope
    strongest in politics. These are persistence trades on the market's own
    price, not forecasts — the no-politics-forecasting rule is about
    forecasts.
    """

    # Uniform Strategy contract (see strategy/protocols.py).
    name = "long_horizon"
    execution_mode = ExecutionMode.GATEWAY_SINGLE

    def __init__(self, db, settings, discovery, exchange, risk_manager,
                 pnl_tracker, calibration, *, venue: str = "polymarket") -> None:
        self._db = db
        self._settings = settings
        self._discovery = discovery
        self._exchange = exchange
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._calibration = calibration
        self._venue = venue
        self._tag = "long_horizon" if venue == "polymarket" else f"long_horizon_{venue}"
        self._gateway = ExecutionGateway(
            router=None, exchange=exchange, exchange_name=venue,
            settings=settings, db=db, pnl_tracker=pnl_tracker,
        )

    # ------------------------------------------------------------------
    # Scan cycle
    # ------------------------------------------------------------------

    async def run_once(self) -> int:
        cfg = self._settings.long_horizon
        if not cfg.enabled:
            return 0
        try:
            await self._harvest_decay(cfg)
        except Exception as e:
            log.error("long_horizon.harvest_error", error=str(e))
        open_count = await self._open_position_count()
        if open_count >= cfg.max_open:
            log.debug("long_horizon.book_full", open=open_count, cap=cfg.max_open)
            return 0
        markets = await self._scan_long_dated(cfg)
        entered = 0
        band = 0
        for market in markets:
            if self._favored(market) is not None and self._eligible(market):
                band += 1
            if entered >= cfg.max_entries_per_cycle:
                break
            if open_count + entered >= cfg.max_open:
                break
            try:
                if await self._try_enter(market):
                    entered += 1
            except Exception as e:
                log.error("long_horizon.entry_error", market_id=market.id, error=str(e))
        # Log EVERY cycle (the settlement_arb #246 lesson): a quiet log must
        # be distinguishable from a dead task, and the funnel numbers are the
        # diagnosis when a venue instance silently finds nothing.
        log.info("long_horizon.cycle", venue=self._venue, scanned=len(markets),
                 in_band=band, open=open_count, entered=entered)
        return entered

    async def _scan_long_dated(self, cfg) -> list[Market]:
        """Fetch the LONG-DATED slice directly via Gamma's resolution-date window,
        paginated. The old single get_markets(limit=scan_limit) call was a no-op:
        Gamma caps a page at 100 and orders by VOLUME, and the top-100-by-volume is
        bimodal (near-term or year-out) — it contains ZERO 14-365d moderate
        favorites, so scan_limit>100 never helped. Querying the [min_days, max_days]
        end-date window ordered by liquidity puts the real candidates in front."""
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        emin = (now + timedelta(days=cfg.min_days_to_resolution)).strftime("%Y-%m-%dT%H:%M:%SZ")
        emax = (now + timedelta(days=cfg.max_days_to_resolution)).strftime("%Y-%m-%dT%H:%M:%SZ")
        out: list[Market] = []
        try:
            for off in range(0, max(int(cfg.scan_limit), 1), 100):
                page = await self._discovery.get_markets(
                    limit=100, offset=off, order="liquidity",
                    end_date_min=emin, end_date_max=emax)
                if not page:
                    break
                out.extend(page)
        except TypeError:
            # A discovery without the date-window kwargs (older client / test stub):
            # fall back to the plain scan so the pillar still runs.
            out = await self._discovery.get_markets(limit=cfg.scan_limit)
        return out

    # ------------------------------------------------------------------
    # Entry pipeline
    # ------------------------------------------------------------------

    def _favored(self, market: Market) -> tuple[float, bool] | None:
        """(favored_price, favored_is_yes) if the favored side is in the moderate
        band, else None."""
        cfg = self._settings.long_horizon
        p_yes = market.outcome_yes_price
        if not (0.0 < p_yes < 1.0):
            return None
        p_fav = max(p_yes, 1.0 - p_yes)
        if not (cfg.band_lo <= p_fav <= cfg.band_hi):
            return None
        return p_fav, p_yes >= 0.5

    def _eligible(self, market: Market) -> bool:
        cfg = self._settings.long_horizon
        if not market.active:
            return False
        if (market.exchange or "polymarket") != self._venue:
            return False  # each instance owns exactly its venue's book
        if market.liquidity < cfg.min_liquidity:
            return False
        # Classify before the block (mislabel-safe, like bias_harvest/the gateway):
        # exclude the global block AND politics (the paper's effect is strongest
        # there but it's the bot's no-edge zone — we test generalization elsewhere).
        venue_excludes = (cfg.kalshi_exclude_categories if self._venue == "kalshi"
                          else cfg.exclude_categories)
        excluded = set(self._settings.risk.blocked_categories) | set(venue_excludes)
        hit = blocked_category_hit(excluded, market.question, market.description,
                                   market.category)
        if hit:
            log.debug("long_horizon.skip_category", market_id=market.id, category=hit)
            return False
        if market.end_date is None:
            return False  # need a horizon to know it's long-dated
        end = market.end_date
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        days_left = (end - datetime.now(timezone.utc)).total_seconds() / 86400.0
        if days_left < cfg.min_days_to_resolution:
            return False  # not a LONG-horizon market — no underconfidence edge
        if days_left > cfg.max_days_to_resolution:
            return False  # capital lock-up cap
        return True

    async def _already_entered_or_held(self, market_id: str) -> bool:
        row = await self._db.fetchone(
            "SELECT 1 FROM signals WHERE market_id = ? AND strategy_source = ? LIMIT 1",
            (market_id, self._tag),
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
                               AND s.strategy_source = ?)""",
            (self._tag,),
        )
        return int(row["n"]) if row else 0

    def _build_signal(self, market: Market, p_fav: float, fav_is_yes: bool,
                      edge_fav: float) -> Signal:
        cfg = self._settings.long_horizon
        fair_fav = p_fav + edge_fav  # slope-corrected fair on the favored side
        # Express the fair as P(YES) for the risk/divergence accounting.
        claude_prob = min(0.99, fair_fav) if fav_is_yes else max(0.01, 1.0 - fair_fav)
        return Signal(
            market_id=market.id,
            market_question=market.question,
            claude_prob=claude_prob,
            claude_confidence=Confidence.MEDIUM,
            market_prob=market.outcome_yes_price,
            edge=edge_fav * 100.0,
            evidence_summary=(
                f"Long-horizon underconfidence: favored side at {p_fav:.2f}, "
                f"slope {cfg.slope:.2f} -> fair {fair_fav:.2f} "
                f"(+{edge_fav * 100:.1f}pts); no forecast claimed."
            ),
            recommended_side=OrderSide.BUY if fav_is_yes else OrderSide.SELL,
            strategy_source=self._tag,
            # Names the structural gap so the name-the-gap divergence gate passes:
            # the divergence IS the thesis (published calibration slope), not an
            # unexplained model disagreement.
            mispricing_reason=(
                "structural: long-horizon underconfidence (arXiv 2602.19520) "
                "underprices long-dated favorites"
            ),
        )

    async def _try_enter(self, market: Market) -> bool:
        cfg = self._settings.long_horizon
        fav = self._favored(market)
        if fav is None or not self._eligible(market):
            return False
        if await self._already_entered_or_held(market.id):
            return False
        p_fav, fav_is_yes = fav

        edge_fav = calibrated_fair(p_fav, cfg.slope) - p_fav
        if edge_fav < cfg.min_edge:
            return False  # correction too small at this price to clear cost

        signal = self._build_signal(market, p_fav, fav_is_yes, edge_fav)
        await self._persist_signal(signal, market)

        # Full risk gate — all checks apply; never bypassed.
        decision = await self._risk.evaluate(signal, market)
        if not decision.approved or decision.position_size <= 0:
            log.info("long_horizon.risk_rejected", market_id=market.id,
                     reason=decision.reason)
            return False
        size = min(decision.position_size, cfg.stake_usd)

        force_paper = cfg.paper or getattr(decision, "force_paper", False)
        res = await self._gateway.submit(TradeIntent(
            signal=signal, market=market, size_dollars=size, force_paper=force_paper))
        if res.status not in ("filled", "paper", "partial", "pending"):
            log.warning("long_horizon.order_rejected", market_id=market.id,
                        status=res.status, error=res.reason)
            return False

        await self._record_position(signal, market, res.order, res.result)
        log.info("long_horizon.entered", market_id=market.id,
                 token=res.order.token.value, price=res.order.price,
                 size=res.order.size, edge=round(edge_fav, 3),
                 paper=res.result.is_paper)
        return True

    # ------------------------------------------------------------------
    # Decay harvest — realize the front-loaded premium, don't wait for the bell
    # ------------------------------------------------------------------

    async def _harvest_decay(self, cfg) -> int:
        """Exit positions whose side has captured >= ``take_profit_capture`` of
        the entry->$1 distance.

        The slope/persistence premium decays FRONT-LOADED (the scare recedes,
        the milestone slips) — holding the last cents to a multi-year
        resolution is bad capital velocity, and worse: a cell whose events
        only realize at resolution can never accrue the graduation ladder's
        20-events-in-90-days record. Harvesting the decay converts marks into
        honest realized ledger events on a weeks clock, recycles the locked
        capital, and keeps the strategy judgeable. Marks must be FRESH
        (stale-active rows freeze prices; the #80 lesson) — positions whose
        market row hasn't updated recently are skipped, not exited."""
        capture_floor = float(getattr(cfg, "take_profit_capture", 0.0))
        if capture_floor <= 0:
            return 0
        rows = await self._db.fetchall(
            """SELECT p.market_id, p.token, p.token_id, p.size, p.avg_price,
                      p.is_paper, m.outcome_yes_price, m.clob_token_yes,
                      m.clob_token_no
               FROM portfolio p JOIN markets m ON m.id = p.market_id
               WHERE p.size > 0 AND p.exchange = ?
                 AND m.last_updated >= datetime('now', '-1 day')
                 AND EXISTS (SELECT 1 FROM signals s
                             WHERE s.market_id = p.market_id
                               AND s.strategy_source = ?)""",
            (self._venue, self._tag))
        exited = 0
        for r in rows:
            token = (r["token"] or "YES").upper()
            p_yes = float(r["outcome_yes_price"] or 0.0)
            if not (0.0 < p_yes < 1.0):
                continue
            mark = p_yes if token == "YES" else 1.0 - p_yes
            entry = float(r["avg_price"] or 0.0)
            if entry <= 0 or entry >= 1.0 or mark <= entry:
                continue
            captured = (mark - entry) / (1.0 - entry)
            if captured < capture_floor:
                continue
            token_id = r["token_id"] or (
                r["clob_token_yes"] if token == "YES" else r["clob_token_no"]) or ""
            order = Order(
                market_id=r["market_id"],
                exchange=self._venue,
                token_id=token_id,
                side=OrderSide.SELL,
                token=TokenType(token),
                size=float(r["size"]),
                price=max(0.01, min(0.99, round(mark, 2))),
                order_type=OrderType.LIMIT,
                dry_run=bool(r["is_paper"]) or not self._settings.is_live,
                source="exit",
            )
            res = await self._gateway.submit_exit(
                order, exchange=self._exchange, exchange_name=self._venue)
            if res.status in ("filled", "paper", "partial", "pending"):
                exited += 1
                log.info("long_horizon.decay_harvested", market_id=r["market_id"],
                         tag=self._tag, entry=entry, mark=round(mark, 2),
                         captured=round(captured, 2), paper=bool(r["is_paper"]))
        return exited

    # ------------------------------------------------------------------
    # Bookkeeping — same rails as bias_harvest / the engine
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
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (signal.market_id, signal.claude_prob, signal.claude_confidence.value,
             signal.market_prob, signal.edge, signal.evidence_summary,
             signal.recommended_side.value, self._tag),
        )
        await self._db.commit()

    async def _record_position(self, signal: Signal, market: Market,
                               order, result) -> None:
        """Portfolio row (resolution tracker settles it) + calibration prediction.
        The fill (-> cost_basis -> pnl_ledger) and the trades-mirror are owned by
        ExecutionGateway; mirrors bias_harvest._record_position."""
        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price
        is_paper = bool(result.is_paper)
        await self._db.execute(
            """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
               current_price, unrealized_pnl, category, token, token_id,
               is_paper, updated_at)
               VALUES (?, ?, 'BUY', ?, ?, ?, 0, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                   size = excluded.size,
                   avg_price = excluded.avg_price,
                   current_price = excluded.current_price,
                   updated_at = excluded.updated_at""",
            (order.market_id, self._venue, fill_size, fill_price, fill_price,
             market.category or "", order.token.value, order.token_id,
             1 if is_paper else 0),
        )
        await self._db.commit()
        try:
            await self._calibration.record_prediction(
                order.market_id, signal.claude_prob, market.category or "")
        except Exception as e:
            log.debug("long_horizon.calibration_error", error=str(e))
