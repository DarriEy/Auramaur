"""Favorite-longshot bias harvest — forecast-free base-load strategy.

Backtest basis (scripts/research/favorite_longshot_backtest.py, 2026-06-09,
414 resolved Polymarket markets with ex-ante signal prices): buying the
favored side when it trades in the 0.60-0.97 band and holding to resolution
returned +3.3% per $1 at 2c slippage (88% win rate); the deep 0.90-0.97 band
won 99.3% of 151 markets vs a 95.5% breakeven. The mirror control (buying
the longshot side at the same moments) lost -87% per $1 in the deep band —
the classic favorite-longshot bias, measured on the bot's own universe. The
edge survives removing the Iran cluster and sports/junk (improves to +4.9%)
but DIES at 4.4c slippage. Live-microstructure research (GWU WP 2026-001 /
Whelan: maker avg ~-9.6% vs taker ~-31.5%) confirms the bias accrues to
MAKERS, not takers — so entries post the BUY at the favored-side BID to
capture the spread (``maker_entry``), and only where a real spread exists
(>= ``maker_min_spread``); never crossed. In paper, ``paper_maker_fill_rate``
deterministically haircuts entries so the book reflects a realistic maker
capture rate rather than assuming 100% fills (the real fill rate is the key
live-validation risk).

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

from auramaur.strategy.protocols import ExecutionMode
from auramaur.experiments.strategies.bias_harvest import (
    BiasHarvestInputs,
    BiasHarvestProposal,
    BiasHarvestRules,
    BiasRejection,
    SOURCE_TAG,
    SOURCE_TAG_WIDE,
    assess_bias_harvest,
    select_bias_band,
)

import hashlib
import time
from datetime import datetime, timezone

import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.strategy.classifier import blocked_category_hit, ensure_category
from auramaur.exchange.models import (
    Confidence,
    Market,
    OrderSide,
    Signal,
)

log = structlog.get_logger()

# How long to skip re-fetching the book for a market that showed no capturable
# maker spread (in-memory, resets on restart). Spreads don't appear within a
# cycle or two, and every fetch queues on the shared CLOB lock.
NO_SPREAD_TTL_SECONDS = 3600.0


class BiasHarvestPillar:
    """Periodic favored-side scanner for Polymarket."""

    # Uniform Strategy contract (see strategy/protocols.py).
    name = "bias_harvest"
    execution_mode = ExecutionMode.GATEWAY_SINGLE

    def __init__(self, db, settings, discovery, exchange, risk_manager,
                 pnl_tracker, calibration) -> None:
        self._db = db
        self._settings = settings
        self._discovery = discovery
        self._exchange = exchange
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._calibration = calibration
        # market_id -> monotonic deadline before which the book is not
        # re-fetched (last fetch showed no capturable maker spread).
        self._no_spread_until: dict[str, float] = {}
        # Shared execution tail (route -> place -> record_fill -> trades-mirror).
        # No router: bias_harvest enters passively at the observed price, so the
        # gateway falls back to prepare_order exactly as before.
        self._gateway = ExecutionGateway(
            router=None, exchange=exchange, exchange_name="polymarket",
            settings=settings, db=db, pnl_tracker=pnl_tracker,
        )

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
            log.info("bias_harvest.cycle", scanned=0, entered=0,
                     book_full=True, open=open_count, cap=cfg.max_open)
            return 0

        markets = await self._discovery.get_markets(limit=cfg.scan_limit)
        from auramaur.data_edge import record_market_snapshot
        await record_market_snapshot(self._db, self.name, markets)
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
        self.last_cycle_detail = {"scanned": len(markets), "entered": entered}
        log.info("bias_harvest.cycle", scanned=len(markets), entered=entered)
        return entered

    # ------------------------------------------------------------------
    # Entry pipeline
    # ------------------------------------------------------------------

    def _band_check(self, market: Market) -> tuple[float, bool, str, str] | None:
        """Return (favored_price, buy_yes, action_desc, source_tag) if in band.

        Applies the bi-directional favorite-longshot bias rules:
        - [0.60, 0.70): buy favored (underpriced favorite)
        - [0.70, 0.90): buy longshot — paper showed the shallow tier's
          favorites busting well above their implied rate (the band_lo
          history in config/settings.py), i.e. the longshot side was the
          cheap one there.
        - [0.90, 0.97): buy favored (underpriced favorite) — the proven,
          graduated deep-band rule.

        The deep band keeps the original ``bias_harvest`` attribution (its
        live probation record was earned there); the two NEW regimes are
        tagged ``bias_harvest_wide`` so they get their OWN graduation cells —
        unproven regimes start ladder-paper-forced and must earn live status
        on their own ledger instead of inheriting the deep band's promotion
        (same isolation idiom as resolution_lens_kalshi).
        """
        cfg = self._settings.bias_harvest
        decision = select_bias_band(
            market.outcome_yes_price,
            band_lo=cfg.band_lo,
            band_hi=cfg.band_hi,
        )
        if decision is None:
            return None
        return (
            decision.favored_price,
            decision.buy_yes,
            decision.action_description,
            decision.source_tag,
        )

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
        # gets corrected after entry. blocked_category_hit checks the stored label
        # OR a fresh classification (mislabel-safe, like the gateway), against the
        # global block AND the bias-harvest no-edge list (weather/sports/politics_us).
        excluded = set(self._settings.risk.blocked_categories) | set(cfg.exclude_categories)
        hit = blocked_category_hit(excluded, market.question, market.description,
                                   market.category)
        if hit:
            log.debug("bias_harvest.skip_category", market_id=market.id, category=hit)
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

    async def _maker_book(
        self, market: Market, buy_yes: bool
    ) -> tuple[float, float] | None:
        """Return the selected outcome's best bid/ask for pure assessment.

        The favorite-longshot edge accrues to makers (capture the spread), not
        takers (pay it). We only harvest where we can actually be a maker: there
        must be at least ``maker_min_spread`` between bid and ask, else posting at
        the bid is ~ taking and the edge dies in slippage (the backtest's 4.4c
        cliff). Execution-only — the risk/divergence accounting stays on the
        observed price (see ``_try_enter``)."""
        token_id = market.clob_token_yes if buy_yes else market.clob_token_no
        if not token_id:
            return None
        try:
            book = await self._exchange.get_order_book(token_id)
        except Exception as e:
            log.debug("bias_harvest.book_fetch_failed", market_id=market.id, error=str(e))
            return None
        best_bid, best_ask = book.best_bid, book.best_ask
        from auramaur.data_edge import DataDelivery, record_delivery
        await record_delivery(self._db, DataDelivery(
            strategy=self.name, component="order_book",
            status="ok" if best_bid is not None and best_ask is not None else "empty",
            provider="polymarket_clob", market_id=market.id,
            source_at=datetime.now(timezone.utc),
            item_count=len(book.bids) + len(book.asks),
            required_fields=("best_bid", "best_ask"),
        ))
        if best_bid is None or best_ask is None:
            return None
        return best_bid, best_ask

    def _paper_admits(self, market_id: str) -> bool:
        """Deterministic maker-capture gate for the PAPER book: admit only
        ``paper_maker_fill_rate`` of markets (stable hash), modelling that not
        every posted maker bid gets hit. Without it the paper ledger would assume
        100% maker fills and read far too rosy — risky because the graduation
        ladder auto-promotes at 20 positive events. Stable (sha1, not the
        salted built-in hash) so the admitted set is reproducible across runs and
        tests. Live fills are governed by the real book, never this gate."""
        rate = self._settings.bias_harvest.paper_maker_fill_rate
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        h = int(hashlib.sha1(market_id.encode("utf-8")).hexdigest(), 16) % 1000
        return h < int(rate * 1000)

    def _with_favored_price(self, market: Market, buy_yes: bool,
                             maker_price: float) -> Market:
        """A copy of the market with the favored side repriced to the maker bid,
        so ``prepare_order`` builds the order at that price. Execution-only: the
        signal keeps the observed price for the risk/divergence gate."""
        field = "outcome_yes_price" if buy_yes else "outcome_no_price"
        return market.model_copy(update={field: maker_price})

    async def _already_entered_or_held(self, market_id: str) -> bool:
        # Both regime tags count: one shot per market across the whole pillar.
        row = await self._db.fetchone(
            "SELECT 1 FROM signals WHERE market_id = ? AND strategy_source IN (?, ?) LIMIT 1",
            (market_id, SOURCE_TAG, SOURCE_TAG_WIDE),
        )
        if row is not None:
            return True  # one shot per market, ever
        row = await self._db.fetchone(
            "SELECT 1 FROM portfolio WHERE market_id = ? LIMIT 1", (market_id,),
        )
        return row is not None  # held by any strategy/mode — stay out

    async def _open_position_count(self) -> int:
        """Count positions THIS strategy actually entered and still holds.

        Counting `portfolio EXISTS signals(strategy_source)` is unsound twice
        over: a signal is written BEFORE the risk gate, so rejected ideas hold
        slots forever, and the row need not be ours — any pillar's position in
        a market we merely signalled consumed a slot. That shape wedged
        long_horizon at 47/40 on 2026-07-25. cost_basis is the authoritative
        holding record; trades proves we were the one who entered.
        """
        row = await self._db.fetchone(
            """SELECT COUNT(*) AS n FROM portfolio p
               WHERE p.size > 0
                 AND EXISTS (SELECT 1 FROM cost_basis cb
                             WHERE cb.market_id = p.market_id
                               AND UPPER(cb.token) = UPPER(p.token)
                               AND cb.is_paper = p.is_paper AND cb.size > 0)
                 AND EXISTS (SELECT 1 FROM trades t
                             WHERE t.market_id = p.market_id
                               AND t.strategy_source IN (?, ?)
                               AND t.is_paper = p.is_paper)""",
            (SOURCE_TAG, SOURCE_TAG_WIDE),
        )
        return int(row["n"]) if row else 0

    def _build_signal(self, market: Market, p_fav: float, buy_yes: bool,
                      action_desc: str, source_tag: str = SOURCE_TAG) -> Signal:
        cfg = self._settings.bias_harvest
        uplift = cfg.edge_uplift
        # claude_prob is P(YES): shift toward the side we are buying by the
        # measured-bias uplift, clamped inside the book.
        if buy_yes:
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
                f"Bias harvest: {action_desc} at {p_fav:.2f} in band "
                f"[{cfg.band_lo:.2f}, {cfg.band_hi:.2f}); measured "
                f"favorite-longshot bias, no forecast claimed."
            ),
            recommended_side=OrderSide.BUY if buy_yes else OrderSide.SELL,
            strategy_source=source_tag,
        )

    def _signal_from_proposal(
        self, market: Market, proposal: BiasHarvestProposal
    ) -> Signal:
        cfg = self._settings.bias_harvest
        return Signal(
            market_id=market.id,
            market_question=market.question,
            claude_prob=proposal.fair_probability,
            claude_confidence=Confidence.MEDIUM,
            market_prob=proposal.market_probability,
            edge=proposal.edge_percent,
            evidence_summary=(
                f"Bias harvest: {proposal.action_description} at "
                f"{proposal.favored_price:.2f} in band "
                f"[{cfg.band_lo:.2f}, {cfg.band_hi:.2f}); measured "
                f"favorite-longshot bias, no forecast claimed."
            ),
            recommended_side=OrderSide.BUY if proposal.buy_yes else OrderSide.SELL,
            strategy_source=proposal.source_tag,
        )

    def _rules(self) -> BiasHarvestRules:
        cfg = self._settings.bias_harvest
        return BiasHarvestRules(
            band_lo=cfg.band_lo,
            band_hi=cfg.band_hi,
            edge_uplift=cfg.edge_uplift,
            stake_usd=cfg.stake_usd,
            min_liquidity=cfg.min_liquidity,
            min_hours_to_resolution=cfg.min_hours_to_resolution,
            max_days_to_resolution=cfg.max_days_to_resolution,
            skip_disputed=cfg.skip_disputed,
            maker_entry=cfg.maker_entry,
            maker_min_spread=cfg.maker_min_spread,
            paper=cfg.paper,
        )

    async def _try_enter(self, market: Market) -> bool:
        cfg = self._settings.bias_harvest
        band = self._band_check(market)
        if band is None or not self._eligible(market):
            return False
        if await self._already_entered_or_held(market.id):
            return False
        _, buy_yes, _, _ = band
        best_bid = best_ask = None

        # Maker entry: post the BUY at the favored-side bid (capture the spread)
        # instead of paying the observed price. The signal/divergence accounting
        # stays on the observed price (so the divergence filter still passes at
        # edge_uplift); only the ORDER is built at the maker bid, so the captured
        # spread lands in the fill -> cost_basis -> pnl_ledger where graduation
        # reads it. order_market carries the maker price for prepare_order only.
        order_market = market
        if cfg.maker_entry:
            # Paper realism: not every posted maker bid is hit. Admit only a
            # deterministic fraction so the paper book reflects a real capture
            # rate. Gate BEFORE the book fetch and BEFORE persisting the signal
            # — the hash verdict never changes, so fetching a book for a
            # never-admitted market is pure wasted CLOB traffic.
            if cfg.paper and not self._paper_admits(market.id):
                return False
            # No-spread cooldown: the widened band makes many more candidates
            # reach this point every cycle, and each book fetch serializes on
            # the shared CLOB lock (the #249 contention lock). A market that
            # just showed no capturable spread won't grow one within a cycle
            # or two — skip re-fetching it for a while.
            now = time.monotonic()
            until = self._no_spread_until.get(market.id, 0.0)
            if now < until:
                return False
            book = await self._maker_book(market, buy_yes)
            if book is None:
                self._no_spread_until[market.id] = now + NO_SPREAD_TTL_SECONDS
                return False  # no capturable spread -> don't harvest as a taker
            best_bid, best_ask = book

        excluded = set(self._settings.risk.blocked_categories) | set(cfg.exclude_categories)
        category_blocked = bool(blocked_category_hit(
            excluded, market.question, market.description, market.category
        ))
        proposal_assessment = assess_bias_harvest(BiasHarvestInputs(
            market_id=market.id,
            yes_price=market.outcome_yes_price,
            no_price=market.outcome_no_price,
            active=market.active,
            venue=market.exchange or "polymarket",
            dispute_risk=market.dispute_risk,
            liquidity=market.liquidity,
            category_blocked=category_blocked,
            end_at=market.end_date,
            as_of=datetime.now(timezone.utc),
            already_entered_or_held=False,
            paper_admitted=True,
            selected_token_available=bool(
                market.clob_token_yes if buy_yes else market.clob_token_no
            ),
            best_bid=best_bid,
            best_ask=best_ask,
        ), self._rules())
        proposal = proposal_assessment.proposal
        if proposal is None:
            if proposal_assessment.rejection in {
                BiasRejection.BOOK_UNAVAILABLE,
                BiasRejection.SPREAD_TOO_THIN,
                BiasRejection.INVALID_MAKER_PRICE,
            }:
                self._no_spread_until[market.id] = time.monotonic() + NO_SPREAD_TTL_SECONDS
            return False
        self._no_spread_until.pop(market.id, None)
        if cfg.maker_entry:
            order_market = self._with_favored_price(
                market, proposal.buy_yes, proposal.entry_price
            )

        signal = self._signal_from_proposal(market, proposal)
        await self._persist_signal(signal, market)

        # Full risk gate — all checks apply; never bypassed.
        decision = await self._risk.evaluate(signal, market)
        if not decision.approved or decision.position_size <= 0:
            log.info("bias_harvest.risk_rejected", market_id=market.id,
                     reason=decision.reason)
            return False
        size = min(decision.position_size, cfg.stake_usd)

        # Passive limit at the observed price (prepare_order uses the seen token
        # price, tick-rounded — no improvement, no crossing). Paper-forced
        # entries pass force_paper so the order carries dry_run=True regardless
        # of the global live gates; the graduation ladder (decision.force_paper)
        # can also paper-force, never un-paper. The gateway owns route -> place
        # -> record_fill -> trades-mirror; the pillar keeps its own
        # portfolio + calibration writes below.
        force_paper = cfg.paper or getattr(decision, "force_paper", False)
        res = await self._gateway.submit(TradeIntent(
            signal=signal, market=order_market, size_dollars=size,
            force_paper=force_paper))
        if res.status not in ("filled", "paper", "partial", "pending"):
            log.warning("bias_harvest.order_rejected", market_id=market.id,
                        status=res.status, error=res.reason)
            return False

        await self._record_position(signal, market, res.order, res.result)
        log.info("bias_harvest.entered", market_id=market.id,
                 token=res.order.token.value, price=res.order.price,
                 size=res.order.size, paper=res.result.is_paper)
        return True

    # ------------------------------------------------------------------
    # Bookkeeping — same rails as the engine
    # ------------------------------------------------------------------

    async def _persist_signal(self, signal: Signal, market: Market) -> None:
        # 2026-08-04 (#353 phase 3): markets stub + signals row land in one
        # span. The trailing commit() this replaced was dead under the
        # autocommit connection (no-op since eed51b8) — the span provides
        # the atomicity it implied.
        async with self._db.transaction(owner="bias_harvest.signal"):
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
                 signal.recommended_side.value, signal.strategy_source),
            )

    async def _record_position(self, signal: Signal, market: Market,
                               order, result) -> None:
        """Write the portfolio row + calibration prediction for a filled entry.

        The fill (-> cost_basis -> pnl_ledger) and the trades-mirror are owned
        by :class:`ExecutionGateway`; this keeps only the two writes the gateway
        does not (yet) own: the portfolio row the resolution tracker settles
        against, and the calibration prediction for resolution scoring.
        """
        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price
        is_paper = bool(result.is_paper)

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
