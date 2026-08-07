"""ExecutionGateway — the single mechanical path from an approved trade to a
recorded fill.

The canonical entry tail historically lived in
``TradingEngine._build_and_place_order`` and was reimplemented, divergently,
inside every standalone strategy. This component extracts that tail so the
strategies (and exits) route through one place and the duplication — the source
of token/accounting bugs — is deleted.

The gateway changes *who* calls the money path, never the path itself: routing,
the ``place_order`` triple-gate (kill-switch / geoblock / live-enabled), the
paper fork, and ``record_fill`` all keep their existing behavior. Risk
evaluation stays with the caller — the caller passes the already risk-approved
``size_dollars`` and ``force_paper`` on the intent.

``submit`` runs the single-leg path. ``submit_paired`` runs a both-or-nothing
pair (the arb pillars): it builds BOTH orders before placing EITHER, places A
then B, and unwinds a live-pending A if B fails — preserving the atomicity a
per-leg ``submit`` could not.

``cancel_resting`` closes the other half of a resting order's lifecycle. It is
a ROUTING AND AUDIT contract, not a gate: unlike a placement, a cancel reduces
exposure, so it carries none of ``submit``'s risk checks and cannot refuse.
See its docstring for why the kill switch must never reach it.
"""

from __future__ import annotations

import asyncio
import math
import uuid
import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import structlog

from auramaur.broker.router import SmartOrderRouter, UnmarketableSignal
from auramaur.db.database import Database
from auramaur.exchange.models import Fill, Market, Order, OrderResult, OrderSide, Signal
from auramaur.exchange.protocols import ExchangeClient
from auramaur.monitoring.display import show_order, show_order_dropped
from auramaur.research.polymarket_strategies import DecisionTracker
from auramaur.strategy.signals import taker_fee_rate
from config.settings import Settings

log = structlog.get_logger()

# Statuses that count as "the order left the building" (recorded / tracked).
_OK_STATUSES = ("filled", "paper", "partial", "pending")
_FILLED_STATUSES = ("filled", "paper", "partial")


@dataclass
class TradeIntent:
    """A risk-approved decision to trade, handed to the gateway to execute.

    ``size_dollars`` is the caller's risk-approved position size and
    ``force_paper`` carries the graduation ladder's (and the strategy's own
    ``cfg.paper``) downgrade to dry-run.
    """

    signal: Signal
    market: Market
    size_dollars: float
    force_paper: bool = False
    kind: Literal["entry"] = "entry"


@dataclass
class ExecutionResult:
    """The outcome of a gateway submission, exposing every decision point.

    ``result`` is the underlying :class:`OrderResult` for any submitted order
    (including a rejected one) and ``None`` when the trade was skipped before
    submission (unmarketable / build failure) — preserving the legacy
    ``_build_and_place_order`` return contract (``None`` == not submitted).
    """

    status: Literal["filled", "paper", "partial", "pending", "rejected", "skipped"]
    order: Order | None = None
    result: OrderResult | None = None
    fill: Fill | None = None
    reason: str = ""


@dataclass
class CancelResult:
    """The outcome of a gateway cancel. Never an exception, never a refusal.

    ``status`` is one of:

    ``cancelled``  the venue acknowledged the cancel.
    ``paper``      a paper order — intercepted, never sent to a venue client.
    ``rejected``   the venue declined (returns False, e.g. already matched).
    ``failed``     the venue call raised; the exception was contained here.
    ``skipped``    no usable order id was supplied — nothing to cancel.
    """

    order_id: str
    status: Literal["cancelled", "paper", "rejected", "failed", "skipped"]
    reason: str = ""

    @property
    def ok(self) -> bool:
        """True when the order is no longer resting because of this call."""
        return self.status in ("cancelled", "paper")


def booked_as_position(res: ExecutionResult | OrderResult | None) -> bool:
    """True when a submission actually EXECUTED and may be booked as a holding.

    THE single predicate for "did this order become a position". Every caller
    that writes a ``portfolio`` row, a per-leg record, or a fill must ask this
    and nothing else.

    It is deliberately the same expression ``_record_result`` applies before it
    writes the fill (``status in _FILLED_STATUSES and filled_size > 0``) — and
    that method now calls THIS function, so the agreement holds by
    construction rather than by nine hand-copied predicates staying in sync. A
    pillar's portfolio row can no longer claim a position the gateway's own
    ``fills``/``cost_basis`` writes never recorded.

    Why the test has to be both halves:

    ``status``      Polymarket's LIVE ``place_order`` ALWAYS returns
                    ``"pending"``, and ``PaperTrader`` defers every
                    non-marketable (maker-priced) order to ``"pending"`` too.
                    Resting is the NORMAL outcome, not the exception.
    ``filled_size`` a ``"paper"``/``"filled"`` status with size 0 is a refusal
                    that got stamped, not an execution — and every
                    ``_record_*`` helper in the codebase falls back to
                    ``order.size`` when ``filled_size`` is 0, so booking one
                    writes a FULL-SIZE phantom.

    Accepts either half of the pair so one predicate covers both call shapes:
    an :class:`ExecutionResult` from the gateway (pillars) or a raw
    :class:`OrderResult` from an exchange adapter / the order monitor. An
    ``ExecutionResult`` that never reached placement carries ``result=None``
    (``status="skipped"``), which is not a position either.

    ``status`` is read from the object the caller passed and ``filled_size``
    from the underlying ``OrderResult``. For an ``OrderResult`` those are the
    same object; for an ``ExecutionResult`` the two agree by construction —
    ``_record_result`` is the ONLY place that builds one carrying a
    ``result``, and it copies ``status=result.status`` verbatim. Attribute
    probing rather than ``isinstance`` so a test double of either shape is
    read the same way the real object would be.
    """
    if res is None:
        return False
    if getattr(res, "status", None) not in _FILLED_STATUSES:
        return False
    inner = getattr(res, "result", None)
    size = getattr(res if inner is None else inner, "filled_size", None)
    return size is not None and size > 0


async def materialize_paper_portfolio_row(db, order) -> None:
    """Project the just-updated paper ``cost_basis`` row into ``portfolio``.

    THE single maintainer of a paper ``portfolio`` row after a booked fill,
    called from the gateway's ``_record_result`` (instant paper fills — entries
    AND exits) and from the order monitor's deferred-fill path. One
    implementation so the two cannot diverge — the same doctrine as
    :func:`booked_as_position` above.

    Why anything must do this at all: ``PnLTracker.record_fill`` writes
    ``fills`` and ``cost_basis`` and NOTHING else, and position sync is
    mode-scoped (``is_paper_flag = 0 if settings.is_live else 1``), so in a
    live bot nothing else maintains paper rows. On the ENTRY side that left
    deferred fills invisible to ``RiskManager`` and every pillar's
    ``_open_position_count`` (measured 2026-08-06: 13 such long_horizon rows
    and 18 llm rows, all settling correctly yet uncounted). On the EXIT side
    the mirror image shipped a day later: an exit that filled INSTANTLY
    through paper interception zeroed ``cost_basis`` and left the portfolio
    row at FULL SIZE until resolution cleanup (observed live 2026-08-06, the
    evening after the deferred-fill fix deployed) — inflating exposure and
    ``max_open`` counts for however long the market kept trading.

    PROJECTED FROM cost_basis, not from the triggering fill, so the two
    cannot diverge. ``cost_basis`` carries the CUMULATIVE size and
    weighted-average cost for (market, token, mode) across every fill — a
    fill-shaped upsert would undercount the second fill in a market (the
    portfolio upsert REPLACES size) and would leave a stale full-size row
    after a partial exit. It is also the exact row
    ``resolution_tracker._settle_position`` falls back to when no portfolio
    row exists.

    Cannot double-book at settlement. ``_settle_position`` reads the
    portfolio row OR the cost_basis row (portfolio preferred) to obtain
    (size, entry_price), but derives its idempotency key from neither:
    ``source_ref = f"settle:{market_id}:{canon_token}:{is_paper_flag}"``
    comes from the position KEY alone, and ``_prior_settlement`` dedupes on
    it. Materializing the row therefore changes only WHICH branch supplies
    the numbers — and both branches read the same cost_basis values.
    ``side`` is hardcoded "BUY" for the same reason ``_settle_position``'s
    cost_basis branch hardcodes it: Polymarket holdings are always long.
    """
    row = await db.fetchone(
        "SELECT size, avg_cost, token, token_id FROM cost_basis "
        "WHERE market_id = ? AND is_paper = 1 AND token = ?",
        (order.market_id, order.token.value),
    )
    if row is None:
        return
    size = float(row["size"])
    if size <= 0:
        # The fill closed the holding. Leaving the portfolio row behind is
        # the stale-exit phantom: it survives until _settle_position's
        # resolution-time cleanup, inflating exposure the whole way — so
        # drop it now, the same cleanup settlement performs at cost_basis
        # zero.
        await db.execute(
            "DELETE FROM portfolio WHERE market_id = ? AND is_paper = 1 "
            "AND UPPER(token) = UPPER(?)",
            (order.market_id, order.token.value),
        )
        return
    price = float(row["avg_cost"])
    await db.execute(
        """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
           current_price, unrealized_pnl, category, token, token_id,
           is_paper, updated_at)
           VALUES (?, ?, 'BUY', ?, ?, ?, 0,
                   COALESCE((SELECT category FROM markets WHERE id = ?), ''),
                   ?, ?, 1, datetime('now'))
           ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
               size = excluded.size,
               avg_price = excluded.avg_price,
               current_price = excluded.current_price,
               updated_at = excluded.updated_at""",
        (order.market_id, order.exchange or "polymarket", size, price, price,
         order.market_id, row["token"] or order.token.value,
         row["token_id"] or order.token_id),
    )


class ExecutionGateway:
    """Routes an approved :class:`TradeIntent` through route → place → record."""

    def __init__(
        self,
        *,
        router: SmartOrderRouter | None,
        exchange: ExchangeClient,
        exchange_name: str,
        settings: Settings,
        db: Database,
        pnl_tracker,
    ) -> None:
        self.router = router
        self.exchange = exchange
        self.exchange_name = exchange_name
        self.settings = settings
        self.db = db
        self.pnl_tracker = pnl_tracker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(self, intent: TradeIntent) -> ExecutionResult:
        """Build an order (via router or direct) and place it, recording the
        fill. Behavior-identical to the former
        ``TradingEngine._build_and_place_order``.
        """
        is_live = self.settings.is_live and not intent.force_paper
        order = await self._build_order(
            intent.signal, intent.market, intent.size_dollars, is_live,
            exchange=self.exchange, router=self.router,
        )
        if order is None:
            return ExecutionResult(status="skipped", reason="not submitted")
        capped = await self._exceeds_market_cap(order, is_live=is_live)
        if capped is not None:
            # Surface the drop on the console like the other pre-submission
            # gates, and bench the market so cycles stop re-analyzing it while
            # the position sits at cap (the held-market filter is the primary
            # exclusion; this covers partially-capped markets it lets through).
            show_order_dropped(order.market_id, capped)
            try:
                await self._serialized_write(
                    """INSERT OR REPLACE INTO order_build_drops
                       (market_id, blocked_until, reason)
                       VALUES (?, datetime('now', '+2 hours'), ?)""",
                    (order.market_id, capped),
                )
            except Exception:
                pass
            return ExecutionResult(status="skipped", reason=capped)
        decision_id = await self._capture_decision(intent, order)
        order.decision_id = decision_id
        return await self._place_and_record(
            order, strategy_source=intent.signal.strategy_source,
            signal_id=getattr(intent.signal, "id", None),
            exchange=self.exchange, exchange_name=self.exchange_name,
            decision_id=decision_id)

    def _per_market_cap(self, order: Order) -> float:
        """Per-(market, token) exposure ceiling, scoped by VENUE.

        risk.max_stake_abs_ceiling sizes a stake on a binary prediction
        contract. It is meaningless for a broker instrument: 21 shares of a
        $28.57 ETF is $600 of perfectly ordinary exposure, and applying the
        prediction-market ceiling to it blocks every IBKR position outright
        (found 2026-07-27 the first time an instrument was routed through this
        gateway). Broker books carry their own envelope — per-book position
        counts and a paper budget — so the ceiling here is the book budget,
        with the prediction-market ceiling still governing everything else.

        Deliberately not a global raise: the prediction-market guard exists
        because stacked sub-cap entries reached ~$90 on a documented $25 limit,
        and widening it for IBKR's sake would reopen exactly that.
        """
        venue = (order.exchange or self.exchange_name or "").lower()
        if venue == "ibkr":
            return float(getattr(self.settings.ibkr, "paper_budget_usd", 5000.0))
        return float(getattr(self.settings.risk, "max_stake_abs_ceiling", 25.0))

    async def _exceeds_market_cap(self, order: Order, *, is_live: bool) -> str | None:
        """Aggregate per-(market, token) stake cap. Returns a skip-reason string
        when this BUY would push TOTAL exposure to a market SIDE past the
        documented ceiling, else None.

        The risk manager's max_stake check only sees a single order, so the bot
        could STACK sub-cap entries into an over-cap position (legacy directional
        favorites reached ~$90 across stacked orders, each individually under the
        limit). This guard runs at the layer where the YES/NO token is finally
        known, so it scopes by (market, token): a side's existing holdings plus
        this order can't exceed the cap. Scoping by token, not market, is
        deliberate — internal arb legitimately holds YES *and* NO of one market,
        so a market-wide sum would false-block the opposite leg.

        Only BUY entries are bounded (a SELL reduces exposure). Scoped to the
        order's own mode (paper vs live) so a paper add isn't blocked by a live
        holding and vice-versa.
        """
        if order.side != OrderSide.BUY:
            return None
        cap = self._per_market_cap(order)
        is_paper_flag = 0 if is_live else 1
        try:
            row = await self.db.fetchone(
                "SELECT COALESCE(SUM(size * avg_cost), 0) AS held "
                "FROM cost_basis WHERE market_id = ? AND UPPER(token) = UPPER(?) "
                "AND is_paper = ? AND size > 0",
                (order.market_id, order.token.value, is_paper_flag),
            )
        except Exception as e:
            # This is the only guard that sees aggregate exposure. The per-order
            # risk check cannot protect against stacked entries, so an unknown
            # aggregate must never become permission to place another live BUY.
            log.error("gateway.market_cap_read_failed",
                      market_id=order.market_id, token=order.token.value,
                      is_live=is_live, error=str(e))
            return "market_cap: aggregate exposure unavailable"
        held = float(row["held"]) if row else 0.0
        proposed = order.size * order.price
        # Both sides are notional; the cap is a capital budget. They are the
        # same number for a prediction contract, a share or an ETF (ratio 1.0,
        # so nothing below changes for them). They are NOT the same for a
        # leveraged instrument: 35.8 shares of 7203.T at Y2,501 is Y89,615 of
        # notional against ~$578 of capital, which read as a $5,000 breach and
        # would have stopped an entry the multiasset book takes correctly today.
        # Converting here rather than storing capital in cost_basis keeps the
        # change out of a table the paper wallet, resolution sweep and reporting
        # all read.
        ratio = order.capital_ratio
        held, proposed = held * ratio, proposed * ratio
        if held + proposed > cap + 1e-9:
            reason = (
                f"market_cap: ${held:.2f} held + ${proposed:.2f} new on "
                f"{order.market_id}/{order.token.value} exceeds ${cap:.2f}"
            )
            log.info(
                "gateway.market_cap_block",
                market_id=order.market_id,
                token=order.token.value,
                held=round(held, 2),
                proposed=round(proposed, 2),
                cap=cap,
            )
            return reason
        return None

    async def submit_exit(
        self,
        order: Order,
        *,
        exchange: ExchangeClient,
        exchange_name: str,
        strategy_source: str = "exit",
    ) -> ExecutionResult:
        """Place a PREBUILT exit order and record it.

        Exits skip risk (the portfolio monitor already decided) and skip routing
        (the caller prices the SELL against the live bid). The gateway places it
        and writes the pending trades-mirror — so the order monitor, which
        finalizes a live exit's fill asynchronously, UPDATEs that row instead of
        inserting an 'order_monitor'-attributed one (and an immediate paper-mode
        fill is recorded here). No double-write: a live exit is pending at
        placement (filled_size 0), so the fill is recorded once, by the monitor.
        """
        if not order.source:
            order.source = strategy_source
        return await self._place_and_record(
            order, strategy_source=strategy_source, signal_id=None,
            exchange=exchange, exchange_name=exchange_name)

    @staticmethod
    def _scope_config(source: str, payload: dict) -> dict:
        """Freeze only the parameters the strategy actually uses.

        strategy_version is a hash of the strategy's config, and
        _prospective_stats joins on the LATEST version — so any change to the
        hashed config discards every decision captured under the old one and
        restarts the 14-day holdout. That is correct when the strategy's own
        parameters move, and wrong when someone else's do.

        `settings.ibkr` is one section covering 77 fields across the ETF arms,
        six multiasset books, the equity client, FX and options. Hashing all of
        it meant tuning etf_signal_horizon_days reset the graduation clock for
        ibkr_international_equity_paper, a book that does not read the field.
        Under active tuning no IBKR book could ever hold a frozen parameter set
        for the 30 days its own bar requires.

        Non-IBKR sections are returned untouched: `nlp` for llm and
        `agent_trader` for the agent arms are already the config those
        strategies actually run on.
        """
        if not source.startswith("ibkr_") or not payload:
            return payload
        if source.startswith("ibkr_etf_"):
            return {k: v for k, v in payload.items() if k.startswith("etf_")}
        # ibkr_<book>_paper: the book's own config plus the shared multiasset
        # knobs, and nothing from the ETF arms or the other five books.
        book = source[len("ibkr_"):-len("_paper")] if source.endswith("_paper") else ""
        scoped = {k: v for k, v in payload.items()
                  if k.startswith("multiasset_") and k != "multiasset_books"}
        books = payload.get("multiasset_books") or {}
        if book and isinstance(books, dict) and book in books:
            scoped["book"] = books[book]
        return scoped

    async def _live_book(self, order: Order):
        """The book the order was actually BUILT against.

        _capture_decision used to resolve best_bid/best_ask solely from
        ``orderbook_snapshots`` — a table a SEPARATE recorder task populates on
        its own cadence. At decision time that row frequently does not exist:
        measured 2026-07-28, 51 of 111 filled decisions were on markets the
        recorder had never sampled, and the remainder missed on token_id
        (Kalshi orders carry ticker-style ids while the book holds Polymarket
        CLOB numerics).

        The consequence was total. best_ask landed NULL, so _place_and_record
        could not tell whether a paper fill crossed and stamped it 'synthetic'
        — which is not in graduation.credible_fill_evidence. 109 of 111 filled
        decisions were therefore uncountable FOREVER, and with
        require_executable_fills on, no strategy could graduate on merit.

        Recording the state actually used, rather than hoping another component
        recorded it first, is the fix. Best-effort by construction: any failure
        leaves the previous behaviour untouched.
        """
        fetch = getattr(self.exchange, "get_order_book", None)
        if not callable(fetch) or not order.token_id:
            return None
        try:
            book = await fetch(order.token_id)
        except Exception:  # noqa: BLE001 - evidence, never the money path
            return None
        def _price(value):
            # Must be a real number. A duck-typed or mocked book hands back
            # whatever attribute access produces, and writing that into a
            # decision snapshot would poison the evidence with something no
            # comparison can interpret.
            try:
                if value is None or isinstance(value, bool):
                    return None
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) and number > 0 else None

        bid, ask = _price(getattr(book, "best_bid", None)), _price(
            getattr(book, "best_ask", None))
        if bid is None and ask is None:
            return None
        return {"best_bid": bid, "best_ask": ask}

    async def _capture_decision(self, intent: TradeIntent, order: Order) -> int | None:
        """Persist a parameter-frozen executable decision before submission."""
        try:
            source = intent.signal.strategy_source or "llm"
            section_name = (
                "agent_trader" if source.startswith("agent_trader_") else
                "technical" if source.startswith("technical_") else
                # An IBKR book cell (ibkr_etf_luna, ibkr_fx_paper, ...) has no
                # settings section of its own; its parameters live under
                # settings.ibkr. Without this the frozen strategy_version
                # hashed an EMPTY config, so a book could be re-tuned mid-clock
                # and keep claiming the same holdout experiment.
                "ibkr" if source.startswith("ibkr_") else
                # llm_kalshi is the same engine on a second venue, so it
                # freezes the same nlp config. Without this it would resolve to
                # a non-existent section and hash an EMPTY dict, making its
                # strategy_version constant and the parameter freeze a no-op.
                "nlp" if source in {"llm", "llm_kalshi", "news_speed"} else source
            )
            section = getattr(self.settings, section_name, None)
            section_payload = (section.model_dump(mode="json")
                               if hasattr(section, "model_dump") else {})
            section_payload = self._scope_config(source, section_payload)
            contract = {
                "strategy_source": source,
                "strategy_config": section_payload,
                "risk": {
                    "min_edge_pct": self.settings.risk.min_edge_pct,
                    "max_spread_pct": self.settings.risk.max_spread_pct,
                    "confidence_floor": self.settings.risk.confidence_floor,
                },
            }
            config_json = json.dumps(contract, sort_keys=True, separators=(",", ":"))
            strategy_version = hashlib.sha256(config_json.encode()).hexdigest()
            book = await self.db.fetchone(
                """SELECT best_bid,best_ask FROM orderbook_snapshots
                   WHERE market_id=? AND (?='' OR token_id=?)
                   ORDER BY recorded_at DESC LIMIT 1""",
                (order.market_id, order.token_id or "", order.token_id or ""),
            )
            if book is None or book["best_ask"] is None:
                book = await self._live_book(order) or book
            coefficient = 0.0 if order.post_only else taker_fee_rate(
                order.exchange or self.exchange_name, intent.market.category)
            fee = order.size * coefficient * order.price * (1.0 - order.price)
            family = intent.market.neg_risk_market_id or intent.market.id
            venue = order.exchange or self.exchange_name
            # Hand the book and the decision id to the order itself: the PAPER
            # trader needs the book to tell a marketable order from a resting
            # one, and a deferred fill needs the decision id to be attributed
            # back when it eventually trades through.
            if book is not None:
                order.best_bid = (None if book["best_bid"] is None
                                  else float(book["best_bid"]))
                order.best_ask = (None if book["best_ask"] is None
                                  else float(book["best_ask"]))
            return await DecisionTracker(self.db).capture(
                market_id=order.market_id, strategy_source=source,
                signal_id=getattr(intent.signal, "id", None), side=order.side.value,
                fair_probability=intent.signal.claude_prob,
                reference_price=intent.signal.market_prob,
                executable_price=order.price,
                best_bid=None if book is None else book["best_bid"],
                best_ask=None if book is None else book["best_ask"],
                requested_size=order.size * order.price, fee_estimate=fee,
                venue=venue, event_family=family, strategy_version=strategy_version,
                cohort_id=f"{venue}:{family}", config_json=config_json,
                holdout_warmup_days=self.settings.graduation.holdout_warmup_days,
                is_paper=order.dry_run,
            )
        except Exception as exc:
            log.warning("gateway.decision_capture_failed", error=str(exc),
                        market_id=order.market_id)

    async def submit_paired(
        self,
        a: TradeIntent,
        b: TradeIntent,
        *,
        exchange_a: ExchangeClient,
        exchange_name_a: str,
        exchange_b: ExchangeClient,
        exchange_name_b: str,
    ) -> tuple[ExecutionResult, ExecutionResult]:
        """Both-or-nothing paired execution for the arb pillars.

        Builds BOTH orders before placing EITHER (a leg that can't be built
        never leaves the other naked), places A then B, and cancels a
        live-pending A if B fails. ``record_fill`` + the trades-mirror are owned
        here; the caller keeps its signals / portfolio / verdict writes and any
        partial-fill bookkeeping. Legs may live on different exchanges
        (cross-venue), so each carries its own exchange + name. Both intents
        should share the same ``force_paper`` (the pillars paper-force the pair
        together).
        """
        is_live_a = self.settings.is_live and not a.force_paper
        is_live_b = self.settings.is_live and not b.force_paper
        order_a = await self._build_order(
            a.signal, a.market, a.size_dollars, is_live_a,
            exchange=exchange_a, router=None)
        order_b = await self._build_order(
            b.signal, b.market, b.size_dollars, is_live_b,
            exchange=exchange_b, router=None)
        if order_a is None or order_b is None:
            skip = ExecutionResult(status="skipped", reason="leg build failed")
            return skip, ExecutionResult(status="skipped", reason="leg build failed")

        cap_a = await self._exceeds_market_cap(order_a, is_live=is_live_a)
        cap_b = await self._exceeds_market_cap(order_b, is_live=is_live_b)
        if cap_a is not None or cap_b is not None:
            return (
                ExecutionResult(status="skipped", order=order_a,
                                reason=cap_a or "paired leg blocked by market cap"),
                ExecutionResult(status="skipped", order=order_b,
                                reason=cap_b or "paired leg blocked by market cap"),
            )

        # Keep the ids. Discarding them left order.decision_id unset, so the
        # mark_fill block in _place_and_record never ran and any paired-arb
        # snapshot this path wrote would stay filled=0 forever. With
        # graduation.require_executable_fills true (the tracked YAML),
        # _prospective_stats appends `AND d.filled = 1 AND d.fill_evidence IN
        # (...)`, so cross_venue_arb and entailment_arb could never graduate
        # paper->live on merit however long they ran.
        #
        # Scope of the damage so far, measured rather than assumed (live DB,
        # 2026-08-06): ZERO decision_snapshots rows and ZERO
        # strategy_experiments rows for either strategy. No holdout clock was
        # ever registered and no snapshot was ever written, so nothing was
        # corrupted and nothing was burned — the defect was a gate these two
        # strategies could not have passed had they started producing
        # evidence, not evidence already spoiled. Fixed before that mattered.
        # submit() at :136 plumbs these correctly; this path did not.
        decision_id_a = await self._capture_decision(a, order_a)
        decision_id_b = await self._capture_decision(b, order_b)
        order_a.decision_id = decision_id_a
        order_b.decision_id = decision_id_b

        res_a = await self._place_and_record(
            order_a, strategy_source=a.signal.strategy_source,
            signal_id=getattr(a.signal, "id", None),
            exchange=exchange_a, exchange_name=exchange_name_a,
            decision_id=decision_id_a)
        if res_a.status not in _OK_STATUSES:
            # Leg A rejected — B is never placed, nothing to unwind.
            return res_a, ExecutionResult(status="skipped", reason="leg_a_not_ok")

        res_b = await self._place_and_record(
            order_b, strategy_source=b.signal.strategy_source,
            signal_id=getattr(b.signal, "id", None),
            exchange=exchange_b, exchange_name=exchange_name_b,
            decision_id=decision_id_b)
        if res_b.status not in _OK_STATUSES:
            # Leg risk: A is in, B failed. Cancel a live-pending A so we don't
            # sit on a naked directional leg (paper / already-filled A can't be
            # cancelled — that's the genuine single-leg the caller logs).
            if (res_a.result is not None and res_a.result.status == "pending"
                    and not res_a.result.is_paper):
                # Through the gateway's own cancel contract: same best-effort
                # semantics as the hand-rolled try/except this replaced (the
                # unwind can fail and the pair still returns), plus the trades
                # mirror now goes terminal instead of lingering 'pending'.
                await self.cancel_resting(
                    res_a.result.order_id, exchange=exchange_a,
                    exchange_name=exchange_name_a,
                    is_paper=res_a.result.is_paper,
                    source=a.signal.strategy_source or "",
                    reason="paired_leg_a_unwind",
                )
        return res_a, res_b

    async def cancel_resting(
        self,
        order_id: str,
        *,
        exchange: ExchangeClient | None = None,
        exchange_name: str | None = None,
        is_paper: bool = False,
        source: str = "",
        reason: str = "",
    ) -> CancelResult:
        """Cancel a resting order. The gateway's cancel contract.

        CLAUDE.md rule 3 makes the gateway the single mechanical path for an
        order's lifecycle, but until this existed only PLACEMENT had a contract:
        every cancel in the codebase — including the market maker's three, and
        the gateway's own paired unwind — reached past the choke point straight
        to ``ExchangeClient.cancel_order``. The rule asserted an invariant its
        own enforcer violated.

        THIS IS A ROUTING AND AUDIT CONTRACT, NOT A GATE. It cannot return
        "blocked" and it has no condition that refuses. A placement adds
        exposure and is therefore gated fifteen ways by ``RiskManager``; a
        cancel REMOVES exposure, so every one of those checks is not merely
        unnecessary here but backwards.

        *** DO NOT ADD A KILL-SWITCH CHECK BELOW (noted 2026-08-06). ***
        This is the specific mistake to avoid, and it looks helpful. #408 made
        arming the kill switch RETIRE live exposure by cancelling resting
        orders (``AuramaurBot._cancel_resting_live_orders``). A cancel path the
        kill switch could block would therefore trap the operator inside the
        exposure the emergency stop exists to shed — the emergency stop would
        become an exposure lock. The kill switch belongs on ``place_order``,
        where it already is, and nowhere on this path.

        What the contract DOES provide:

        * **Paper interception.** A paper order never reaches a venue client.
          Quoting runs paper orders through the LIVE client object (paper-ness
          is per-order ``dry_run``, not a separate adapter), so this is a real
          boundary, not a formality.
        * **Terminal-state bookkeeping.** The trades mirror goes 'cancelled'
          rather than lingering 'pending' forever — the same repair
          ``_cancel_resting_live_orders`` and the order monitor's TTL sweep
          each hand-rolled (#94).
        * **A uniform result type** and one greppable audit event.
        * **Exception containment.** A venue error is REPORTED, never raised:
          these run inside a quoting loop, and callers already treated cancels
          as best-effort.

        ``exchange``/``exchange_name`` default to the gateway's own, exactly as
        ``submit`` resolves them; a cross-venue caller passes its own the way
        ``submit_exit`` and ``submit_paired`` do.
        """
        client = exchange if exchange is not None else self.exchange
        venue = exchange_name or self.exchange_name
        oid = str(order_id or "").strip()
        if not oid:
            return CancelResult(order_id="", status="skipped", reason="no order id")

        # Paper-ness is an OR, never an override: an explicit flag intercepts,
        # and a PAPER-shaped id intercepts on its own even if a caller forgot
        # (or wrongly computed) the flag. The failure mode being designed out —
        # a paper order id sent to the live venue — is not worth trusting one
        # argument for.
        if is_paper or oid.upper().startswith("PAPER"):
            await self._mark_cancelled(oid)
            log.debug("gateway.cancel", order_id=oid, exchange=venue,
                      status="paper", strategy_source=source, cancel_reason=reason)
            return CancelResult(order_id=oid, status="paper", reason="paper order")

        try:
            acknowledged = bool(await client.cancel_order(oid))
        except Exception as e:  # noqa: BLE001 — reported, never raised at a caller
            log.warning("gateway.cancel", order_id=oid, exchange=venue,
                        status="failed", strategy_source=source,
                        cancel_reason=reason, error=str(e))
            return CancelResult(order_id=oid, status="failed", reason=str(e))

        if not acknowledged:
            # The venue declined (already matched, unknown id). Deliberately no
            # terminal write: the order may still be live, and the order monitor
            # resolves it on the next status poll.
            log.warning("gateway.cancel", order_id=oid, exchange=venue,
                        status="rejected", strategy_source=source,
                        cancel_reason=reason)
            return CancelResult(order_id=oid, status="rejected",
                                reason="venue declined the cancel")

        await self._mark_cancelled(oid)
        log.info("gateway.cancel", order_id=oid, exchange=venue,
                 status="cancelled", strategy_source=source, cancel_reason=reason)
        return CancelResult(order_id=oid, status="cancelled")

    async def _mark_cancelled(self, order_id: str) -> None:
        """Drive the trades mirror to a terminal state for a cancelled order.

        Same repair as ``AuramaurBot._cancel_resting_live_orders`` and the order
        monitor's TTL sweep: without it the row stays 'pending' forever even
        though the collateral was released (#94). Scoped to 'pending' rows —
        those two callers only ever hand it resting-order ids, so the guard is a
        no-op for them, while here it makes a mis-supplied id incapable of
        rewriting a settled row.

        Best-effort by construction: this runs in a quoting loop, and a database
        fault must not surface as a cancel failure when the venue already
        acknowledged one.
        """
        try:
            await self._serialized_write(
                "UPDATE trades SET status = 'cancelled' "
                "WHERE order_id = ? AND status = 'pending'",
                (order_id,),
            )
        except Exception as e:  # noqa: BLE001 — bookkeeping, never the cancel
            log.debug("gateway.cancel_bookkeeping_failed",
                      order_id=order_id, error=str(e))

    async def _serialized_write(self, sql: str, params: tuple = ()) -> None:
        """Land gateway bookkeeping without bleeding across shared writers."""
        async with self.db.transaction(owner="execution_gateway"):
            await self.db.execute(sql, params)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _build_order(
        self, signal: Signal, market: Market, size_dollars: float, is_live: bool,
        *, exchange: ExchangeClient, router: SmartOrderRouter | None,
    ) -> Order | None:
        """Route (or prepare) an order. Returns ``None`` — having recorded the
        appropriate ``order_build_drops`` cooldown — when the signal is
        unmarketable at the book or the order can't be built.
        """
        try:
            if router:
                order = await router.route(signal, market, size_dollars, is_live)
            elif callable(getattr(type(exchange), "prepare_executable_order", None)):
                order = await exchange.prepare_executable_order(
                    signal, market, size_dollars, is_live)
            else:
                order = exchange.prepare_order(signal, market, size_dollars, is_live)
        except UnmarketableSignal as skip:
            # No realizable edge at the book. This gate runs before the
            # paper/live split on purpose: paper fills simulate at the
            # reference price, so letting paper books keep these trades
            # would graduate strategies on fills live could never get.
            show_order_dropped(market.id, f"unmarketable: {skip}")
            log.info(
                "engine.entry_unmarketable",
                market_id=market.id,
                strategy=signal.strategy_source,
                reason=str(skip),
            )
            try:
                # Shorter block than build failures: the book can move.
                await self._serialized_write(
                    """INSERT OR REPLACE INTO order_build_drops
                       (market_id, blocked_until, reason)
                       VALUES (?, datetime('now', '+30 minutes'), ?)""",
                    (market.id, f"unmarketable: {skip}"),
                )
            except Exception:
                pass
            return None

        if order is None:
            show_order_dropped(market.id, f"order build failed (${size_dollars:.2f} — could not build a valid order)")
            log.warning(
                "engine.order_dropped",
                market_id=market.id,
                size_dollars=size_dollars,
                reason="prepare_order returned None (bad price/token or router rejection)",
            )
            # Sub-minimum sizing is now bumped up in prepare_order, so a None
            # order here is a genuine build failure (bad price/token). Block
            # only briefly so a transient issue can retry next cycle.
            try:
                await self._serialized_write(
                    """INSERT OR REPLACE INTO order_build_drops
                       (market_id, blocked_until, reason)
                       VALUES (?, datetime('now', '+2 hours'), ?)""",
                    (market.id, f"order build failed at ${size_dollars:.2f}"),
                )
            except Exception:
                pass  # Table may not exist yet
            return None

        if not order.source:
            order.source = signal.strategy_source or "llm"
        return order

    async def _place_and_record(
        self, order: Order, *, strategy_source: str, signal_id,
        exchange: ExchangeClient, exchange_name: str, decision_id: int | None = None,
    ) -> ExecutionResult:
        """Place a built order, then log slippage, record the fill, and mirror
        to ``trades``. Shared by the single-leg, paired, and exit paths.
        """
        result = await exchange.place_order(order)
        show_order(result.status, result.order_id, order.side.value, order.size, order.price, result.is_paper, exchange=exchange_name, error_message=result.error_message, market_id=order.market_id)
        if decision_id is not None and booked_as_position(result):
            evidence = "venue_fill"
            if result.is_paper:
                snap = await self.db.fetchone(
                    "SELECT best_bid,best_ask FROM decision_snapshots WHERE id=?",
                    (decision_id,),
                )
                price = result.filled_price or order.price
                crossed = bool(snap) and (
                    (order.side == OrderSide.BUY and snap["best_ask"] is not None
                     and price >= float(snap["best_ask"]))
                    or (order.side == OrderSide.SELL and snap["best_bid"] is not None
                        and price <= float(snap["best_bid"]))
                )
                evidence = "book_cross" if crossed else "synthetic"
            await DecisionTracker(self.db).mark_fill(
                decision_id, filled_price=result.filled_price or order.price,
                evidence=evidence)
        return await self._record_result(
            order, result, strategy_source=strategy_source,
            signal_id=signal_id, exchange_name=exchange_name)

    async def record_external_fill(
        self, order: Order, result: OrderResult, *,
        strategy_source: str, exchange_name: str,
    ) -> ExecutionResult:
        """Record a fill for an order placed OUTSIDE the gateway.

        The arb scanner places its legs CONCURRENTLY (asyncio.gather) to minimize
        the leg-risk window, so it can't go through submit_*; this gives those
        already-placed legs the same recording invariant — slippage, record_fill
        (paper/filled), and the pending trades-mirror the order monitor later
        UPDATEs for live fills. No double-record: a live leg is pending at
        placement (filled_size 0), so its fill is recorded once, by the monitor.
        """
        return await self._record_result(
            order, result, strategy_source=strategy_source,
            signal_id=None, exchange_name=exchange_name)

    async def place_legs(
        self,
        legs: list[tuple[Order, ExchangeClient, str]],
        *,
        strategy_source: str,
        concurrent: bool = True,
        show: bool = False,
    ) -> list[tuple[OrderResult, ExecutionResult]]:
        """Place multiple already-built legs through the gateway, then record each.

        The single owned entry point for multi-leg flows that must place directly
        rather than via ``submit_paired``'s A-then-B atomicity: arb legs placed
        CONCURRENTLY (asyncio.gather, to minimize the leg-risk window) or
        SEQUENTIALLY for same-exchange pairs. Each placed leg is recorded with the
        same invariant as ``record_external_fill`` (slippage, record_fill, the
        pending trades-mirror the monitor finalizes for live fills). Returns the
        raw ``OrderResult`` alongside the ``ExecutionResult`` per leg so the caller
        keeps its own half-fill / rollback / inventory logic. ``legs`` items are
        ``(order, exchange_client, exchange_name)``.
        """
        # External multi-leg paths bypass ``submit`` but are still entries.
        # Check every BUY before any leg leaves the building. When two legs add
        # the same market/token in one batch, reserve earlier legs locally so
        # the batch cannot collectively exceed the cap.
        reservations: dict[tuple[str, str, int], float] = {}
        for order, _client, _exchange_name in legs:
            if order.side != OrderSide.BUY:
                continue
            is_live = not order.dry_run
            blocked = await self._exceeds_market_cap(order, is_live=is_live)
            key = (order.market_id, order.token.value, 0 if is_live else 1)
            proposed = order.size * order.price
            cap = getattr(self.settings.risk, "max_stake_abs_ceiling", 25.0)
            if blocked is None and reservations.get(key, 0.0) + proposed > cap + 1e-9:
                blocked = "market_cap: multi-leg batch exceeds aggregate ceiling"
            if blocked is not None:
                skipped = ExecutionResult(status="skipped", order=order, reason=blocked)
                rejected = OrderResult(order_id="MARKET_CAP", market_id=order.market_id,
                                       status="rejected", is_paper=order.dry_run,
                                       error_message=blocked)
                return [(rejected, skipped) for order, _client, _name in legs]
            reservations[key] = reservations.get(key, 0.0) + proposed

        if concurrent:
            results = await asyncio.gather(
                *(client.place_order(order) for order, client, _ in legs))
        else:
            results = [await client.place_order(order) for order, client, _ in legs]
        out: list[tuple[OrderResult, ExecutionResult]] = []
        for (order, _client, exchange_name), result in zip(legs, results):
            if show:
                show_order(result.status, result.order_id, order.side.value,
                           order.size, order.price, result.is_paper,
                           exchange=exchange_name,
                           error_message=result.error_message,
                           market_id=order.market_id)
            exec_res = await self.record_external_fill(
                order, result, strategy_source=strategy_source,
                exchange_name=exchange_name)
            out.append((result, exec_res))
        return out

    async def place_quote_pair(
        self, bid_order: Order, ask_order: Order, *,
        exchange: ExchangeClient,
    ) -> tuple[OrderResult, OrderResult]:
        """Place a two-sided maker quote (bid then ask) through the gateway.

        The market maker's owned placement entry point. The MM has already built
        fully-priced ``post_only`` orders (source stamped) and needs BOTH raw
        ``OrderResult``s back to run its own inventory / pending-order /
        partial-leg-cancel bookkeeping, so this places SEQUENTIALLY (matching the
        MM's order, NOT both-or-nothing — the MM owns one-legged cleanup) and
        returns the pair. Recording stays with the order monitor (orders carry
        ``source="market_maker"``); the gateway owns the placement so no strategy
        calls ``exchange.place_order`` directly.

        PAPER quotes never reach the exchange client: the paper branch of
        ``place_order`` fills every order INSTANTLY into the shared PaperTrader
        book, whose in-memory positions are keyed by market_id alone — so a
        two-sided quote's YES-bid and NO-ask legs merged into one blended
        ~mid-priced phantom position that regrew on every refresh, was never
        recorded to trades/fills (by design, anti-flooding), and was then
        persisted by position_sync as an untraceable orphan (2026-07-23: 7 rows,
        ~$700 phantom cost). A resting post-only quote must not fill at
        placement; until the MM gets a real paper fill simulation, paper quotes
        rest synthetically and expire without ever filling.
        """
        if bid_order.dry_run or ask_order.dry_run:
            def _resting(order: Order) -> OrderResult:
                return OrderResult(
                    order_id=f"PAPER-QUOTE-{uuid.uuid4().hex[:12]}",
                    market_id=order.market_id,
                    status="pending",
                    is_paper=True,
                )
            return _resting(bid_order), _resting(ask_order)
        bid_result = await exchange.place_order(bid_order)
        ask_result = await exchange.place_order(ask_order)
        return bid_result, ask_result

    async def _record_result(
        self, order: Order, result: OrderResult, *,
        strategy_source: str, signal_id, exchange_name: str,
    ) -> ExecutionResult:
        """Post-placement recording shared by _place_and_record (single-leg /
        paired / exit) and record_external_fill (the concurrently-placed arb
        legs): API-error cooldown, slippage, record_fill, and the trades-mirror.
        """
        # Cooldown on API errors — retry in 30 min, not every cycle
        if result.status == "rejected" and result.order_id == "ERROR":
            try:
                await self._serialized_write(
                    """INSERT OR REPLACE INTO order_build_drops
                       (market_id, blocked_until, reason)
                       VALUES (?, datetime('now', '+30 minutes'), ?)""",
                    (order.market_id, "place_order API error"),
                )
            except Exception:
                pass

        # Log slippage only for actual executions.  Live pending orders echo
        # the limit price but have not filled yet.
        if result.status in _FILLED_STATUSES and result.filled_price > 0:
            slippage_bps = (result.filled_price - order.price) / order.price * 10000
            if order.side == OrderSide.SELL:
                slippage_bps = -slippage_bps  # For sells, lower fill = worse
            try:
                await self._serialized_write(
                    """INSERT INTO slippage_log (market_id, exchange, side, expected_price, filled_price, slippage_bps, size, order_type)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (order.market_id, order.exchange or exchange_name, order.side.value,
                     order.price, result.filled_price, round(slippage_bps, 2), order.size,
                     order.order_type.value if hasattr(order, 'order_type') else 'limit'),
                )
            except Exception:
                pass

        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price

        # Record P&L only for actual executions.  Pending live orders are
        # mirrored to trades below, then finalized by the order monitor.
        recorded_fill: Fill | None = None
        if booked_as_position(result):
            fill = Fill(
                order_id=result.order_id,
                market_id=order.market_id,
                token_id=order.token_id,
                side=order.side,
                token=order.token,
                size=result.filled_size,
                price=fill_price,
                is_paper=result.is_paper,
            )
            if self.pnl_tracker:
                try:
                    await self.pnl_tracker.record_fill(fill)
                    recorded_fill = fill
                except Exception as e:
                    # The order already executed. Never turn a persistence fault
                    # into an apparent placement failure that callers may retry.
                    log.critical(
                        "gateway.fill_record_failed",
                        order_id=result.order_id,
                        market_id=order.market_id,
                        error=str(e),
                    )
            if recorded_fill is not None and result.is_paper:
                # record_fill maintains cost_basis only; nothing else maintains
                # paper portfolio rows in a live bot (position sync is
                # mode-scoped). Without this, an instant-fill paper EXIT leaves
                # its full-size portfolio row standing until resolution.
                try:
                    await materialize_paper_portfolio_row(self.db, order)
                except Exception as e:
                    # Contained like record_fill above — but at error, never
                    # debug: a swallow here recreates the stale row this call
                    # exists to prevent, and readiness must see it.
                    log.error(
                        "gateway.paper_portfolio_projection_failed",
                        order_id=result.order_id,
                        market_id=order.market_id,
                        error=str(e),
                    )

        if result.status in _OK_STATUSES:
            # Mirror into legacy `trades` table so the CLI stats view,
            # order monitor, and holding-period lookups stay in sync.
            # PnLTracker writes authoritative execution rows to `fills`.
            try:
                trade_status = "filled" if result.status == "paper" else result.status
                await self._serialized_write(
                    """INSERT INTO trades
                       (market_id, signal_id, side, size, price, is_paper,
                        order_id, status, kelly_fraction, exchange, strategy_source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        order.market_id,
                        signal_id,
                        order.side.value,
                        fill_size,
                        fill_price,
                        1 if result.is_paper else 0,
                        result.order_id,
                        trade_status,
                        None,
                        order.exchange or exchange_name,
                        strategy_source,
                    ),
                )
            except Exception as e:
                log.debug("engine.trade_mirror_error", error=str(e))

        return ExecutionResult(
            status=result.status, order=order, result=result, fill=recorded_fill,
            reason=result.error_message or "",
        )
