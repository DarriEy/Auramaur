"""Smart order router — decides limit vs market and computes optimal price."""

from __future__ import annotations

import math

import structlog

from auramaur.exchange.models import (
    Market,
    Order,
    OrderBook,
    OrderSide,
    OrderType,
    Signal,
    TokenType,
)

log = structlog.get_logger()


class UnmarketableSignal(Exception):
    """The signal cannot be executed at a price that clears the minimum-edge
    floor — there is no realizable edge at the book, so the trade is skipped.

    Edge is measured against a reference price, but the only price a taker
    can actually get is the ask. Entries that can't cross within budget used
    to post passively at bid+1 tick: those filled ~25-29% of the time before
    the TTL reaper killed them, and the fills were adversely selected (a
    resting bid fills when flow moves against the thesis). Skipping is the
    honest outcome — the edge was never realizable.
    """


class SmartOrderRouter:
    """Builds optimally-priced orders by inspecting the order book.

    The router delegates to ``exchange.prepare_order()`` for the base order
    (which handles Polymarket's always-BUY / token-swap semantics), then
    refines the price and order type based on current book conditions.
    """

    def __init__(self, settings, exchange):
        self._settings = settings
        self._exchange = exchange

    async def route(
        self,
        signal: Signal,
        market: Market,
        size_dollars: float,
        is_live: bool,
    ) -> Order | None:
        """Build an optimally-priced order.

        1. Call ``exchange.prepare_order()`` to get the base order (handles
           Polymarket token swap logic).
        2. Fetch the order book for the relevant token.
        3. BUY entries are taker-or-skip: price at the best ask when the
           realizable edge (edge minus cross cost) clears the minimum-edge
           floor and the ask is within ``entry_max_cross_cents`` of the
           reference price (the cents cap is waived above 40% edge, but
           only when the fair value for the bought token also clears the
           ask by the minimum-edge floor in absolute points); otherwise
           raise :class:`UnmarketableSignal`. The token count is re-derived
           from the dollar intent at the crossed price so the notional
           can't inflate when the ask sits above the reference.
        4. SELL orders keep the passive path (limit one tick inside the
           book, market-order fallbacks) — exits must get out, and "no
           fill" is not an acceptable outcome for them.

        Raises ``UnmarketableSignal`` when a BUY entry has no realizable
        edge at the book. Returns None when no base order could be built.
        """
        base_order = self._exchange.prepare_order(
            signal=signal,
            market=market,
            position_size=size_dollars,
            is_live=is_live,
        )
        if base_order is None:
            log.info(
                "router.no_base_order",
                market_id=market.id,
                size_dollars=size_dollars,
            )
            return None

        # Fetch live order book for the token we're trading
        try:
            book = await self._exchange.get_order_book(base_order.token_id)
        except Exception:
            book = OrderBook()  # Empty book — fall back to market order

        edge_pct = abs(signal.edge)
        spread = book.spread

        # Crossing budget: how far above the signal's reference price a BUY may
        # lift the ask and still clear the minimum-edge floor. The CLOB has no
        # true market orders — every order is a GTC limit — so a "market" order
        # priced at the (stale) reference price just rests until the TTL
        # reaper kills it. Marketable means priced at the actual best ask.
        try:
            min_edge_pct = float(self._settings.risk.min_edge_pct)
        except Exception:
            min_edge_pct = 2.5
        try:
            max_cross = float(self._settings.execution.entry_max_cross_cents) / 100.0
        except Exception:
            max_cross = 0.04

        # Determine order type.
        # BUY entries are taker-or-skip. Passive entries (bid+1 tick) filled
        # 25-29% of the time before the TTL reaper killed them, and a resting
        # bid only fills when flow moves against the thesis — the survivors
        # are adversely selected. The realizable price for an entry is the
        # ask: if the edge measured THERE doesn't clear the floor, the signal
        # has no executable edge and the trade is skipped (recorded upstream
        # as a rejection, with cooldown). SELLs keep the passive path — exits
        # must get out, so "no fill" is not an acceptable outcome for them.
        has_book = spread is not None and (book.best_bid is not None or book.best_ask is not None)
        crossed_to_ask = False

        if base_order.side == OrderSide.BUY:
            if book.best_ask is None:
                # Dead or one-sided book: nothing to take, and a resting bid
                # on a market with no sellers is a pure coin flip.
                raise UnmarketableSignal(
                    f"no asks on the book for {market.id} — dead or one-sided market"
                )
            ask = book.best_ask
            # Signal.edge is documented as ABSOLUTE points (see the model),
            # so subtracting crossing costs in points below is sound — but a
            # producer that slips back to a relative figure would inflate the
            # budget exactly on cheap markets where thin books park asks at
            # multiples of fair value. Clamp the edge basis to the gap the
            # signal's own fair value implies for the token being bought; for
            # conforming producers the fair-implied gap is >= the (fee-
            # adjusted) edge, so this is a no-op.
            fair_token = (
                signal.claude_prob
                if base_order.token == TokenType.YES
                else 1.0 - signal.claude_prob
            )
            edge_pct = min(
                edge_pct, max(0.0, (fair_token - base_order.price) * 100.0)
            )
            cross_cost_pts = (ask - base_order.price) * 100.0
            realizable_edge = edge_pct - cross_cost_pts
            if realizable_edge < min_edge_pct:
                raise UnmarketableSignal(
                    f"edge at the ask {realizable_edge:.2f}% below minimum "
                    f"{min_edge_pct:.2f}% (ref {base_order.price:.2f}, ask {ask:.2f})"
                )
            # Cents cap: never chase more than entry_max_cross_cents above
            # the reference price. Very high edges (>40 points) may waive the
            # cap, but only when fair value for the token being bought also
            # clears the ASK itself by the minimum-edge floor.
            waive_cents_cap = (
                edge_pct > 40.0
                and (fair_token - ask) * 100.0 >= min_edge_pct
            )
            if not waive_cents_cap and (ask - base_order.price) > max_cross + 1e-9:
                raise UnmarketableSignal(
                    f"ask {ask:.2f} is {round((ask - base_order.price) * 100)}c above "
                    f"reference {base_order.price:.2f} (cap {round(max_cross * 100)}c)"
                )
            crossed_to_ask = True
            order_type = OrderType.LIMIT
            limit_price = max(0.01, min(0.99, round(ask, 2)))

            # The token count was derived at the REFERENCE price; the fill
            # happens at limit_price. Re-derive it from the original dollar
            # intent so the notional can't silently inflate by
            # limit/reference when crossing up (shrink-only: a cheaper fill
            # keeps the token count and simply spends less). Applied again
            # after depth-aware sweep pricing, which can lift the limit.
            intended_dollars = base_order.size * base_order.price

            def _rederived_size(current: float, price: float) -> float:
                if price <= base_order.price + 1e-9 or price <= 0:
                    return current
                resized = round(intended_dollars / price, 2)
                # Venue minimums (5 tokens / $1 notional), same floor as
                # prepare_order; the bump never exceeds the pre-cross size.
                resized = max(resized, max(5.0, math.ceil(100.0 / price) / 100))
                return min(current, resized)

            new_size = _rederived_size(base_order.size, limit_price)
            if new_size < base_order.size - 1e-9:
                log.info(
                    "router.size_rederived_at_cross",
                    market_id=market.id,
                    reference=base_order.price,
                    limit_price=limit_price,
                    original_size=round(base_order.size, 2),
                    resized=round(new_size, 2),
                )
                base_order = base_order.model_copy(update={"size": new_size})

            # Depth-aware sizing: the gate above only proved the BEST ask clears
            # the floor. For the FULL size we walk the asks up to the slippage
            # budget — the price at which realizable edge would fall to the
            # min-edge floor, also bounded by the cents cap — and trim the order
            # to the in-budget depth (a fraction of it, so we're not the whole
            # book). If too little of the requested size fits within budget, the
            # rest would only fill by paying through the floor, so skip.
            try:
                depth_aware = bool(self._settings.execution.depth_aware_routing)
            except Exception:
                depth_aware = False
            if depth_aware and base_order.size > 0:
                cap_frac = float(getattr(self._settings.execution, "book_capacity_fraction", 0.5))
                min_fill = float(getattr(self._settings.execution, "min_fill_fraction", 0.5))
                # Highest price that still leaves >= min_edge after slippage.
                edge_budget_pts = max(0.0, edge_pct - min_edge_pct)
                price_cap = base_order.price + edge_budget_pts / 100.0
                # Honor the cents cap too (same waiver as above).
                if not waive_cents_cap:
                    price_cap = min(price_cap, base_order.price + max_cross)
                price_cap = max(price_cap, ask)  # always at least take the best ask
                price_cap = min(0.99, price_cap)

                in_budget_depth = book.depth_within(price_cap, is_buy=True)
                capacity = in_budget_depth * cap_frac
                executable = min(base_order.size, capacity)
                if executable < min_fill * base_order.size - 1e-9:
                    raise UnmarketableSignal(
                        f"book absorbs only {executable:.1f} of {base_order.size:.1f} "
                        f"shares within slippage budget (cap {price_cap:.2f}, "
                        f"min fill {min_fill:.0%}) for {market.id}"
                    )
                _, vwap, sweep_price = book.fill_to_size(executable, is_buy=True)
                if executable < base_order.size - 1e-9:
                    log.info(
                        "router.size_trimmed_to_capacity",
                        market_id=market.id,
                        requested=round(base_order.size, 1),
                        executable=round(executable, 1),
                        in_budget_depth=round(in_budget_depth, 1),
                        price_cap=round(price_cap, 3),
                    )
                    base_order = base_order.model_copy(update={"size": executable})
                # Price the limit at the SWEEP price — the worst level the size
                # actually needs — not the full budget cap: a size that fits in
                # the best ask still prices at the ask, and only a multi-level
                # sweep lifts the limit (always <= price_cap). Cheapest-first
                # matching means the effective fill is the VWAP (<= the limit).
                if sweep_price > 0:
                    limit_price = max(0.01, min(0.99, round(sweep_price, 2)))
                    # Sweep pricing can lift the limit past the best ask —
                    # re-derive the size at the price we will actually pay.
                    resized = _rederived_size(base_order.size, limit_price)
                    if resized < base_order.size - 1e-9:
                        base_order = base_order.model_copy(update={"size": resized})
                cross_cost_pts = (vwap - base_order.price) * 100.0 if vwap else cross_cost_pts
                realizable_edge = edge_pct - cross_cost_pts

            log.info(
                "router.crossed_to_ask",
                market_id=market.id,
                edge_pct=round(edge_pct, 2),
                realizable_edge=round(realizable_edge, 2),
                ask=ask,
                limit_price=limit_price,
                size=round(base_order.size, 1),
                cross_cost_pts=round(cross_cost_pts, 2),
            )
        elif not has_book:
            # SELL with no book data — fall back to market order
            order_type = OrderType.MARKET
            limit_price = base_order.price
            log.info(
                "router.market_order",
                market_id=market.id,
                edge_pct=round(edge_pct, 2),
                reason="no_book",
            )
        elif edge_pct > 40.0:
            # Very high urgency SELL — cross immediately at the reference.
            order_type = OrderType.MARKET
            limit_price = base_order.price
            log.info(
                "router.market_order",
                market_id=market.id,
                edge_pct=round(edge_pct, 2),
                limit_price=limit_price,
                reason="high_urgency",
            )
        else:
            # SELL limit order — price one tick inside the book
            candidate_price = self._compute_limit_price(
                book, base_order.side, base_order.token
            )

            # Sanity check: limit price shouldn't deviate >30% from fair price
            fair_price = base_order.price
            if fair_price > 0 and abs(candidate_price - fair_price) / fair_price > 0.30:
                # Book too thin — use market order at fair price
                order_type = OrderType.MARKET
                limit_price = fair_price
                log.info(
                    "router.market_order",
                    market_id=market.id,
                    edge_pct=round(edge_pct, 2),
                    reason="thin_book",
                    candidate=candidate_price,
                    fair=fair_price,
                )
            else:
                order_type = OrderType.LIMIT
                limit_price = candidate_price
                reason = "tight_spread" if (spread is not None and spread < 0.01) else "price_improvement"
                log.info(
                    "router.limit_order",
                    market_id=market.id,
                    spread=round(spread, 4) if spread else None,
                    edge_pct=round(edge_pct, 2),
                    limit_price=limit_price,
                    reason=reason,
                )

        # Post-only on LIMIT orders: we use `best_bid + 1 tick` / `best_ask - 1
        # tick` as the price, which shouldn't cross — but the book can move in
        # the gap between fetch and submit. post_only guarantees we don't
        # accidentally take, which keeps us on the maker-reward side of the
        # fee schedule and flushes out any router/pricing mistakes loudly
        # (rejection) instead of quietly (taker fill at a worse price).
        # A deliberate cross-to-ask is the opposite intent: it must be allowed
        # to take, so it never sets post_only.
        want_post_only = order_type == OrderType.LIMIT and not crossed_to_ask

        # Build the final order with refined price / type
        routed_order = base_order.model_copy(
            update={
                "price": limit_price,
                "order_type": order_type,
                "post_only": want_post_only,
            }
        )

        log.info(
            "router.order_ready",
            market_id=market.id,
            token=routed_order.token.value,
            side=routed_order.side.value,
            order_type=routed_order.order_type.value,
            price=routed_order.price,
            size=routed_order.size,
        )

        return routed_order

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_limit_price(
        book: OrderBook,
        side: OrderSide,
        token: TokenType,
    ) -> float:
        """Compute a limit price one tick inside the book.

        For BUY orders:  best_bid + 0.01 (step ahead of resting bids).
        For SELL orders: best_ask - 0.01 (step ahead of resting asks).

        When 1-tick improvement would cross (1-cent spread markets), fall
        back to joining the BBO instead — post_only would otherwise reject.

        Result is clamped to the valid Polymarket tick range [0.01, 0.99].
        """
        if side == OrderSide.BUY:
            if book.best_bid is not None and book.best_ask is not None:
                improved = book.best_bid + 0.01
                # If improvement would touch or cross best_ask, join the bid
                # queue instead (still a maker post, at the same price as
                # everyone resting at best_bid).
                price = book.best_bid if improved >= book.best_ask else improved
            elif book.best_bid is not None:
                price = book.best_bid + 0.01
            elif book.best_ask is not None:
                price = book.best_ask - 0.01
            else:
                price = 0.50
        else:
            # SELL side
            if book.best_ask is not None and book.best_bid is not None:
                improved = book.best_ask - 0.01
                price = book.best_ask if improved <= book.best_bid else improved
            elif book.best_ask is not None:
                price = book.best_ask - 0.01
            elif book.best_bid is not None:
                price = book.best_bid + 0.01
            else:
                price = 0.50

        # Clamp to valid Polymarket price range and round to tick
        price = max(0.01, min(0.99, round(price, 2)))
        return price
