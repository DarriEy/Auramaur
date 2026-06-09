"""IBKR directional equity pillar — gated stocks/ETFs speculation.

Mirrors the Kraken directional pillar: momentum per symbol -> capped, budgeted
long entry/exit via IBKREquityClient. Off by default (kraken-style guardrails);
no validated edge. Every order passes the equity client's gates.
"""

from __future__ import annotations

import structlog

from auramaur.exchange.models import OrderSide

log = structlog.get_logger()


class IBKRDirectionalPillar:
    def __init__(self, settings, equity_client, console=None):
        self._s = settings
        self._eq = equity_client
        self._console = console
        # In-memory open longs: symbol -> entry price (experimental; off by default).
        self._long: dict[str, float] = {}

    async def run_once(self) -> None:
        cfg = self._s.ibkr
        if not cfg.directional_equity_enabled:
            return
        # Auto FX top-up: keep settled USD buying power for the USD-priced book by
        # converting idle base-currency cash (CAD->USD). Gated by auto_fx_enabled;
        # the conversion itself dry-runs unless the live gates are open. T+1 on a
        # cash account, so it funds a buffer for future cycles, not this one.
        if cfg.auto_fx_enabled:
            try:
                await self._eq.ensure_usd_float(
                    target_usd=cfg.fx_target_usd,
                    max_convert_usd=cfg.fx_max_convert_usd,
                    min_convert_usd=cfg.fx_min_convert_usd,
                    source_ccy=cfg.fx_source_currency,
                )
            except Exception as e:  # noqa: BLE001 — never kill the loop on FX
                log.error("ibkr_equity.fx_convert.pillar_error", error=str(e)[:120])
        # Reconcile open positions from the IBKR account so a restart doesn't lose
        # track (re-buy / miss exits). Empty when disconnected — safe.
        try:
            held = await self._eq.get_positions()
            for sym in cfg.directional_equity_symbols:
                if abs(held.get(sym, 0)) > 1e-6:
                    self._long.setdefault(sym, 0.0)   # discovered holding
            for sym in list(self._long):
                if abs(held.get(sym, 0)) <= 1e-6:
                    self._long.pop(sym, None)          # closed externally
        except Exception as e:  # noqa: BLE001
            log.debug("ibkr_equity.reconcile_error", error=str(e)[:80])
        for sym in cfg.directional_equity_symbols:
            try:
                await self._eval(sym)
            except Exception as e:  # noqa: BLE001 — never kill the loop
                log.error("ibkr_equity.directional.error", symbol=sym, error=str(e)[:120])

    async def _eval(self, sym: str) -> None:
        cfg = self._s.ibkr
        mom = await self._eq.momentum(sym)
        if mom is None:
            return
        holding = sym in self._long

        if not holding and mom >= cfg.directional_equity_momentum_pct:
            from auramaur.risk.tolerance import scale_budget, current_tolerance
            budget = scale_budget(cfg.directional_equity_budget_usd, current_tolerance(self._s))
            allocated = len(self._long) * cfg.equity_max_order_usd
            if allocated + cfg.equity_max_order_usd > budget:
                log.info("ibkr_equity.directional.budget_full", symbol=sym,
                         allocated=allocated, budget=round(budget, 2))
                return
            res = await self._eq.place_order(sym, OrderSide.BUY, cfg.equity_max_order_usd)
            if res.order_id not in ("ERROR", "BLOCKED"):
                self._long[sym] = res.filled_price or 0.0
            log.info("ibkr_equity.directional.entry", symbol=sym, momentum=round(mom, 2),
                     status=res.status, err=res.error_message[:80])

        elif holding and mom <= -cfg.directional_equity_momentum_pct:
            # Close the whole position (cashQty-valued sell). res is None when
            # nothing is actually held — drop it from the book either way unless
            # the close errored/was blocked.
            res = await self._eq.close_position(sym)
            if res is None or res.order_id not in ("ERROR", "BLOCKED"):
                self._long.pop(sym, None)
            log.info("ibkr_equity.directional.exit", symbol=sym, momentum=round(mom, 2),
                     status=(res.status if res else "noop"),
                     err=(res.error_message[:80] if res else ""))
