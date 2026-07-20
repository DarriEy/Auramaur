"""Provisional-strategy task loops — extracted from AuramaurBot (Phase 5 split).

Pure structural move: the periodic loops that construct and drive each
provisional pillar (bias_harvest, entailment_arb, cross_venue_arb,
econ_indicator, intraday_drift, hydro_watch, weather_temp, resolution_lens,
oddlot_tender) live here as StrategyTaskMixin, mixed into AuramaurBot. Pillar
classes are imported lazily inside each loop (unchanged); behavior is identical
and the loops still operate on the bot's self (components, settings, kill switch).
"""

from __future__ import annotations

import asyncio

import structlog

log = structlog.get_logger()


class StrategyTaskMixin:
    """Provisional-strategy task loops for AuramaurBot (see module docstring)."""

    async def _task_bias_harvest(self) -> None:
        """Periodic favorite-longshot bias harvest scan (paper-forced by config)."""
        from auramaur.strategy.bias_harvest import BiasHarvestPillar

        pillar = BiasHarvestPillar(
            db=self._components.db,
            settings=self.settings,
            discovery=self._components.discovery,
            exchange=self._components.exchange,
            risk_manager=self._components.risk_manager,
            pnl_tracker=self._components.pnl_tracker,
            calibration=self._components.calibration,
        )
        interval = max(60, self.settings.bias_harvest.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await pillar.run_once()
            except Exception as e:
                log.error("bias_harvest.cycle_error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_long_horizon(self) -> None:
        """Periodic long-horizon favorite-underpricing scan (paper-forced).
        Runs the Polymarket instance and, when configured and composed, a
        second Kalshi instance (long_horizon_kalshi — its own ladder cell,
        politics_intl admitted; see the pillar docstring)."""
        from auramaur.strategy.long_horizon import LongHorizonPillar

        pillars = [LongHorizonPillar(
            db=self._components.db,
            settings=self.settings,
            discovery=self._components.discovery,
            exchange=self._components.exchange,
            risk_manager=self._components.risk_manager,
            pnl_tracker=self._components.pnl_tracker,
            calibration=self._components.calibration,
        )]
        if self.settings.long_horizon.kalshi_enabled:
            kalshi_discovery = (self._components.discoveries or {}).get("kalshi")
            kalshi_exchange = (self._components.exchanges or {}).get("kalshi")
            if kalshi_discovery is not None and kalshi_exchange is not None:
                pillars.append(LongHorizonPillar(
                    db=self._components.db,
                    settings=self.settings,
                    discovery=kalshi_discovery,
                    exchange=kalshi_exchange,
                    risk_manager=self._components.risk_manager,
                    pnl_tracker=self._components.pnl_tracker,
                    calibration=self._components.calibration,
                    venue="kalshi",
                ))
            else:
                log.info("long_horizon.kalshi_not_composed")
        interval = max(60, self.settings.long_horizon.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            for pillar in pillars:
                try:
                    await pillar.run_once()
                except Exception as e:
                    log.error("long_horizon.cycle_error", venue=pillar._venue,
                              error=str(e))
            await asyncio.sleep(interval)

    async def _task_agent_trader(self) -> None:
        """Periodic multi-model LLM day-trader (paper-forced; the
        intelligence-cap A/B — one attribution cell per model)."""
        from auramaur.strategy.agent_trader import AgentTraderPillar

        pillar = AgentTraderPillar(
            db=self._components.db,
            settings=self.settings,
            discovery=self._components.discovery,
            exchange=self._components.exchange,
            risk_manager=self._components.risk_manager,
            pnl_tracker=self._components.pnl_tracker,
            calibration=self._components.calibration,
        )
        interval = max(600, self.settings.agent_trader.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await pillar.run_once()
            except Exception as e:
                log.error("agent_trader.cycle_error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_term_structure(self) -> None:
        """Periodic deadline-ladder curve reader (paper-forced; one LLM read
        prices a whole family of strikes)."""
        from auramaur.strategy.term_structure import TermStructurePillar

        pillar = TermStructurePillar(
            db=self._components.db,
            settings=self.settings,
            discovery=self._components.discovery,
            exchange=self._components.exchange,
            risk_manager=self._components.risk_manager,
            pnl_tracker=self._components.pnl_tracker,
            calibration=self._components.calibration,
        )
        interval = max(600, self.settings.term_structure.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await pillar.run_once()
            except Exception as e:
                log.error("term_structure.cycle_error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_vol_anchor(self) -> None:
        """Periodic deterministic vol-anchored crypto threshold pricing
        (paper-forced; zero LLM cost)."""
        from auramaur.strategy.vol_anchor import VolAnchorPillar

        pillar = VolAnchorPillar(
            db=self._components.db,
            settings=self.settings,
            discovery=self._components.discovery,
            exchange=self._components.exchange,
            risk_manager=self._components.risk_manager,
            pnl_tracker=self._components.pnl_tracker,
            calibration=self._components.calibration,
        )
        interval = max(600, self.settings.vol_anchor.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await pillar.run_once()
            except Exception as e:
                log.error("vol_anchor.cycle_error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_informed_flow(self) -> None:
        """Periodic Kalshi informed-flow follower (abnormal-trade-size). Paper-
        forced; no-ops cleanly when the Kalshi venue isn't composed."""
        from auramaur.strategy.informed_flow_pillar import InformedFlowPillar

        kalshi_discovery = (self._components.discoveries or {}).get("kalshi")
        kalshi_exchange = (self._components.exchanges or {}).get("kalshi")
        if kalshi_discovery is None or kalshi_exchange is None:
            log.info("informed_flow.disabled", reason="kalshi venue not composed")
            return
        pillar = InformedFlowPillar(
            db=self._components.db,
            settings=self.settings,
            kalshi_discovery=kalshi_discovery,
            exchange=kalshi_exchange,
            risk_manager=self._components.risk_manager,
            pnl_tracker=self._components.pnl_tracker,
            calibration=self._components.calibration,
        )
        interval = max(60, self.settings.informed_flow.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await pillar.run_once()
            except Exception as e:
                log.error("informed_flow.cycle_error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_entailment_arb(self) -> None:
        """Periodic entailment-arbitrage scan (paper-forced by config)."""
        from auramaur.strategy.entailment_arb import EntailmentArbPillar

        pillar = EntailmentArbPillar(
            db=self._components.db,
            settings=self.settings,
            discovery=self._components.discovery,
            exchange=(self._components.exchanges or {}).get("polymarket"),
            risk_manager=self._components.risk_manager,
            pnl_tracker=self._components.pnl_tracker,
            analyzer=self._components.analyzer,
            # Kalshi econ-bin ladder arb fetches its own series; pass the Kalshi
            # discovery if present (no-op when Kalshi isn't configured).
            kalshi_discovery=(self._components.discoveries or {}).get("kalshi"),
            exchanges=self._components.exchanges or {},
        )
        interval = max(60, self.settings.entailment_arb.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await pillar.run_once()
            except Exception as e:
                log.error("entailment.cycle_error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_cross_venue_arb(self) -> None:
        """Periodic cross-venue (Poly×Kalshi) semantic-equivalence arb scan.
        Paper-forced by config; no-op unless Kalshi discovery is wired."""
        from auramaur.strategy.cross_venue_arb import CrossVenueArbPillar

        pillar = CrossVenueArbPillar(
            db=self._components.db,
            settings=self.settings,
            discovery=(self._components.discoveries or {}).get("polymarket"),
            exchange=(self._components.exchanges or {}).get("polymarket"),
            risk_manager=self._components.risk_manager,
            pnl_tracker=self._components.pnl_tracker,
            analyzer=self._components.analyzer,
            kalshi_discovery=(self._components.discoveries or {}).get("kalshi"),
            exchanges=self._components.exchanges or {},
        )
        interval = max(60, self.settings.cross_venue_arb.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await pillar.run_once()
            except Exception as e:
                log.error("cross_venue.cycle_error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_econ_indicator(self) -> None:
        """Periodic data-driven Kalshi econ-indicator bin pricing (paper-forced)."""
        from auramaur.data_sources.fred import FREDSource
        from auramaur.strategy.econ_indicator import EconIndicatorPillar

        kalshi_discovery = (self._components.discoveries or {}).get("kalshi")
        kalshi_exchange = (self._components.exchanges or {}).get("kalshi")
        if kalshi_discovery is None or kalshi_exchange is None or not self.settings.fred_api_key:
            log.info("econ_indicator.disabled", reason="missing kalshi or FRED key")
            return
        pillar = EconIndicatorPillar(
            db=self._components.db,
            settings=self.settings,
            kalshi_discovery=kalshi_discovery,
            fred_source=FREDSource(api_key=self.settings.fred_api_key),
            exchange=kalshi_exchange,
            risk_manager=self._components.risk_manager,
            pnl_tracker=self._components.pnl_tracker,
            calibration=self._components.calibration,
        )
        interval = max(60, self.settings.econ_indicator.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await pillar.run_once()
            except Exception as e:
                log.error("econ_indicator.cycle_error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_interim_manager(self) -> None:
        """Operator-proposed interim entries (docs/INTERIM_MANAGER.md)."""
        from auramaur.strategy.interim_manager import InterimManagerPillar

        pillar = InterimManagerPillar(
            db=self._components.db,
            settings=self.settings,
            discoveries=self._components.discoveries,
            exchanges=self._components.exchanges,
            risk_manager=self._components.risk_manager,
            pnl_tracker=self._components.pnl_tracker,
            calibration=self._components.calibration,
        )
        interval = max(60, self.settings.interim_manager.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await pillar.run_once()
            except Exception as e:
                log.error("interim_manager.cycle_error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_settlement_arb(self) -> None:
        """Settlement-lag / known-outcome arb over Polymarket econ markets,
        resolved deterministically against FRED (paper-forced, default off)."""
        from auramaur.data_sources.fred import FREDSource
        from auramaur.strategy.settlement_arb import SettlementArbPillar

        if not self.settings.fred_api_key:
            log.info("settlement_arb.disabled", reason="missing FRED key")
            return
        pillar = SettlementArbPillar(
            db=self._components.db,
            settings=self.settings,
            discovery=self._components.discovery,
            exchange=self._components.exchange,
            risk_manager=self._components.risk_manager,
            pnl_tracker=self._components.pnl_tracker,
            fred_source=FREDSource(api_key=self.settings.fred_api_key),
            analyzer=self._components.analyzer,
            # Kalshi monthly macro bins are where the settlement lag lives; pass
            # the Kalshi venue when composed so the pillar scans + trades them.
            kalshi_discovery=(self._components.discoveries or {}).get("kalshi"),
            kalshi_exchange=(self._components.exchanges or {}).get("kalshi"),
        )
        interval = max(60, self.settings.settlement_arb.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await pillar.run_once()
            except Exception as e:
                log.error("settlement_arb.cycle_error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_intraday_drift(self) -> None:
        """Measurement spike: track post-signal price drift toward the LLM estimate
        (no trading). Gates the intraday-convergence strategy on real evidence."""
        from auramaur.monitoring.intraday_drift import IntradayDriftTracker

        tracker = IntradayDriftTracker(
            db=self._components.db,
            settings=self.settings,
            discovery=self._components.discovery,
        )
        interval = max(60, self.settings.intraday_drift.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await tracker.run_once()
            except Exception as e:
                log.error("intraday_drift.cycle_error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_hydro_watch(self) -> None:
        """Alert when a tradeable hydrology market appears (compHydro moat armed)."""
        from auramaur.monitoring.hydro_market_watch import HydroMarketWatcher

        watcher = HydroMarketWatcher(
            db=self._components.db,
            settings=self.settings,
            discoveries=self._components.discoveries or {},
            alerts=self._components.alerts,
        )
        interval = max(300, self.settings.hydro_watch.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await watcher.run_once()
            except Exception as e:
                log.error("hydro_watch.cycle_error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_weather_temp(self) -> None:
        """Open-Meteo ensemble pricing of Polymarket city-temperature bins (paper)."""
        from auramaur.data_sources.openmeteo import OpenMeteoSource
        from auramaur.strategy.weather_temp import WeatherTempPillar

        pillar = WeatherTempPillar(
            db=self._components.db,
            settings=self.settings,
            discovery=self._components.discovery,
            exchange=self._components.exchange,
            risk_manager=self._components.risk_manager,
            pnl_tracker=self._components.pnl_tracker,
            calibration=self._components.calibration,
            weather=OpenMeteoSource(),
        )
        interval = max(60, self.settings.weather_temp.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await pillar.run_once()
            except Exception as e:
                log.error("weather_temp.cycle_error", error=str(e))
            await asyncio.sleep(interval)

    async def _task_resolution_lens(self) -> None:
        """Periodic resolution-language lens scan (paper-forced by config)."""
        from auramaur.strategy.resolution_lens import ResolutionLensPillar

        pillar = ResolutionLensPillar(
            db=self._components.db,
            settings=self.settings,
            discovery=self._components.discovery,
            exchange=self._components.exchange,
            risk_manager=self._components.risk_manager,
            pnl_tracker=self._components.pnl_tracker,
            calibration=self._components.calibration,
            analyzer=self._components.analyzer,
            # Phase 3: evidence-grounded comprehension taps the shared aggregator.
            aggregator=self._components.aggregator,
        )
        interval = max(60, self.settings.resolution_lens.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await pillar.run_once()
            except Exception as e:
                log.error("lens.cycle_error", error=str(e), exc_info=True)
            await asyncio.sleep(interval)

    async def _task_resolution_lens_kalshi(self) -> None:
        """Kalshi resolution-lens measurement spike (paper-forced, default off).

        A SECOND lens instance bound to the Kalshi discovery/exchange, attributed
        to 'resolution_lens_kalshi' so it earns its own graduation cells and
        cannot dilute the proven Polymarket lens. Tests whether Kalshi's
        CFTC-legalistic resolution criteria carry fine-print mispricing. No-ops
        cleanly if the Kalshi venue isn't composed.
        """
        from auramaur.strategy.resolution_lens import ResolutionLensPillar

        discovery = self._components.discoveries.get("kalshi")
        exchange = self._components.exchanges.get("kalshi")
        if discovery is None or exchange is None:
            log.info("lens.kalshi_spike_skipped", reason="kalshi venue not composed")
            return

        pillar = ResolutionLensPillar(
            db=self._components.db,
            settings=self.settings,
            discovery=discovery,
            exchange=exchange,
            risk_manager=self._components.risk_manager,
            pnl_tracker=self._components.pnl_tracker,
            calibration=self._components.calibration,
            analyzer=self._components.analyzer,
            aggregator=self._components.aggregator,
            exchange_name="kalshi",
            source_tag="resolution_lens_kalshi",
        )
        interval = max(60, self.settings.resolution_lens.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await pillar.run_once()
            except Exception as e:
                # exc_info: 'database is locked' with no stack burned an
                # evening of diagnosis — always capture where a cycle died.
                log.error("lens.kalshi_cycle_error", error=str(e), exc_info=True)
            await asyncio.sleep(interval)

    async def _task_oddlot_tender(self) -> None:
        """EDGAR odd-lot tender scan (detection always; entries paper-forced)."""
        from auramaur.data_sources.edgar import EdgarClient
        from auramaur.strategy.oddlot_tender import OddLotTenderPillar

        edgar = EdgarClient()
        equity = None
        if self.settings.ibkr.enabled:
            from auramaur.exchange.ibkr_equity import IBKREquityClient
            equity = IBKREquityClient(self.settings)
        pillar = OddLotTenderPillar(
            db=self._components.db,
            settings=self.settings,
            edgar=edgar,
            analyzer=self._components.analyzer,
            alerts=self._components.alerts,
            equity_client=equity,
            pnl_tracker=self._components.pnl_tracker,
        )
        interval = max(600, self.settings.oddlot_tender.interval_seconds)
        try:
            while self._running:
                if await self._check_kill_switch():
                    return
                try:
                    await pillar.run_once()
                except Exception as e:
                    log.error("oddlot.cycle_error", error=str(e))
                await asyncio.sleep(interval)
        finally:
            await edgar.close()
            if equity is not None:
                await equity.close()

    async def _task_platform_consensus(self) -> None:
        """Periodic platform consensus follower (Metaculus / Manifold)."""
        from auramaur.strategy.platform_consensus import PlatformConsensusPillar

        pillar = PlatformConsensusPillar(
            db=self._components.db,
            settings=self.settings,
            discovery=self._components.discovery,
            exchange=self._components.exchange,
            risk_manager=self._components.risk_manager,
            pnl_tracker=self._components.pnl_tracker,
            calibration=self._components.calibration,
        )
        interval = max(60, self.settings.platform_consensus.interval_seconds)
        while self._running:
            if await self._check_kill_switch():
                return
            try:
                await pillar.run_once()
            except Exception as e:
                log.error("platform_consensus.cycle_error", error=str(e))
            await asyncio.sleep(interval)

