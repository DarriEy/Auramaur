"""Read-only multi-asset market-data adapter for local IBKR paper books.

There is deliberately no order method in this module.  A live TWS login may be
used as the quote source, but all execution belongs to Auramaur's local paper
simulator.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import math
import time
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import structlog

from auramaur.exchange.ibkr_instruments import ContractKind, IBKRBook, InstrumentSpec

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class MarketDataQuote:
    key: str
    bid: float
    ask: float
    timestamp: float
    con_id: int
    currency: str
    multiplier: float
    source: str = "ibkr"


class IBKRReadOnlyMarketData:
    """Resolve contracts and acquire BBO/history through a read-only socket."""

    def __init__(self, settings, *, client_id: int | None = None):
        self._settings = settings
        self._client_id = client_id
        self.readonly = True
        self._ib = None
        self._connected = False
        self._cooldown_until = 0.0
        self._connect_lock = asyncio.Lock()
        self._contracts: dict[str, tuple[Any, float | None]] = {}
        self._trading_hours: dict[int, tuple[str, str, float]] = {}

    async def _ensure_connected(self) -> None:
        if self._connected and self._ib is not None:
            return
        async with self._connect_lock:
            if self._connected and self._ib is not None:
                return
            if time.monotonic() < self._cooldown_until:
                raise ConnectionError("IBKR multi-asset connection on cooldown")
            from ib_async import IB

            cfg = self._settings.ibkr
            try:
                self._ib = IB()
                await self._ib.connectAsync(
                    host=cfg.host,
                    port=cfg.etf_quote_port,
                    clientId=(self._client_id if self._client_id is not None
                              else cfg.multiasset_client_id),
                    readonly=True,
                )
                self._ib.reqMarketDataType(cfg.market_data_type)
                self._connected = True
                log.info("ibkr_multiasset.connected", port=cfg.etf_quote_port,
                         client_id=(self._client_id if self._client_id is not None
                                    else cfg.multiasset_client_id), readonly=True)
            except Exception:
                self._ib = None
                self._cooldown_until = time.monotonic() + 300
                raise

    async def close(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()
        self._ib = None
        self._connected = False

    async def resolve(self, spec: InstrumentSpec):
        """Return one unambiguous, qualified contract for an instrument spec."""
        await self._ensure_connected()
        cached = self._contracts.get(spec.key)
        if cached is not None:
            contract, expires_at = cached
            if expires_at is None or time.monotonic() < expires_at:
                return contract
            self._contracts.pop(spec.key, None)

        if spec.kind is ContractKind.STOCK:
            contract = await self._stock(spec)
        elif spec.kind is ContractKind.FOREX:
            contract = await self._forex(spec)
        elif spec.kind is ContractKind.FUTURE:
            contract = await self._front_future(spec)
        elif spec.kind is ContractKind.OPTION:
            contract = await self._atm_option(spec)
        elif spec.kind is ContractKind.BOND:
            contract = await self._bond(spec)
        else:  # pragma: no cover - exhaustive enum guard
            raise ValueError(f"unsupported IBKR contract kind: {spec.kind}")
        dynamic = spec.kind in {ContractKind.FUTURE, ContractKind.OPTION, ContractKind.BOND}
        expires_at = (time.monotonic()
                      + self._settings.ibkr.multiasset_contract_cache_seconds
                      if dynamic else None)
        self._contracts[spec.key] = (contract, expires_at)
        return contract

    async def _qualify_one(self, contract):
        qualified = await self._ib.qualifyContractsAsync(contract)
        if len(qualified) != 1 or qualified[0] is None:
            raise LookupError(f"IBKR did not uniquely qualify {contract}")
        return qualified[0]

    async def _stock(self, spec: InstrumentSpec):
        from ib_async import Stock
        contract = Stock(spec.symbol, spec.exchange, spec.currency,
                         primaryExchange=spec.primary_exchange)
        return await self._qualify_one(contract)

    async def _forex(self, spec: InstrumentSpec):
        from ib_async import Forex
        return await self._qualify_one(Forex(spec.key, exchange=spec.exchange))

    async def _front_future(self, spec: InstrumentSpec):
        from ib_async import Future
        seed = Future(spec.symbol, exchange=spec.exchange, currency=spec.currency)
        details = await self._ib.reqContractDetailsAsync(seed)
        cutoff = date.today() + timedelta(days=7)
        candidates = []
        for detail in details or []:
            contract = detail.contract
            try:
                contract_multiplier = float(getattr(contract, "multiplier", 0) or 0)
            except (TypeError, ValueError):
                continue
            expected_multiplier = spec.contract_multiplier or spec.multiplier
            if contract_multiplier and not math.isclose(
                    contract_multiplier, expected_multiplier, rel_tol=0, abs_tol=1e-9):
                continue
            raw = str(getattr(contract, "lastTradeDateOrContractMonth", ""))[:8]
            try:
                expiry = datetime.strptime(raw, "%Y%m%d").date()
            except ValueError:
                try:
                    expiry = datetime.strptime(raw[:6], "%Y%m").date()
                except ValueError:
                    continue
            if expiry >= cutoff:
                candidates.append((expiry, contract))
        if not candidates:
            raise LookupError(f"no non-expiring front future found for {spec.key}")
        return await self._qualify_one(min(candidates, key=lambda item: item[0])[1])

    async def _underlying(self, symbol: str):
        from ib_async import Stock
        return await self._qualify_one(Stock(symbol, "SMART", "USD"))

    async def _atm_option(self, spec: InstrumentSpec):
        from ib_async import Option
        underlying = await self._underlying(spec.symbol)
        ticker = (await self._ib.reqTickersAsync(underlying))[0]
        spot = float(ticker.marketPrice())
        if not math.isfinite(spot) or spot <= 0:
            raise LookupError(f"no underlying price for {spec.key}")
        chains = await self._ib.reqSecDefOptParamsAsync(
            underlying.symbol, "", underlying.secType, underlying.conId)
        chain = next((c for c in chains
                      if c.exchange == "SMART"
                      and c.tradingClass == spec.symbol
                      and str(c.multiplier) == str(int(spec.multiplier))), None)
        if chain is None:
            chain = next((c for c in chains
                          if c.exchange == "SMART"
                          and str(c.multiplier) == str(int(spec.multiplier))), None)
        if chain is None and chains:
            chain = chains[0]
        if chain is None:
            raise LookupError(f"no option chain for {spec.key}")
        today = date.today()
        expiries = []
        for raw in chain.expirations:
            try:
                expiry = datetime.strptime(raw, "%Y%m%d").date()
            except ValueError:
                continue
            dte = (expiry - today).days
            if spec.option_dte_min <= dte <= spec.option_dte_max:
                expiries.append((dte, raw))
        if not expiries:
            raise LookupError(f"no option expiry in DTE window for {spec.key}")
        expiry = min(expiries, key=lambda item: item[0])[1]
        strikes = [float(s) for s in chain.strikes if float(s) > 0]
        if not strikes:
            raise LookupError(f"no strikes for {spec.key}")
        strike = min(strikes, key=lambda value: abs(value - spot))
        contract = Option(spec.symbol, expiry, strike, spec.option_right,
                          "SMART", multiplier=str(int(spec.multiplier)),
                          currency=spec.currency,
                          tradingClass=getattr(chain, "tradingClass", ""))
        return await self._qualify_one(contract)

    async def _bond(self, spec: InstrumentSpec):
        """Discover an exact bond contract from IBKR's US bond scanner."""
        from ib_async import ScannerSubscription, TagValue
        target_years = int(spec.bond_query.rsplit(":", 1)[-1].removesuffix("Y"))
        corporate = spec.bond_query.startswith("CORP:")
        target = date.today() + timedelta(days=round(target_years * 365.25))
        lower = (target - timedelta(days=183)).strftime("%Y%m%d")
        upper = (target + timedelta(days=183)).strftime("%Y%m%d")
        subscription = ScannerSubscription(
            instrument="BOND" if corporate else "BOND.GOVT",
            locationCode="BOND.US" if corporate else "BOND.GOVT.US",
            scanCode="LOW_BOND_ASK_YIELD_ALL" if corporate else "HIGH_BOND_ASK_YIELD_ALL",
            numberOfRows=50,
        )
        rows = await self._ib.reqScannerDataAsync(
            subscription,
            scannerSubscriptionFilterOptions=[
                TagValue("maturityDateAbove", lower),
                TagValue("maturityDateBelow", upper),
            ],
        )
        candidates = [row.contractDetails.contract for row in rows or []
                      if getattr(row.contractDetails.contract, "secType", "") == "BOND"]
        if not candidates:
            raise LookupError(f"bond scanner found no match for {spec.key}")
        # Scanner ordering provides the requested yield ranking; conId makes the
        # selected issue unambiguous even when descriptive fields are sparse.
        return await self._qualify_one(candidates[0])

    async def get_quote(self, spec: InstrumentSpec) -> MarketDataQuote | None:
        contract = await self.resolve(spec)
        return await self._quote_contract(spec, contract)

    async def get_quote_by_con_id(self, spec: InstrumentSpec, con_id: int):
        """Mark a held derivative using the exact contract originally filled."""
        contract = await self._contract_by_con_id(spec, con_id)
        return await self._quote_contract(spec, contract)

    async def _contract_by_con_id(self, spec: InstrumentSpec, con_id: int):
        from ib_async import Contract
        await self._ensure_connected()
        cache_key = f"conid:{con_id}"
        cached = self._contracts.get(cache_key)
        if cached is not None:
            return cached[0]
        contract = await self._qualify_one(Contract(conId=con_id, exchange=spec.exchange))
        self._contracts[cache_key] = (contract, None)
        return contract

    async def is_market_open(self, spec: InstrumentSpec, *, con_id: int = 0,
                             now: datetime | None = None) -> bool:
        """Use IBKR's dated liquid-hours schedule, including holidays and breaks."""
        contract = (await self._contract_by_con_id(spec, con_id)
                    if con_id else await self.resolve(spec))
        contract_id = int(getattr(contract, "conId", 0))
        cached = self._trading_hours.get(contract_id)
        if cached is None or time.monotonic() >= cached[2]:
            details = await self._ib.reqContractDetailsAsync(contract)
            if not details:
                return False
            detail = details[0]
            hours = str(getattr(detail, "liquidHours", "")
                        or getattr(detail, "tradingHours", ""))
            zone = str(getattr(detail, "timeZoneId", "") or "UTC")
            cached = (hours, zone, time.monotonic()
                      + self._settings.ibkr.multiasset_contract_cache_seconds)
            self._trading_hours[contract_id] = cached
        hours, zone_name, _ = cached
        try:
            zone = ZoneInfo(zone_name)
        except ZoneInfoNotFoundError:
            log.warning("ibkr_multiasset.unknown_timezone", key=spec.key, zone=zone_name)
            return False
        local_now = (now or datetime.now(timezone.utc)).astimezone(zone)
        for day in hours.split(";"):
            if not day or day.endswith(":CLOSED"):
                continue
            for session in day.split(","):
                start_text, separator, end_text = session.partition("-")
                if not separator:
                    continue
                try:
                    start = datetime.strptime(start_text, "%Y%m%d:%H%M").replace(tzinfo=zone)
                    end = datetime.strptime(end_text, "%Y%m%d:%H%M").replace(tzinfo=zone)
                except ValueError:
                    continue
                if start <= local_now < end:
                    return True
        return False

    async def _quote_contract(self, spec: InstrumentSpec, contract):
        try:
            ticker = (await self._ib.reqTickersAsync(contract))[0]
            bid, ask = float(ticker.bid or 0), float(ticker.ask or 0)
            if not all(math.isfinite(value) for value in (bid, ask)):
                return await self._synthetic_option_quote(spec, contract)
            if bid <= 0 or ask <= 0 or bid > ask:
                return await self._synthetic_option_quote(spec, contract)
            tick_time = getattr(ticker, "time", None)
            if tick_time is None:
                return await self._synthetic_option_quote(spec, contract)
            timestamp = tick_time.timestamp()
            if not math.isfinite(timestamp):
                return None
            multiplier = spec.multiplier
            return MarketDataQuote(spec.key, bid, ask, timestamp,
                                   int(getattr(contract, "conId", 0)),
                                   spec.currency, multiplier)
        except Exception as exc:  # noqa: BLE001
            if self._is_pacing_error(exc):
                raise
            log.warning("ibkr_multiasset.quote_error", key=spec.key,
                        error=str(exc)[:160])
            return await self._synthetic_option_quote(spec, contract)

    @staticmethod
    def _is_pacing_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(marker in text for marker in ("pacing", "error 162", "error 420"))

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return 0.5 * (1 + math.erf(value / math.sqrt(2)))

    @classmethod
    def _black_scholes(cls, spot: float, strike: float, years: float,
                       volatility: float, right: str) -> float:
        years = max(years, 1 / 365)
        volatility = max(volatility, 0.05)
        root_t = math.sqrt(years)
        rate = 0.04
        d1 = (math.log(spot / strike) + (rate + volatility ** 2 / 2) * years) / (
            volatility * root_t)
        d2 = d1 - volatility * root_t
        if right == "C":
            return spot * cls._normal_cdf(d1) - strike * math.exp(-rate * years) * cls._normal_cdf(d2)
        return strike * math.exp(-rate * years) * cls._normal_cdf(-d2) - spot * cls._normal_cdf(-d1)

    async def _underlying_history(self, symbol: str):
        underlying = await self._underlying(symbol)
        bars = await self._ib.reqHistoricalDataAsync(
            underlying, endDateTime="", durationStr="3 M", barSizeSetting="1 day",
            whatToShow="TRADES", useRTH=True)
        return underlying, bars or []

    async def _synthetic_option_quote(self, spec: InstrumentSpec, contract):
        if spec.kind is not ContractKind.OPTION:
            return None
        underlying, bars = await self._underlying_history(spec.symbol)
        ticker = (await self._ib.reqTickersAsync(underlying))[0]
        spot = float(ticker.marketPrice())
        closes = [float(bar.close) for bar in bars if float(bar.close or 0) > 0]
        if not math.isfinite(spot) or spot <= 0 or len(closes) < 21:
            return None
        returns = [math.log(b / a) for a, b in zip(closes, closes[1:])]
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / max(1, len(returns) - 1)
        volatility = math.sqrt(variance * 252)
        expiry = datetime.strptime(contract.lastTradeDateOrContractMonth[:8], "%Y%m%d").date()
        years = max((expiry - date.today()).days, 1) / 365
        fair = self._black_scholes(spot, float(contract.strike), years,
                                   volatility, spec.option_right)
        half_spread = max(0.01, fair * 0.01)
        bid, ask = max(0.01, fair - half_spread), fair + half_spread
        return MarketDataQuote(spec.key, bid, ask, time.time(),
                               int(contract.conId), spec.currency,
                               spec.multiplier, source="synthetic_option")

    async def get_daily_bars(self, spec: InstrumentSpec, duration: str = "3 M"):
        contract = await self.resolve(spec)
        return await self._daily_bars_contract(spec, contract, duration)

    async def get_daily_bars_by_con_id(self, spec: InstrumentSpec, con_id: int,
                                       duration: str = "3 M"):
        """History for the exact contract held, never a newly rolled discovery."""
        contract = await self._contract_by_con_id(spec, con_id)
        return await self._daily_bars_contract(spec, contract, duration)

    async def _daily_bars_contract(self, spec: InstrumentSpec, contract, duration: str):
        what = "MIDPOINT" if spec.kind in {ContractKind.FOREX, ContractKind.BOND} else "TRADES"
        bars = await self._ib.reqHistoricalDataAsync(
            contract, endDateTime="", durationStr=duration,
            barSizeSetting="1 day", whatToShow=what, useRTH=False)
        out = []
        for bar in bars or []:
            close = float(getattr(bar, "close", 0) or 0)
            raw_date = getattr(bar, "date", "")
            day = raw_date.isoformat() if hasattr(raw_date, "isoformat") else str(raw_date)
            if close > 0 and math.isfinite(close):
                out.append((day[:10], close))
        if not out and spec.kind is ContractKind.OPTION:
            _, underlying_bars = await self._underlying_history(spec.symbol)
            expiry = datetime.strptime(
                contract.lastTradeDateOrContractMonth[:8], "%Y%m%d").date()
            closes = [float(bar.close) for bar in underlying_bars
                      if float(bar.close or 0) > 0]
            returns = [math.log(b / a) for a, b in zip(closes, closes[1:])]
            if len(returns) >= 20:
                mean = sum(returns) / len(returns)
                variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
                volatility = math.sqrt(variance * 252)
                for bar in underlying_bars:
                    raw_date = getattr(bar, "date", "")
                    day_text = raw_date.isoformat() if hasattr(raw_date, "isoformat") else str(raw_date)
                    try:
                        bar_day = datetime.fromisoformat(day_text[:10]).date()
                    except ValueError:
                        continue
                    value = self._black_scholes(
                        float(bar.close), float(contract.strike),
                        max((expiry - bar_day).days, 1) / 365,
                        volatility, spec.option_right)
                    out.append((day_text[:10], value))
        return out

    async def get_fx_to_usd(self, currency: str) -> float | None:
        """Current USD value of one unit of *currency*."""
        if currency == "USD":
            return 1.0
        direct = f"{currency}USD"
        inverse = f"USD{currency}"
        pair, invert = (direct, False) if currency in {"EUR", "GBP", "AUD", "NZD"} else (inverse, True)
        spec = InstrumentSpec(pair, IBKRBook.FX, ContractKind.FOREX,
                              pair[:3], "IDEALPRO", pair[3:], "fx",
                              pair, multiplier=1000, calendar="FX_24X5")
        quote = await self.get_quote(spec)
        if quote is None:
            return None
        age = time.time() - quote.timestamp
        if age < 0 or age > self._settings.ibkr.multiasset_max_quote_age_seconds:
            return None
        mid = (quote.bid + quote.ask) / 2
        return 1 / mid if invert else mid

    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)
