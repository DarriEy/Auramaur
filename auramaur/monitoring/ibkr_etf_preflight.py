"""Operational readiness checks for the paper-only IBKR ETF experiment."""

from __future__ import annotations

from dataclasses import dataclass
import time
from urllib.parse import quote

import aiohttp

from auramaur.exchange.ibkr_equity import IBKREquityClient


@dataclass(frozen=True)
class ETFPreflightResult:
    name: str
    severity: str
    detail: str


@dataclass(frozen=True)
class ETFPreflightReport:
    results: tuple[ETFPreflightResult, ...]

    @property
    def ready(self) -> bool:
        return not any(result.severity == "BLOCK" for result in self.results)


async def _check_openai_models(api_key: str, models: list[str], timeout: int) -> dict[str, str | None]:
    """Return model -> error; a missing error means the model lookup succeeded."""
    if not api_key:
        return {model: "OPENAI_API_KEY is missing" for model in models}
    headers = {"Authorization": f"Bearer {api_key}",
               "User-Agent": "auramaur-ibkr-etf-preflight/1.0"}
    client_timeout = aiohttp.ClientTimeout(total=min(timeout, 30))
    async with aiohttp.ClientSession(headers=headers, timeout=client_timeout) as session:
        results = {}
        for model in models:
            try:
                async with session.get(
                    f"https://api.openai.com/v1/models/{quote(model, safe='')}"
                ) as response:
                    if response.status == 200:
                        results[model] = None
                    else:
                        body = await response.text()
                        results[model] = f"HTTP {response.status}: {body[:120]}"
            except Exception as exc:  # noqa: BLE001
                results[model] = str(exc)[:160]
        return results


async def preflight(settings, db, *, client=None, model_checker=None) -> ETFPreflightReport:
    """Verify that an ETF paper run can acquire every required dependency."""
    cfg = settings.ibkr
    results: list[ETFPreflightResult] = []

    def add(name: str, severity: str, detail: str) -> None:
        results.append(ETFPreflightResult(name, severity, detail))

    if not cfg.enabled or not cfg.etf_paper_enabled:
        add("feature gates", "BLOCK", "ibkr.enabled and ibkr.etf_paper_enabled must both be true")
    else:
        add("feature gates", "OK", "IBKR master and ETF paper gates are enabled")

    own_client = client is None
    client = client or IBKREquityClient(settings, force_paper_readonly=True)
    if getattr(client, "_force_paper_readonly", False):
        add("paper isolation", "OK",
            f"simulated local book; read-only quote port {cfg.etf_quote_port}, "
            f"client {cfg.equity_client_id}, read-only")
    else:
        add("paper isolation", "BLOCK", "client is not forced onto paper/read-only routing")

    probe = cfg.etf_symbols[0]
    try:
        current = time.time()
        quote_data = await client.get_quote(probe)
        age = current - quote_data.timestamp if quote_data else float("inf")
        if quote_data and -5 <= age <= max(60, cfg.etf_cycle_seconds * 2):
            add("market quote", "OK", f"{probe} BBO available ({age:.0f}s old)")
        else:
            add("market quote", "BLOCK", f"{probe} has no fresh timestamped BBO")
    except Exception as exc:  # noqa: BLE001
        add("market quote", "BLOCK", str(exc)[:180])
    try:
        closes = await client.get_adjusted_daily_closes(probe)
        if closes:
            add("adjusted bars", "OK", f"{probe} ADJUSTED_LAST through {closes[-1][0]}")
        else:
            add("adjusted bars", "BLOCK", f"{probe} returned no adjusted daily bars")
    except Exception as exc:  # noqa: BLE001
        add("adjusted bars", "BLOCK", str(exc)[:180])
    if own_client:
        await client.close()

    required_tables = ("ibkr_etf_forecasts", "ibkr_etf_state",
                       "ibkr_etf_cooldowns", "ibkr_etf_openai_attempts",
                       "ibkr_etf_positions", "ibkr_etf_fills", "ibkr_etf_ledger")
    rows = await db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
    missing = sorted(set(required_tables) - {row["name"] for row in rows})
    if missing:
        add("database schema", "BLOCK", "missing: " + ", ".join(missing))
    else:
        add("database schema", "OK", "isolated ETF state and accounting tables are present")

    unpriced = [arm.alias for arm in cfg.etf_models
                if arm.input_cost_per_million <= 0 or arm.output_cost_per_million <= 0]
    if unpriced:
        add("token pricing", "BLOCK", "set explicit input/output token prices for: "
            + ", ".join(unpriced))
    else:
        add("token pricing", "OK", "every model arm books usage cost into its cell ledger")

    checker = model_checker or _check_openai_models
    models = [arm.model for arm in cfg.etf_models]
    errors = await checker(settings.openai_api_key, models, cfg.etf_openai_timeout_seconds)
    failed = [f"{model}: {error}" for model, error in errors.items() if error]
    if failed:
        add("OpenAI models", "BLOCK", "; ".join(failed)[:500])
    else:
        add("OpenAI models", "OK", f"access confirmed for {len(models)} model arm(s)")

    add("experiment cells", "OK",
        ", ".join(f"{arm.alias}={arm.model}/{arm.effort}" for arm in cfg.etf_models))
    return ETFPreflightReport(tuple(results))
