"""Vol-anchor pillar — deterministic GBM pricing of crypto threshold markets.

The edge (observed 2026-07-09, ETH): crowd prices for threshold markets at
DIFFERENT horizons back out to the SAME implied volatility — a flat vol term
structure anchored on the current tape (a 4-day and a 6-month ETH market both
implied sigma ~52% during an unusually calm week). Volatility mean-reverts;
from a depressed tape the long-horizon expectation belongs above the recent
realized, and from an elevated tape below it. So long-dated touch/threshold
markets are systematically mispriced whenever recent realized vol sits far
from the long-run anchor: buy long-dated touch when the tape is calm, fade it
when the tape is wild.

This is the reading edge in its cheapest form — no LLM anywhere. Inputs are
spot and daily closes (CoinGecko, free endpoints), a strike and deadline
parsed from the question text, and two closed-form GBM probabilities under
the martingale convention (E[S_T] = S0, log-drift -sigma^2/2 — the
finance-standard "driftless"; deliberately conservative for touch-up vs the
zero-log-drift reflection). Directional drift views are OUT of scope: the
edge claim is vol anchoring, not price forecasting, and drift is where
forecasting would sneak back in.

Sigma blends recent realized toward a per-asset long-run anchor with horizon:
    sigma(T) = anchor + (realized - anchor) * exp(-T / tau)
so a 4-day market prices off the tape (as it should) and a 6-month market
prices mostly off the anchor — exactly the term structure the crowd flattens.

Candidates come from LIVE discovery only. The markets table freezes resolved
rows at active=1 with stale prices — a stale-DB scan of this family showed
strike-monotonicity "violations" that evaporated against live quotes.

Rides the standard rails: signals + trades attribution, the full RiskManager
gate, ExecutionGateway placement, resolution-tracker settlement, calibration.
PAPER-FORCED, own graduation cell, one position per market bot-wide.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone

import aiohttp
import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.data_sources.deribit_iv import DeribitIVSource
from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from auramaur.strategy.classifier import blocked_category_hit, ensure_category

log = structlog.get_logger()

_COINGECKO = "https://api.coingecko.com/api/v3"

# Asset detection: question keyword -> coingecko id. Crypto only in v1 —
# commodities ("WTI", "Silver") have different vol dynamics and no clean
# spot/anchor source here.
_ASSETS = {
    "bitcoin": "bitcoin", "btc": "bitcoin",
    "ethereum": "ethereum", "eth": "ethereum",
    "solana": "solana", "sol": "solana",
    "xrp": "ripple",
    "dogecoin": "dogecoin", "doge": "dogecoin",
}

_MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}

_STRIKE = r"\$([0-9][0-9,]*(?:\.[0-9]+)?)"
_DATE = r"([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?"

# kind: 'touch_up' / 'touch_down' / 'above' / 'below'
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("touch_up", re.compile(
        rf"\b(?:reach|hit)\s+{_STRIKE}.*\bby\s+{_DATE}\s*\??\s*$", re.I)),
    ("touch_down", re.compile(
        rf"\b(?:dip|fall|drop)\s+to\s+{_STRIKE}.*\bby\s+{_DATE}\s*\??\s*$", re.I)),
    ("above", re.compile(
        rf"\bbe\s+above\s+{_STRIKE}\s+on\s+{_DATE}\s*\??\s*$", re.I)),
    ("below", re.compile(
        rf"\bbe\s+below\s+{_STRIKE}\s+on\s+{_DATE}\s*\??\s*$", re.I)),
]


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def touch_prob(spot: float, barrier: float, sigma: float, t_years: float) -> float:
    """P(barrier is touched within T) under GBM with the martingale
    convention (log-drift mu = -sigma^2/2). Standard first-passage formula:
    P = Phi((mu*T - b)/(s√T)) + exp(2*mu*b/s^2) * Phi((-mu*T - b)/(s√T)),
    with b = |ln(barrier/spot)| and the sign of mu flipped for a downward
    barrier (by GBM symmetry in log space)."""
    if spot <= 0 or barrier <= 0 or sigma <= 0 or t_years <= 0:
        return 0.0
    b = math.log(barrier / spot)
    mu = -0.5 * sigma * sigma
    if b == 0:
        return 1.0
    if b < 0:  # downward barrier: reflect
        b, mu = -b, -mu
    st = sigma * math.sqrt(t_years)
    p = _phi((mu * t_years - b) / st) + \
        math.exp(2.0 * mu * b / (sigma * sigma)) * _phi((-mu * t_years - b) / st)
    return min(1.0, max(0.0, p))


def terminal_above_prob(spot: float, strike: float, sigma: float,
                        t_years: float) -> float:
    """P(S_T > strike) under the same martingale GBM."""
    if spot <= 0 or strike <= 0 or sigma <= 0 or t_years <= 0:
        return 0.0
    st = sigma * math.sqrt(t_years)
    d = (math.log(spot / strike) - 0.5 * sigma * sigma * t_years) / st
    return min(1.0, max(0.0, _phi(d)))


def blended_sigma(realized: float, anchor: float, t_years: float,
                  tau_years: float) -> float:
    """Horizon-dependent vol: the tape for short T, the long-run anchor for
    long T — the mean-reverting term structure the crowd flattens."""
    w = math.exp(-t_years / max(tau_years, 1e-6))
    return anchor + (realized - anchor) * w


def parse_threshold(question: str) -> tuple[str, float, datetime] | None:
    """(kind, strike, deadline) from the question text, or None. Only
    unambiguous phrasings are traded."""
    q = (question or "").strip()
    for kind, pat in _PATTERNS:
        m = pat.search(q)
        if not m:
            continue
        try:
            strike = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        month = _MONTHS.get(m.group(2).lower())
        if month is None:
            continue
        day = int(m.group(3))
        year = int(m.group(4)) if m.group(4) else datetime.now(timezone.utc).year
        try:
            deadline = datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            continue
        return kind, strike, deadline
    return None


def detect_asset(question: str) -> str | None:
    ql = (question or "").lower()
    for kw, cg_id in _ASSETS.items():
        if re.search(rf"\b{kw}\b", ql):
            return cg_id
    return None


class VolAnchorPillar:
    """Deterministic vol-anchored threshold pricing over Polymarket crypto."""

    name = "vol_anchor"

    def __init__(self, db, settings, discovery, exchange, risk_manager,
                 pnl_tracker, calibration) -> None:
        self._db = db
        self._settings = settings
        self._discovery = discovery
        self._exchange = exchange
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._calibration = calibration
        self._gateway = ExecutionGateway(
            router=None, exchange=exchange, exchange_name="polymarket",
            settings=settings, db=db, pnl_tracker=pnl_tracker,
        )
        va = settings.vol_anchor
        self._iv_source = (DeribitIVSource(
            dict(va.deribit_currencies), ttl_seconds=va.deribit_ttl_seconds)
            if getattr(va, "sigma_source", "blend") == "deribit_iv" else None)

    # ------------------------------------------------------------------
    # Market data — spot + realized vol, fetched fresh each cycle
    # ------------------------------------------------------------------

    async def _asset_state(self, cg_id: str, cfg) -> tuple[float, float] | None:
        """(spot, realized_vol_annualized) or None. Fail CLOSED: no fresh
        data, no trades on that asset this cycle."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{_COINGECKO}/coins/{cg_id}/market_chart",
                    params={"vs_currency": "usd",
                            "days": str(int(cfg.realized_window_days)),
                            "interval": "daily"},
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except Exception as e:
            log.warning("vol_anchor.price_fetch_failed", asset=cg_id,
                        error=str(e)[:120])
            return None
        closes = [p[1] for p in (data.get("prices") or []) if p and p[1]]
        if len(closes) < 10:
            return None
        rets = [math.log(closes[i] / closes[i - 1])
                for i in range(1, len(closes)) if closes[i - 1] > 0]
        n = len(rets)
        mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / max(n - 1, 1)
        realized = math.sqrt(var * 365.0)
        return closes[-1], realized

    # ------------------------------------------------------------------
    # Cycle
    # ------------------------------------------------------------------

    async def run_once(self) -> int:
        cfg = self._settings.vol_anchor
        if not cfg.enabled:
            return 0
        candidates = await self._candidates(cfg)
        if not candidates:
            log.info("vol_anchor.no_candidates")
            return 0

        # One data fetch per asset per cycle.
        state: dict[str, tuple[float, float]] = {}
        for cg_id in {a for a, _, _, _, _ in candidates}:
            s = await self._asset_state(cg_id, cfg)
            if s is not None:
                state[cg_id] = s

        now = datetime.now(timezone.utc)
        entered = 0
        priced = 0
        for cg_id, kind, strike, deadline, market in candidates:
            if entered >= cfg.max_entries_per_cycle:
                break
            if cg_id not in state:
                continue
            spot, realized = state[cg_id]
            t_years = (deadline - now).total_seconds() / (365.0 * 86400.0)
            if t_years <= 0:
                continue
            anchor = cfg.long_run_vol.get(cg_id, 0.0)
            if anchor <= 0:
                continue
            sigma = None
            sigma_src = "blend"
            if self._iv_source is not None:
                sigma = await self._iv_source.term_sigma(cg_id, t_years)
                if sigma is not None:
                    sigma_src = "deribit_iv"
            if sigma is None:
                # Fallback = the calibrated estimate (pre-Deribit behavior).
                sigma = blended_sigma(realized, anchor, t_years, cfg.tau_years)
            if kind in ("touch_up", "touch_down"):
                fair = touch_prob(spot, strike, sigma, t_years)
            elif kind == "above":
                fair = terminal_above_prob(spot, strike, sigma, t_years)
            else:  # below
                fair = 1.0 - terminal_above_prob(spot, strike, sigma, t_years)
            priced += 1
            edge_pts = abs(fair - market.outcome_yes_price) * 100.0
            log.info("vol_anchor.priced", market_id=market.id, asset=cg_id,
                     kind=kind, strike=strike, spot=round(spot, 2),
                     sigma=round(sigma, 3), sigma_src=sigma_src,
                     realized=round(realized, 3),
                     fair=round(fair, 3), market=market.outcome_yes_price,
                     edge=round(edge_pts, 1))
            if edge_pts < cfg.min_edge_pts:
                continue
            if await self._market_claimed(market.id):
                continue
            if await self._try_enter(market, fair, sigma, realized, kind, cfg):
                entered += 1
        log.info("vol_anchor.cycle", candidates=len(candidates), priced=priced,
                 entered=entered)
        return entered

    async def _candidates(self, cfg) -> list[tuple[str, str, float, datetime, Market]]:
        now = datetime.now(timezone.utc)
        emin = (now + timedelta(days=cfg.min_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        emax = (now + timedelta(days=cfg.max_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        raw: list[Market] = []
        try:
            for off in range(0, max(int(cfg.scan_limit), 1), 100):
                page = await self._discovery.get_markets(
                    limit=100, offset=off, order="volume",
                    end_date_min=emin, end_date_max=emax)
                if not page:
                    break
                raw.extend(page)
        except TypeError:
            raw = await self._discovery.get_markets(limit=cfg.scan_limit)

        out = []
        for m in raw:
            if not self._eligible(m, cfg):
                continue
            asset = detect_asset(m.question)
            if asset is None:
                continue
            parsed = parse_threshold(m.question)
            if parsed is None:
                continue
            kind, strike, deadline = parsed
            out.append((asset, kind, strike, deadline, m))
        return out

    def _eligible(self, market: Market, cfg) -> bool:
        if not market.active:
            return False
        if (market.exchange or "polymarket") != "polymarket":
            return False
        if market.liquidity < cfg.min_liquidity:
            return False
        if not (0.02 <= market.outcome_yes_price <= 0.98):
            return False
        excluded = set(self._settings.risk.blocked_categories) | set(cfg.exclude_categories)
        if blocked_category_hit(excluded, market.question, market.description,
                                market.category):
            return False
        return True

    async def _market_claimed(self, market_id: str) -> bool:
        row = await self._db.fetchone(
            "SELECT 1 FROM trades WHERE market_id = ? LIMIT 1", (market_id,))
        if row is not None:
            return True
        row = await self._db.fetchone(
            "SELECT 1 FROM portfolio WHERE market_id = ? LIMIT 1", (market_id,))
        return row is not None

    async def _try_enter(self, market: Market, fair: float, sigma: float,
                         realized: float, kind: str, cfg) -> bool:
        market_yes = market.outcome_yes_price
        side = OrderSide.BUY if fair > market_yes else OrderSide.SELL
        signal = Signal(
            market_id=market.id,
            market_question=market.question,
            claude_prob=min(0.99, max(0.01, fair)),
            claude_confidence=Confidence.MEDIUM,
            market_prob=market_yes,
            edge=abs(fair - market_yes) * 100.0,
            evidence_summary=(
                f"Vol-anchor {kind}: blended sigma {sigma:.2f} (realized "
                f"{realized:.2f} mean-reverted to anchor) -> GBM fair "
                f"{fair:.2f} vs market {market_yes:.2f}; no drift view."),
            recommended_side=side,
            strategy_source="vol_anchor",
            mispricing_reason=(
                "structural: crowd prices all horizons off recent realized "
                "vol (flat implied term structure); mean-reversion says "
                "long-dated thresholds are mispriced when the tape is "
                "calm/wild"),
        )
        await self._persist_signal(signal, market)

        decision = await self._risk.evaluate(signal, market)
        if not decision.approved or decision.position_size <= 0:
            log.info("vol_anchor.risk_rejected", market_id=market.id,
                     reason=decision.reason)
            return False
        size = min(decision.position_size, cfg.stake_usd)
        force_paper = cfg.paper or getattr(decision, "force_paper", False)
        res = await self._gateway.submit(TradeIntent(
            signal=signal, market=market, size_dollars=size,
            force_paper=force_paper))
        if res.status not in ("filled", "paper", "partial", "pending"):
            log.info("vol_anchor.order_rejected", market_id=market.id,
                     status=res.status, error=res.reason)
            return False
        await self._record_position(signal, market, res.order, res.result)
        log.info("vol_anchor.entered", market_id=market.id,
                 token=res.order.token.value, price=res.order.price,
                 size=res.order.size, fair=round(fair, 3),
                 paper=res.result.is_paper)
        return True

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
               VALUES (?, ?, ?, ?, ?, ?, ?, 'vol_anchor')""",
            (signal.market_id, signal.claude_prob, signal.claude_confidence.value,
             signal.market_prob, signal.edge, signal.evidence_summary,
             signal.recommended_side.value),
        )
        await self._db.commit()

    async def _record_position(self, signal: Signal, market: Market,
                               order, result) -> None:
        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price
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
             1 if result.is_paper else 0),
        )
        await self._db.commit()
        try:
            await self._calibration.record_prediction(
                order.market_id, signal.claude_prob, market.category or "")
        except Exception as e:
            log.debug("vol_anchor.calibration_error", error=str(e))
