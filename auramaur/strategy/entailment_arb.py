"""Entailment arbitrage — trade logical-implication violations between markets.

If market A logically implies market B at the RESOLUTION-CRITERIA level
(A resolving YES guarantees B resolves YES), then P(A) <= P(B) must hold.
A violation P(A) > P(B) + gap is model-free profit: buy NO on A at
(1 - P(A)) and YES on B at P(B) in equal token counts n. Cost is
n*(1 - gap); payout is n if A=YES (then B=YES: 0 + 1), 2n if A=NO,B=YES,
n if both NO — i.e. >= n in every world where the entailment holds.
The only real risks are (1) the entailment being wrong in the fine print
and (2) leg risk (one side fills, the other doesn't).

Pair sources, by trust:
  * LADDERS (deterministic, no LLM): families of markets differing only in
    a numeric threshold — "BTC above 70,200 / 71,000 on <same date>",
    "<same player> Top 5 / Top 10 / Top 20 at <same event>". Direction is
    mathematical: above(hi) => above(lo); Top(small) => Top(big).
  * market_relationships 'conditional' pairs, LLM-verified ADVERSARIALLY
    ("can A resolve YES while B resolves NO?") and cached in
    entailment_verdicts. Stored direction is never trusted — the
    correlator's rows are direction-ambiguous and often junk.

Dead-book guard: the 2026-06 relationship scan was full of phantom
"violations" — illiquid books showing placeholder mids near 0.5 (the same
pattern PR #80 fixed for the market maker). Legs require real liquidity
AND a sane spread before any price is believed.

Safety: PAPER-FORCED by default; both legs risk-checked and placed only
together (no single-leg); one shot per pair; rides the standard rails
(signals/trades/fills/portfolio) under strategy_source='entailment_arb',
which the graduation ladder scores like any other cell.
"""

from __future__ import annotations

import json
import re
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


# ----------------------------------------------------------------------
# Deterministic ladder detection
# ----------------------------------------------------------------------

_THRESHOLD_RE = re.compile(
    r"^(?P<pre>.*?)\b(?P<dir>above|below)\b\s*\$?(?P<num>\d[\d,]*(?:\.\d+)?)(?P<post>.*)$",
    re.IGNORECASE,
)
_TOPN_RE = re.compile(r"\btop\s+(?P<n>\d+)\b", re.IGNORECASE)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def parse_threshold(question: str):
    """Return (family_key, direction, value) for 'X above/below N' questions."""
    m = _THRESHOLD_RE.match(question or "")
    if not m:
        return None
    value = float(m.group("num").replace(",", ""))
    direction = m.group("dir").lower()
    key = ("thr", _norm(m.group("pre")), direction, _norm(m.group("post")))
    return key, direction, value


def parse_topn(question: str):
    """Return (family_key, n) for '<player> Top N at <event>' questions."""
    m = _TOPN_RE.search(question or "")
    if not m:
        return None
    n = int(m.group("n"))
    key = ("topn", _norm(_TOPN_RE.sub("top {n}", question, count=1)))
    return key, n


def ladder_pairs(markets: list[Market]) -> list[tuple[Market, Market, str]]:
    """All (implier, implied, why) pairs from threshold/Top-N families.

    implier => implied, so P(implier) <= P(implied) must hold.
    """
    thr_families: dict[tuple, list[tuple[float, Market]]] = {}
    top_families: dict[tuple, list[tuple[int, Market]]] = {}
    for m in markets:
        t = parse_threshold(m.question)
        if t:
            key, _direction, value = t
            thr_families.setdefault(key, []).append((value, m))
            continue
        t2 = parse_topn(m.question)
        if t2:
            key2, n = t2
            top_families.setdefault(key2, []).append((n, m))

    pairs: list[tuple[Market, Market, str]] = []
    for key, members in thr_families.items():
        if len(members) < 2:
            continue
        direction = key[2]
        members.sort()
        for i, (v_lo, m_lo) in enumerate(members):
            for v_hi, m_hi in members[i + 1:]:
                if v_hi == v_lo:
                    continue
                if direction == "above":
                    # above(hi) => above(lo)
                    pairs.append((m_hi, m_lo, f"above {v_hi:g} => above {v_lo:g}"))
                else:
                    # below(lo) => below(hi)
                    pairs.append((m_lo, m_hi, f"below {v_lo:g} => below {v_hi:g}"))
    for _key, members2 in top_families.items():
        if len(members2) < 2:
            continue
        members2.sort()
        for i, (n_small, m_small) in enumerate(members2):
            for n_big, m_big in members2[i + 1:]:
                if n_big == n_small:
                    continue
                # Top 5 => Top 10
                pairs.append((m_small, m_big, f"top {n_small} => top {n_big}"))
    return pairs


# ----------------------------------------------------------------------
# Kalshi "Above X" econ-bin ladders (ticker-keyed, not question-keyed)
# ----------------------------------------------------------------------
#
# Kalshi expresses a threshold ladder as a set of markets sharing one event
# title ("What will CPI YoY be?") with per-bin subtitles ("Above 4.5%"); the
# THRESHOLD and the FAMILY both live in the ticker, e.g.
# KXCPIYOY-26NOV-T4.5 -> family "KXCPIYOY-26NOV", threshold 4.5. The
# question-text parser can't be used here: every series uses "Above N%", so
# grouping on the subtitle would wrongly merge CPI / unemployment / GDP bins.
# Only the "-T<number>" strike form is a MONOTONIC ladder (above(hi) =>
# above(lo)); categorical strikes (KXFEDDECISION -H26/-C25) are excluded.

_KALSHI_LADDER_RE = re.compile(r"^(?P<family>.+)-T(?P<num>-?\d[\d,]*(?:\.\d+)?)$")


def parse_kalshi_ladder(ticker: str):
    """Return (family_key, value) for a Kalshi '-T<number>' threshold market,
    else None (categorical strikes and non-ladder tickers don't match)."""
    m = _KALSHI_LADDER_RE.match((ticker or "").strip())
    if not m:
        return None
    try:
        value = float(m.group("num").replace(",", ""))
    except ValueError:
        return None
    return ("kxthr", m.group("family")), value


def kalshi_ladder_pairs(markets: list[Market]) -> list[tuple[Market, Market, str]]:
    """(implier, implied, why) pairs from Kalshi 'Above X' ticker ladders.

    above(hi) => above(lo), so P(implier) <= P(implied) must hold — same
    model-free bound as ladder_pairs, but families/strikes come from the
    ticker. Kalshi-only; non-Kalshi or non-ladder tickers are ignored.
    """
    families: dict[tuple, list[tuple[float, Market]]] = {}
    for m in markets:
        if (m.exchange or "") != "kalshi":
            continue
        parsed = parse_kalshi_ladder(m.ticker)
        if parsed is None:
            continue
        key, value = parsed
        families.setdefault(key, []).append((value, m))

    pairs: list[tuple[Market, Market, str]] = []
    for _key, members in families.items():
        if len(members) < 2:
            continue
        members.sort()
        for i, (v_lo, m_lo) in enumerate(members):
            for v_hi, m_hi in members[i + 1:]:
                if v_hi == v_lo:
                    continue
                # "Above hi" implies "Above lo".
                pairs.append((m_hi, m_lo, f"above {v_hi:g} => above {v_lo:g}"))
    return pairs


# ----------------------------------------------------------------------
# LLM verification prompt (fuzzy pairs only)
# ----------------------------------------------------------------------

ENTAILMENT_PROMPT = """You are auditing two prediction markets for STRICT logical entailment at the resolution-criteria level. Be adversarial: the default answer is "none".

Market A: "{question_a}"
Resolution details A: {description_a}

Market B: "{question_b}"
Resolution details B: {description_b}

Strict entailment means: in EVERY possible world, if the implier resolves YES then the implied market MUST resolve YES under its own written criteria. Consider timing windows, qualifying language, different resolution sources, and edge cases. If you can construct ANY plausible scenario where the implier resolves YES and the implied resolves NO, the entailment fails.

Respond with ONLY this JSON:
{{"direction": "a_implies_b" | "b_implies_a" | "none", "confidence": 0.0-1.0, "counterexample": "<the best scenario that breaks the strongest candidate direction, or 'none found'>"}}"""


# ----------------------------------------------------------------------
# Pillar
# ----------------------------------------------------------------------

class EntailmentArbPillar:
    def __init__(self, db, settings, discovery, exchange, risk_manager,
                 pnl_tracker, analyzer=None) -> None:
        self._db = db
        self._settings = settings
        self._discovery = discovery
        self._exchange = exchange
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._analyzer = analyzer

    # -- market quality -------------------------------------------------

    def _real_book(self, m: Market) -> bool:
        """Reject dead/placeholder books before believing any price."""
        cfg = self._settings.entailment_arb
        if not m.active or not (0.0 < m.outcome_yes_price < 1.0):
            return False
        if (m.exchange or "polymarket") != "polymarket":
            return False
        if m.liquidity < cfg.min_liquidity:
            return False
        if m.spread and m.spread * 100.0 > cfg.max_spread_pct:
            return False
        if m.end_date is None:
            return False
        end = m.end_date if m.end_date.tzinfo else m.end_date.replace(tzinfo=timezone.utc)
        hours = (end - datetime.now(timezone.utc)).total_seconds() / 3600.0
        return hours >= cfg.min_hours_to_resolution

    # -- pair sources ---------------------------------------------------

    async def _conditional_pairs(self, by_id: dict[str, Market]):
        rows = await self._db.fetchall(
            "SELECT market_id_a, market_id_b FROM market_relationships "
            "WHERE relationship_type = 'conditional'",
        )
        # Load each leg DIRECTLY rather than intersecting with the scan window.
        # The scan is the top-N markets by VOLUME; conditional pairs are mostly
        # niche/low-volume markets, so both legs almost never co-occur in by_id
        # and the strategy evaluated ~none of its 50+ ready pairs. Prefer the
        # freshly-scanned market when present, else fetch it once (cached for
        # this cycle). Dead/illiquid legs are filtered by _real_book.
        fetch_cache: dict[str, Market | None] = {}

        async def _leg(mid: str) -> Market | None:
            if mid in by_id:
                return by_id[mid]
            if mid not in fetch_cache:
                try:
                    fetch_cache[mid] = await self._discovery.get_market(mid)
                except Exception as e:
                    log.debug("entailment.leg_fetch_error", market_id=mid, error=str(e))
                    fetch_cache[mid] = None
            return fetch_cache[mid]

        out = []
        for r in rows or []:
            a = await _leg(r["market_id_a"])
            b = await _leg(r["market_id_b"])
            if (a is not None and b is not None
                    and self._real_book(a) and self._real_book(b)):
                out.append((a, b))
        if out:
            log.info("entailment.conditional_pairs", evaluable=len(out),
                     total=len(rows or []))
        return out

    async def _verify_llm(self, a: Market, b: Market):
        """Cached adversarial entailment verdict for a fuzzy pair."""
        key = tuple(sorted((a.id, b.id)))
        row = await self._db.fetchone(
            "SELECT direction, confidence FROM entailment_verdicts "
            "WHERE market_id_a = ? AND market_id_b = ?", key,
        )
        if row is not None:
            return row["direction"], float(row["confidence"])
        if self._analyzer is None:
            return "none", 0.0
        first, second = (a, b) if key[0] == a.id else (b, a)
        prompt = ENTAILMENT_PROMPT.format(
            question_a=first.question, description_a=(first.description or "")[:600],
            question_b=second.question, description_b=(second.description or "")[:600],
        )
        direction, confidence, reasoning = "none", 0.0, ""
        try:
            raw = await self._analyzer._call_llm(prompt)
            parsed = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            direction = str(parsed.get("direction", "none"))
            confidence = float(parsed.get("confidence", 0.0))
            reasoning = str(parsed.get("counterexample", ""))[:400]
            if direction not in ("a_implies_b", "b_implies_a"):
                direction = "none"
        except Exception as e:
            log.warning("entailment.llm_parse_error", a=a.id, b=b.id, error=str(e))
        await self._db.execute(
            """INSERT OR REPLACE INTO entailment_verdicts
               (market_id_a, market_id_b, direction, confidence, source, reasoning)
               VALUES (?, ?, ?, ?, 'llm', ?)""",
            (key[0], key[1], direction, confidence, reasoning),
        )
        await self._db.commit()
        return direction, confidence

    async def _already_traded(self, a_id: str, b_id: str) -> bool:
        row = await self._db.fetchone(
            """SELECT 1 FROM trades WHERE strategy_source = 'entailment_arb'
               AND market_id IN (?, ?) LIMIT 1""", (a_id, b_id),
        )
        return row is not None

    # -- main cycle -------------------------------------------------------

    async def run_once(self) -> int:
        cfg = self._settings.entailment_arb
        if not cfg.enabled:
            return 0
        markets = await self._discovery.get_markets(limit=cfg.scan_limit)
        good = [m for m in markets if self._real_book(m)]
        by_id = {m.id: m for m in good}

        # 1. Deterministic ladders — direction known mathematically.
        candidates: list[tuple[Market, Market, str, float]] = [
            (impl, imp, why, 1.0) for impl, imp, why in ladder_pairs(good)
        ]

        # 2. Fuzzy conditional pairs — verify with the LLM ONLY when a
        # violation is on the table (saves calls), direction never trusted
        # from the correlator.
        if cfg.llm_enabled and self._analyzer is not None:
            for a, b in await self._conditional_pairs(by_id):
                gap_ab = a.outcome_yes_price - b.outcome_yes_price
                if abs(gap_ab) < cfg.min_gap:
                    continue
                direction, conf = await self._verify_llm(a, b)
                if conf < cfg.llm_min_confidence:
                    continue
                if direction == "a_implies_b":
                    candidates.append((a, b, "llm-verified", conf))
                elif direction == "b_implies_a":
                    candidates.append((b, a, "llm-verified", conf))

        placed = 0
        for implier, implied, why, conf in candidates:
            if placed >= cfg.max_pairs_per_cycle:
                break
            gap = implier.outcome_yes_price - implied.outcome_yes_price
            if gap < cfg.min_gap:
                continue
            if await self._already_traded(implier.id, implied.id):
                continue
            try:
                if await self._enter_pair(implier, implied, gap, why, conf):
                    placed += 1
            except Exception as e:
                log.error("entailment.entry_error", a=implier.id, b=implied.id,
                          error=str(e))
        if placed:
            log.info("entailment.cycle_done", pairs=placed)
        return placed

    # -- execution --------------------------------------------------------

    def _leg_signal(self, market: Market, side: OrderSide, fair: float,
                    gap: float, why: str) -> Signal:
        return Signal(
            market_id=market.id,
            market_question=market.question,
            # fair value is the OTHER leg's price — the entailment bound.
            claude_prob=max(0.01, min(0.99, fair)),
            # Verified structural implication, not a forecast — HIGH is honest
            # and keeps the divergence filter meaningful for true forecasts.
            claude_confidence=Confidence.HIGH,
            market_prob=market.outcome_yes_price,
            edge=gap * 100.0,
            evidence_summary=f"Entailment arb ({why}): P(implier) must be <= P(implied).",
            recommended_side=side,
            strategy_source="entailment_arb",
        )

    async def _enter_pair(self, implier: Market, implied: Market,
                          gap: float, why: str, conf: float) -> bool:
        cfg = self._settings.entailment_arb
        # Leg 1: NO on the implier (its YES is overpriced vs the bound).
        sig_a = self._leg_signal(implier, OrderSide.SELL,
                                 fair=implied.outcome_yes_price, gap=gap, why=why)
        # Leg 2: YES on the implied (its YES is underpriced vs the bound).
        sig_b = self._leg_signal(implied, OrderSide.BUY,
                                 fair=implier.outcome_yes_price, gap=gap, why=why)

        dec_a = await self._risk.evaluate(sig_a, implier)
        dec_b = await self._risk.evaluate(sig_b, implied)
        if not (dec_a.approved and dec_b.approved):
            log.info("entailment.risk_rejected", a=implier.id, b=implied.id,
                     reason_a=dec_a.reason, reason_b=dec_b.reason)
            return False

        # Equal-notional legs, capped by config; both-or-nothing.
        size = min(dec_a.position_size, dec_b.position_size, cfg.stake_usd)
        if size <= 0:
            return False
        force_paper = (getattr(dec_a, "force_paper", False)
                       or getattr(dec_b, "force_paper", False))
        is_live = self._settings.is_live and not cfg.paper and not force_paper

        order_a = self._exchange.prepare_order(sig_a, implier, size, is_live)
        order_b = self._exchange.prepare_order(sig_b, implied, size, is_live)
        if order_a is None or order_b is None:
            return False

        result_a = await self._exchange.place_order(order_a)
        if result_a.status not in ("filled", "paper", "partial", "pending"):
            log.warning("entailment.leg_a_rejected", a=implier.id,
                        status=result_a.status)
            return False
        result_b = await self._exchange.place_order(order_b)
        if result_b.status not in ("filled", "paper", "partial", "pending"):
            # Leg risk: A is in, B failed. Flag loudly — the position is
            # directional until B fills or A is exited.
            log.error("entailment.leg_b_failed_single_leg", a=implier.id,
                      b=implied.id, status=result_b.status,
                      error=result_b.error_message)

        for market, order, result in ((implier, order_a, result_a),
                                      (implied, order_b, result_b)):
            if result.status in ("filled", "paper", "partial", "pending"):
                await self._record_leg(market, order, result, why, gap)

        await self._db.execute(
            "UPDATE entailment_verdicts SET traded_at = datetime('now') "
            "WHERE market_id_a = ? AND market_id_b = ?",
            tuple(sorted((implier.id, implied.id))),
        )
        await self._db.commit()
        log.info("entailment.entered", a=implier.id, b=implied.id,
                 gap=round(gap, 3), why=why, confidence=conf,
                 paper=result_a.is_paper)
        return True

    async def _record_leg(self, market: Market, order, result, why: str,
                          gap: float) -> None:
        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price
        is_paper = bool(result.is_paper)
        if result.status in ("filled", "paper", "partial") and fill_size > 0:
            await self._pnl.record_fill(Fill(
                order_id=result.order_id, market_id=order.market_id,
                token_id=order.token_id, side=order.side, token=order.token,
                size=fill_size, price=fill_price, is_paper=is_paper,
            ))
        await self._db.execute(
            """INSERT OR IGNORE INTO markets (id, exchange, question, category,
               active, outcome_yes_price, outcome_no_price, volume, liquidity,
               last_updated)
               VALUES (?, 'polymarket', ?, ?, 1, ?, ?, ?, ?, datetime('now'))""",
            (market.id, market.question,
             ensure_category(market.question, market.description, market.category),
             market.outcome_yes_price, market.outcome_no_price,
             market.volume, market.liquidity),
        )
        await self._db.execute(
            """INSERT INTO signals (market_id, claude_prob, claude_confidence,
               market_prob, edge, evidence_summary, action, strategy_source)
               VALUES (?, ?, 'HIGH', ?, ?, ?, ?, 'entailment_arb')""",
            (market.id, market.outcome_yes_price, market.outcome_yes_price,
             gap * 100.0, f"entailment ({why})",
             order.side.value),
        )
        await self._db.execute(
            """INSERT INTO trades (market_id, timestamp, side, size, price,
               is_paper, order_id, status, strategy_source, exchange)
               VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, 'entailment_arb',
                       'polymarket')""",
            (order.market_id, order.side.value, fill_size, fill_price,
             1 if is_paper else 0, result.order_id,
             "filled" if result.status in ("filled", "paper") else result.status),
        )
        await self._db.execute(
            """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
               current_price, unrealized_pnl, category, token, token_id,
               is_paper, updated_at)
               VALUES (?, 'polymarket', 'BUY', ?, ?, ?, 0, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                   size = excluded.size, avg_price = excluded.avg_price,
                   current_price = excluded.current_price,
                   updated_at = excluded.updated_at""",
            (order.market_id, fill_size, fill_price, fill_price,
             market.category or "", order.token.value, order.token_id,
             1 if is_paper else 0),
        )
        await self._db.commit()
