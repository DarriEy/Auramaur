"""Name-the-gap audit — the adverse-selection antidote.

Measured fact (edge_vs_market.py + the divergence-bucket P&L): when the
model disagrees with the market and cannot say WHY the market is wrong,
the market is usually right — the mid-divergence (10-20%) bucket realized
a net loss at a low win rate in backtest. This module makes the disagreement
justify itself.

The estimation pipeline is deliberately PRICE-BLIND (anti-anchoring), so
the audit is a separate post-hoc call made only when an LLM signal is
about to trade a significant divergence: "you said X, the market says Y —
name the mechanism or fold." Mechanisms:

  structural    — fees, mechanical flows, attention gaps, market plumbing
  behavioral    — headline-vs-fine-print, longshot bias, vibe pricing
  informational — specific evidence the crowd verifiably lacks (cite it)
  none          — no nameable mechanism; the market probably knows better

"none" (or a refusal to answer) blocks the trade when the gate is enabled.
Audits are cached per market and re-run only when either probability has
moved materially or the audit is stale.
"""

from __future__ import annotations

import json

import structlog

log = structlog.get_logger()

GAP_AUDIT_PROMPT = """You are the final risk gate for a prediction-market trade. An analyst (working WITHOUT seeing the market price, to avoid anchoring) estimated P(YES) = {claude_prob:.2f}. The market prices YES at {market_prob:.2f}.

Question: "{question}"
Resolution criteria: {description}

Before this trade is allowed, you must name the MECHANISM that explains why the market is mispriced. Be skeptical: prediction markets aggregate real money from informed participants. An unexplained disagreement usually means the market knows something the analyst doesn't — that is how this bot lost money in the past.

Valid mechanisms:
- "structural": fees, mechanical order flow, attention/coverage gaps, market plumbing quirks
- "behavioral": the crowd prices the headline but the resolution criteria say otherwise (qualifying language, deadlines, literal counting, permanence bars), longshot bias, narrative/vibe pricing
- "informational": SPECIFIC evidence the crowd verifiably lacks or hasn't digested — you must cite it concretely
- "none": you cannot name one. Choosing "none" is the correct answer more often than not.

Respond with ONLY this JSON:
{{"mechanism": "structural" | "behavioral" | "informational" | "none", "reason": "<one concrete sentence naming the mechanism and the evidence for it, or why none exists>"}}"""

_VALID = {"structural", "behavioral", "informational", "none"}


class GapAuditor:
    """Cached post-hoc mispricing audits via the shared LLM router."""

    def __init__(self, db, analyzer, settings) -> None:
        self._db = db
        self._analyzer = analyzer
        self._settings = settings

    async def audit(self, signal, market) -> str:
        """Return the mispricing_reason string ("<mechanism>: <reason>" or "none")."""
        rc = self._settings.risk
        cached = await self._db.fetchone(
            "SELECT claude_prob, market_prob, mechanism, reason, audited_at "
            "FROM gap_audits WHERE market_id = ? AND audited_at >= datetime('now', ?)",
            (signal.market_id, f"-{int(rc.mispricing_audit_ttl_hours)} hours"),
        )
        if cached is not None:
            moved = (abs(float(cached["claude_prob"]) - signal.claude_prob) > 0.05
                     or abs(float(cached["market_prob"]) - signal.market_prob) > 0.05)
            if not moved:
                return self._format(cached["mechanism"], cached["reason"])

        mechanism, reason = "none", "audit unavailable"
        try:
            prompt = GAP_AUDIT_PROMPT.format(
                claude_prob=signal.claude_prob,
                market_prob=signal.market_prob,
                question=market.question,
                description=(market.description or "")[:600],
            )
            raw = await self._analyzer._call_llm(prompt)
            parsed = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            mechanism = str(parsed.get("mechanism", "none")).lower().strip()
            if mechanism not in _VALID:
                mechanism = "none"
            reason = str(parsed.get("reason", ""))[:400]
        except Exception as e:
            # Fail CLOSED to "none": an unauditable divergence trades like an
            # unexplained one. The gate (if enabled) will block it.
            log.warning("gap_audit.error", market_id=signal.market_id, error=str(e))

        await self._db.execute(
            """INSERT OR REPLACE INTO gap_audits
               (market_id, claude_prob, market_prob, mechanism, reason)
               VALUES (?, ?, ?, ?, ?)""",
            (signal.market_id, signal.claude_prob, signal.market_prob,
             mechanism, reason),
        )
        await self._db.commit()
        log.info("gap_audit.verdict", market_id=signal.market_id,
                 mechanism=mechanism)
        return self._format(mechanism, reason)

    @staticmethod
    def _format(mechanism: str, reason: str) -> str:
        if (mechanism or "none") == "none":
            return "none"
        return f"{mechanism}: {reason}"
