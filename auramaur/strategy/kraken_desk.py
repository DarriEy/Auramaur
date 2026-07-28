"""One Opus call that looks at the whole Kraken book and decides.

Replaces a per-pair "give me a probability" call that could see neither the
other pairs, nor the positions already open, nor what a trade costs. It was
gated on a MEDIUM confidence floor while being instructed to answer LOW on weak
evidence, so it rejected 100% of its own signals and Kraken has not traded
since 2026-06-11.

Two rules govern this module, and they are what keep a stronger model from
being a more expensive way to lose money:

1. **The model proposes; arithmetic disposes.** A desk decision is only ever a
   request. `edge_needed_prob` — the probability at which the trade stops
   losing to Kraken's 0.26%-per-side fee — is computed from live volatility and
   applied by the caller AFTER the model has spoken. The model cannot argue its
   way past its own costs.
2. **Parsing fails closed.** Anything unreadable, out of range, or about a pair
   that was not asked for yields no decision, which means no trade. A desk that
   cannot be understood does not get to act.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

# What the model may ask for. Anything else is discarded rather than guessed at.
ACTIONS = frozenset({"enter", "hold", "exit", "skip"})

SYSTEM_INSTRUCTIONS = """\
You are running a small crypto spot desk on Kraken. You are shown the current \
market state for each pair, the positions you already hold, and your realised \
P&L so far.

For EACH pair, decide one of:
  enter - open a new long (only if flat in that pair)
  hold  - keep an existing position unchanged
  exit  - close an existing position
  skip  - do nothing

State a calibrated probability that the pair trades HIGHER at the stated \
horizon. Calibrated means what you actually believe, not what justifies a \
trade: if the evidence is weak, say so and the number should sit near 0.50.

Every pair's block ends with the probability at which a trade stops losing to \
its own costs. You do not need to enforce that — it is applied after you — but \
it tells you how strong a view has to be before it is worth anything. A view \
of 0.55 on a pair needing 0.60 is a legitimate 'skip', not a failure.

Prefer skip. Trading costs money; being right about direction does not.

Return ONLY a JSON array, one object per pair, no prose and no code fence:
[{"pair":"XBTUSDC","action":"skip","probability":0.52,"confidence":"LOW",\
"reasoning":"one short sentence"}]
"""


@dataclass(frozen=True)
class DeskDecision:
    pair: str
    action: str
    probability: float
    confidence: str
    reasoning: str


def _extract_json_array(text: str) -> list | None:
    """Pull the JSON array out of a model reply, tolerating fences and prose."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            return None
        candidate = text[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, list) else None


def parse_desk_response(text: str, expected_pairs) -> dict[str, DeskDecision]:
    """Strict, fail-closed parse of the desk reply.

    A malformed entry is dropped rather than defaulted: defaulting a missing
    probability to 0.5, or an unknown action to 'skip', would silently convert a
    broken response into a confident-looking decision.
    """
    allowed = {str(p) for p in expected_pairs}
    rows = _extract_json_array(text)
    if not rows:
        return {}
    out: dict[str, DeskDecision] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        pair = str(row.get("pair", "")).strip()
        action = str(row.get("action", "")).strip().lower()
        if pair not in allowed or action not in ACTIONS:
            continue
        try:
            probability = float(row.get("probability"))
        except (TypeError, ValueError):
            continue
        if not 0.0 <= probability <= 1.0:
            continue
        out[pair] = DeskDecision(
            pair=pair, action=action, probability=probability,
            confidence=str(row.get("confidence", "")).strip().upper() or "LOW",
            reasoning=str(row.get("reasoning", ""))[:300])
    return out


def tradeable(decision: DeskDecision, edge_needed: float | None) -> tuple[bool, str]:
    """Does the arithmetic allow what the desk asked for?

    Applied AFTER the model, deliberately. ``edge_needed`` is the probability at
    which expected edge covers the round-trip fee; None means no probability
    does at this horizon, which is a refusal rather than a licence.
    """
    if decision.action != "enter":
        return False, f"action={decision.action}"
    if edge_needed is None:
        return False, "no probability covers the cost at this horizon"
    if decision.probability < edge_needed:
        return False, (f"p={decision.probability:.3f} below the "
                       f"{edge_needed:.3f} its costs require")
    return True, f"p={decision.probability:.3f} >= {edge_needed:.3f}"
