"""Deterministic post-check on LLM-proposed arbitrage pairs (#405 item 1).

Both paired-arb pillars — cross_venue_arb (semantic equivalence across
Polymarket x Kalshi) and entailment_arb (logical implication between two
books) — delegate PAIRING CORRECTNESS entirely to an LLM verdict. #403
hardened the batch matcher's prompt, but prompt hardening only raises the
cost of abuse; the decision itself stays the model's word.

Per-leg risk checks structurally cannot cover this. Each leg is evaluated on
its own merits (stake cap, edge, category exposure) and nothing in the risk
manager observes that the two legs fail to OFFSET. A false "these are the
same market" verdict books naked directional exposure under a strategy_source
that says "arbitrage".

So: the model proposes, a rule confirms. A pair the LLM endorses must also
clear cheap, deterministic invariants derived from the two markets' own
metadata before it can trade.

--------------------------------------------------------------------------
CALIBRATION (measured 2026-08-06 against the live trading DB, read-only)
--------------------------------------------------------------------------
309 cross_venue_verdicts rows (2026-06-21 .. 2026-08-06) and 24
entailment_verdicts rows (2026-06-18 .. 2026-07-31); 36 'conditional'
market_relationships rows (the entailment LLM's candidate source).

Resolution-date deltas:
  * cross_venue candidates that reached the LLM: MINIMUM |end_a - end_b| over
    295 resolvable pairs was 3090 days; median 15,708 days. Not one candidate
    pair in the pillar's entire history resolved within eight years of its
    partner. The LLM answered "none" to all 309 — the date rule agrees with
    every verdict it has ever produced, and would have saved every call.
  * entailment: all 5 non-"none" verdicts have |end_a - end_b| == 0.0 exactly.
    'none' verdicts run out to 365 days.

Token containment (this module's tokenizer, min-set normalized):
  * the 5 real positive entailment verdicts score 0.400, 0.429, 0.833, 0.833,
    0.889 — minimum 0.400 ("Exact Score: Australia 0 - 2 Egypt?" vs "Australia
    vs. Egypt: O/U 7.5", a genuine mutual exclusivity). Under the scanner's
    existing `_word_overlap_score` the same five are 0.333, 0.429, 0.857,
    0.857, 0.889; the floor below is set under the lower of the two.
  * the cross_venue candidate population is bimodal: 217 of 296 pairs score
    below 0.1 (zero shared tokens) and a cluster sits at 0.4-0.67 built almost
    entirely on ONE or TWO incidental shared words.

That second point is why a bare ratio is not enough. Containment divides by the
SMALLER token set, and Kalshi event titles are short ("Who will the next Pope
be?" -> {pope} once stopwords are dropped), so a single shared token scores
0.5. Before "next" was added to the scanner's stopword list (d8e8d50,
2026-07-02), ~200 cross-venue pairs cleared the 0.5 pre-filter on the word
"next" alone. Hence MIN_SHARED_TOKENS: a ratio AND an absolute floor.

Thresholds are module constants, not config, deliberately: cross_venue_arb and
entailment_arb are NOT in graduation.exempt_strategies, so an edit inside their
config sections would reset their 14-day holdout clocks. A safety floor should
not cost a strategy its evidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from auramaur.nlp.prompts import format_untrusted_text
from auramaur.strategy.arbitrage_scanner import _STOP_WORDS

# --- thresholds -------------------------------------------------------
#
# MIN_CONTAINMENT: the smallest containment observed on a REAL positive verdict
# is 0.400 here (0.333 under the scanner's older tokenizer); 0.30 is the
# largest round value strictly below both, so the rule refuses 0 of the 5
# historical positives while removing the whole 0.0-0.30 band (73% of the
# cross-venue candidate population, all of which the LLM also refused, and 3 of
# the 36 correlator 'conditional' pairs). An attacker who controls one market's
# text must now make that
# market share 30% of the smaller token set with the victim market AND resolve
# on the victim's date — at which point it is, structurally, the same market.
MIN_CONTAINMENT = 0.30

# MIN_SHARED_TOKENS: two content tokens in common. One shared token is noise
# (the "next Pope" family above); every historical positive shares >= 2.
MIN_SHARED_TOKENS = 2

# Resolution-date tolerance. Venues stamp the end of the same real-world event
# differently — Kalshi records the market's close_time, Polymarket the expected
# settlement instant — so same-event pairs legitimately differ by hours, and
# occasionally by a weekend or a meeting-date-vs-month-end convention. 48h
# covers that and is still ~1500x tighter than the smallest mismatch the
# cross-venue lane has ever produced (3090 days).
MAX_DELTA_HOURS = 48.0
# 48h is far too loose for short-dated books, where whole ladders of distinct
# markets live inside two days ("BTC above 70k on July 21" vs "on July 22" are
# near-identical in text and do NOT entail each other). Scale the tolerance to
# the shorter leg's remaining horizon so a 6-hour book gets ~1.2h of slack.
DELTA_HORIZON_FRACTION = 0.20
MIN_DELTA_HOURS = 1.0

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# 20,000 -> 20000, so a comma-grouped number matches its ungrouped twin.
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
# 100k -> 100000, 1.2m -> 1200000: the same threshold, written short.
_MAGNITUDE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s?(bn|k|m|b)\b")
_MAGNITUDES = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "bn": 1_000_000_000}
# nasdaq100 -> nasdaq 100; letter/digit runs split apart.
_ALNUM_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")


def _expand_magnitude(m: re.Match) -> str:
    value = float(m.group(1)) * _MAGNITUDES[m.group(2)]
    return f" {int(value)} " if value == int(value) else f" {value} "


def normalized_tokens(text: object) -> set[str]:
    """Content tokens of a market question, comparable across venues.

    Runs the untrusted-text scrubber first (NFKC, control/BiDi strip,
    whitespace collapse) so the score cannot be moved by invisible framing,
    then normalizes the three ways venues actually differ while meaning the
    same number — thousands separators ("20,000"), magnitude suffixes ("100k")
    and glued letter/digit runs ("NASDAQ100"). Stopwords and bare single
    letters (initials, "O/U") are dropped as noise.
    """
    cleaned = format_untrusted_text(text, 600).lower()
    cleaned = _THOUSANDS_RE.sub("", cleaned)
    cleaned = _MAGNITUDE_RE.sub(_expand_magnitude, cleaned)
    cleaned = _ALNUM_BOUNDARY_RE.sub(" ", cleaned)
    words = set(_TOKEN_RE.findall(cleaned)) - _STOP_WORDS
    return {w for w in words if len(w) > 1 or w.isdigit()}


def containment_score(text_a: object, text_b: object) -> tuple[float, int]:
    """Return (shared / smaller-token-set, shared token count).

    Containment rather than plain Jaccard because genuine equivalents are
    worded at different lengths across venues — "Trump to win?" vs "Will
    Donald Trump win the 2028 presidential election?" is a real pair, and
    Jaccard buries it. The absolute shared-token count carries the guard that
    containment alone loses on very short questions.
    """
    a, b = normalized_tokens(text_a), normalized_tokens(text_b)
    if not a or not b:
        return 0.0, 0
    shared = a & b
    return len(shared) / min(len(a), len(b)), len(shared)


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def date_tolerance_hours(end_a: datetime | None, end_b: datetime | None,
                         now: datetime | None = None) -> float:
    """Allowed |end_a - end_b|, scaled to the shorter leg's remaining horizon."""
    now = now or datetime.now(timezone.utc)
    ends = [d for d in (_aware(end_a), _aware(end_b)) if d is not None]
    if not ends:
        return MIN_DELTA_HOURS
    horizon = min((d - now).total_seconds() / 3600.0 for d in ends)
    scaled = DELTA_HORIZON_FRACTION * max(horizon, 0.0)
    return min(MAX_DELTA_HOURS, max(MIN_DELTA_HOURS, scaled))


@dataclass(frozen=True)
class PairCheck:
    """Verdict of the deterministic post-check on one proposed pair."""

    ok: bool
    reason: str            # "" when ok, else a stable machine-readable code
    score: float           # token containment, [0, 1]
    shared_tokens: int
    delta_hours: float | None      # |end_a - end_b|, None if a date is missing
    tolerance_hours: float

    @property
    def detail(self) -> dict:
        """Structlog-ready payload (also what gets persisted on refusal)."""
        return {
            "reason": self.reason,
            "score": round(self.score, 3),
            "shared_tokens": self.shared_tokens,
            "delta_hours": (None if self.delta_hours is None
                            else round(self.delta_hours, 2)),
            "tolerance_hours": round(self.tolerance_hours, 2),
        }


def check_pair(question_a: object, end_a: datetime | None,
               question_b: object, end_b: datetime | None,
               *, id_a: str | None = None, id_b: str | None = None,
               now: datetime | None = None) -> PairCheck:
    """Confirm an LLM-proposed pair against the markets' own metadata.

    Refusal reasons, in evaluation order:
      same_market      — a market cannot be paired against itself.
      missing_end_date — a leg with no resolution date cannot be date-matched.
      date_mismatch    — the legs resolve too far apart (2028 vs 2026).
      low_shared_tokens/low_overlap — the questions are not about one claim.
    """
    score, shared = containment_score(question_a, question_b)
    tol = date_tolerance_hours(end_a, end_b, now)
    a, b = _aware(end_a), _aware(end_b)
    delta = (None if (a is None or b is None)
             else abs((a - b).total_seconds()) / 3600.0)

    def refuse(reason: str) -> PairCheck:
        return PairCheck(False, reason, score, shared, delta, tol)

    if id_a is not None and id_a == id_b:
        return refuse("same_market")
    if delta is None:
        return refuse("missing_end_date")
    if delta > tol:
        return refuse("date_mismatch")
    if shared < MIN_SHARED_TOKENS:
        return refuse("low_shared_tokens")
    if score < MIN_CONTAINMENT:
        return refuse("low_overlap")
    return PairCheck(True, "", score, shared, delta, tol)


# --- untrusted prompt payload ----------------------------------------
#
# Both pillars hand the two legs' question + description to an LLM whose JSON
# answer IS the trade gate. Market text is authored outside this system, so it
# rides inside ONE data boundary — never as bare interpolation, and never in
# two separately-delimited regions (the gap between them reads as ours).

def format_pair_legs(*, label_a: str, question_a: object, description_a: object,
                     label_b: str, question_b: object, description_b: object,
                     description_limit: int = 600) -> str:
    """Encode both legs as a single scrubbed JSON object for prompt insertion.

    Same contract as nlp.prompts.format_market_context — NFKC normalize, strip
    control/BiDi, collapse whitespace, bound length, JSON-encode, escape angle
    brackets so hostile text cannot synthesize the trust-boundary delimiters —
    but keyed by leg and with the pillars' tighter description bound.
    """
    payload = {
        label_a: {
            "question": format_untrusted_text(question_a, 1000),
            "description": format_untrusted_text(description_a, description_limit),
        },
        label_b: {
            "question": format_untrusted_text(question_b, 1000),
            "description": format_untrusted_text(description_b, description_limit),
        },
    }
    return json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"),
    ).replace("<", "\\u003c").replace(">", "\\u003e")
