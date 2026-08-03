# Event-driven IBKR book — design brief

**Status: Phase 0 (brief + gates). No strategy code exists, by design.**
Decided 2026-08-03 (operator: "B as the one bet, let's design slowly and
replay hard"), the same day the global_etf momentum book was killed on
replay evidence (3112f0c; post-mortem in the `ibkr-momentum-no-edge`
memory and the dated comments in `runtime/config/defaults.local.yaml`).

## Posture this sits inside

IBKR is the vault, not the hunting ground: directed-orders beta (VT) and
treasury compound capital; the prediction venues remain the edge engine.
This book is the **one** designed exception — it earns existence through
the gates below, or it doesn't exist.

## Constraints inherited from the momentum post-mortem

These are not preferences; they are measured facts about this account and
this venue, and every design choice must respect them.

1. **The cost floor is a design input.** $1 minimum commission per leg
   means position size ≥ ~$2,000, or costs eat any measurable edge
   (momentum paid ~30bps round trip on $600–830 positions; 96% of its
   loss was fees before the signal even mattered).
2. **No daily-bar signals on hyper-liquid instruments.** Momentum IC on
   the ETF universe was statistically zero at every horizon. This shop's
   demonstrated edge — the entire Polymarket record — is information
   processing against lazily priced counterparties, not time-series
   patterns against institutions.
3. **Replay before paper, always.** The forward gate needs 180 elapsed
   days; a replay answers "does the mechanical core have any edge at our
   costs" today. Momentum burned a month of forward paper on a strategy
   whose 5-year replay failed — never again.
4. **Evidence cadence must fit the bar.** 30 round trips per book is the
   contract; the strategy's natural turnover must reach that in months,
   not years (event-paced: target 5–15 trips/month across the book).

## Hypothesis (mechanism named, per house rule)

Scheduled information events — macro prints (CPI, FOMC, ECB/BoE) and
earnings for the international-equity roster — repriced instruments over
**days**, not minutes, in two exploitable shapes:

- **Post-event drift**: the day-1 move under-reacts to the surprise and
  continues over a bounded window (the PEAD family for earnings; macro
  announcement drift for rates-sensitive instruments).
- **Cross-instrument read-through**: the event is priced into the direct
  instrument quickly but lags into correlated ones (an ASML print reaching
  SAP/semis peers; a CPI surprise reaching sector rotation after rates
  have already moved).

We are explicitly NOT competing on reaction speed (HFT owns the first
minutes) and NOT positioning ahead of events (that is a volatility bet
carrying event risk with no informational edge).

**Honest prior:** these anomalies are published and have decayed in US
large caps. The replay exists to measure whether anything remains on OUR
instruments at OUR costs. A null result is a cheap, acceptable outcome —
that is what "one bet" means.

## The replay-honesty problem, and the two-stage answer

An LLM-driven signal cannot be replayed honestly: any model call made
today "about" 2023 leaks hindsight. The design splits accordingly:

- **Stage 1 — mechanical skeleton (replayable).** Entry/exit rules that
  need only the event calendar, the consensus/actual surprise, and
  prices: e.g. *enter at the post-event close in the surprise direction
  when |surprise| ≥ threshold; exit at +N sessions or stop*. Every
  parameter fixed before scoring, out-of-sample split mandatory (the
  momentum exit-study showed this codebase how train rankings
  anti-generalise).
- **Stage 2 — LLM layer (forward-only).** If and only if Stage 1 passes,
  the LLM becomes a *filter/sizer* on top of the mechanical core (reading
  the release context, flagging one-off distortions), evaluated purely
  forward in paper, pre-registered like every other lane. The LLM layer
  must never be credited with the skeleton's edge.

## Pre-registered gates (set now, before any code)

- **G1 — data gate.** A clean historical event dataset (dates, consensus,
  actuals) for ≥5 years across the chosen events, with documented
  provenance and no look-ahead in the consensus series. If assembling
  this honestly proves impossible, the project stops here.
- **G2 — replay gate.** The mechanical skeleton, replayed over ≥5 years
  with the existing cost model (real commission arithmetic, 25bps assumed
  spread for equities), must show: 95% LCB on mean net P&L per trip > 0
  on the TEST split, and drawdown within the book's budget. Anything
  less: stop, and write the null result into the docs like the momentum
  one.
- **G3 — forward gate.** Only after G2: paper book via the standard
  contract (30 trips, LCB > 0, drawdown budget), entries sized ≥ $2,000,
  LLM layer pre-registered separately. The 180-day forward clock applies
  as to any IBKR book.

## Phases

- **Phase 0 (this document).** Brief + gates. Done 2026-08-03.
- **Phase 1 — data spike.** Source the event/surprise history (macro:
  FRED actuals are easy, consensus history is the hard part; earnings:
  dates + EPS surprise for the 16-name intl roster). Deliverable: a
  documented dataset and a one-page provenance note. *No strategy code.*
- **Phase 2 — replay harness.** A sibling of `ibkr_replay.py` for
  event-window replays (never writes to the forward tables), plus the
  Stage-1 rule set with parameters frozen before scoring.
- **Phase 3 — G2 decision.** Run, score, decide, document — whichever way
  it goes.
- **Phase 4 — forward paper.** Standard book wiring (the `entries_enabled`
  lever ships enabled:false until G2 passes), LLM layer design doc as a
  separate Phase-4 artifact.

## Stage-1 FROZEN specification (2026-08-03, fixed before first scoring run)

Scope: the **earnings leg only**. The macro legs (CPI/FOMC/ECB) need an
instrument-mapping design of their own (which instruments express a CPI
surprise?) — that is a separate Stage-1b spec, deliberately not rushed
into this one.

**Primary rule (the only one G2 judges):**

- Universe: the international-equity roster; events from the frozen
  `earnings-yfinance-adr-retrieved-2026-08-03.csv` joined to the frozen
  timing file. An event is usable iff estimate and actual are both
  present; surprise := actual − estimate (Yahoo-internal basis, honoring
  the G1 sign-only constraint); zero surprise → skipped.
- Direction: **long-only on positive surprise**. Negative-surprise
  events are skipped in the primary (shorting European names adds borrow
  costs the book does not model).
- Entry: at the home listing's close on the report date when
  timing_class = BMO, else at the next available session's close for
  that instrument; buy at close × (1 + half-spread) with the book's
  `adverse_fill` slippage.
- Exit: at the close of the **5th session after entry**, sell at
  close × (1 − half-spread) with `adverse_fill`; commission charged both
  ways via the book's own equity schedule.
- Size: **fixed $2,000 notional per event** (fractional), the
  cost-floor design input; no volatility sizing in the skeleton.
- Costs: assumed spread = the deployed intl book's `max_spread_bps`
  (the worst the live book would accept — errs pessimistic), plus the
  book's `slippage_bps` per leg, plus commissions.
- Currency: home-listing closes convert at **static FX rates frozen
  here** (GBX 0.0127, EUR 1.17, USD 1.0, CAD 0.71, JPY 0.0068,
  HKD 0.128, AUD 0.65 per unit → USD). A documented approximation: it
  keeps notional and commission arithmetic realistic; it does not model
  FX P&L over the 5-session hold.
- Split and gate: **G2 verdict = `evaluate_ibkr_evidence` on TEST-split
  trips (entry ≥ 2024-01-01), min 30 observations, LCB > 0 and drawdown
  within the book budget.** Train (pre-2024) and full-period results
  reported for context, never for the verdict.
- Coverage honesty: instruments whose 5-year bars cannot be fetched are
  skipped and their event counts REPORTED as coverage loss, not silently
  dropped.

**Pre-registered sensitivities (informational only, never the gate):**
S1 symmetric long/short; S2 hold N=10; S3 hold N=21; S4 spread 25bps.
Anything not in this list that looks interesting later is a NEW
pre-registration, not a peek.

## Stage-1 earnings leg: G2 FAIL (scored 2026-08-03) — leg CLOSED

The primary ran the same day the spec froze, from the frozen datasets,
with zero tuning passes. 329 events loaded, 199 eligible trips (102
negative-surprise skipped per the long-only primary, 17 without bars —
Toyota has no TSEJ data permission — 9 zero-surprise, 2 without a full
hold window). Fifteen of sixteen instruments delivered 7-year bars.

| run | TEST trips | net | mean/trip | win | LCB95 | verdict |
|---|---|---|---|---|---|---|
| PRIMARY (long-only, N=5, 50bps) | 83 | −$1,113 | −$13.41 | 47.0% | −$31.13 | **FAIL** |
| S1 long/short | 105 | −$2,169 | −$20.66 | 42.9% | −$38.83 | FAIL |
| S2 hold N=10 | 82 | −$1,155 | −$14.09 | 37.8% | −$37.54 | FAIL |
| S3 hold N=21 | 81 | −$46 | −$0.57 | 50.6% | −$32.13 | FAIL |
| S4 spread 25bps | 83 | −$703 | −$8.47 | 49.4% | −$26.26 | FAIL |

**Interpretation, stated plainly.** Unlike momentum (flat gross, fees
fatal), the primary's gross is itself negative: mid-to-mid, the
post-earnings drift on these mega-cap names is worth roughly +15bps
over 5 sessions — real-ish, and about a fifth of the ~65bps it costs to
trade at $2,000 size. The sensitivities sharpen the picture without
changing it: shorting negative surprises makes things worse (S1); a
21-session hold turns gross positive and net near break-even (S3
gross +$278, net −$46) but with month-scale variance the LCB is nowhere
near zero. The "published anomaly, decayed in large caps" prior is
confirmed on our instruments with our costs. Train and test agree
throughout — this is not a split artifact.

**Consequences.** No paper book is built on this rule set; the earnings
leg is closed. Any successor idea — the S3 whisper at longer horizons,
small-cap universes, the macro legs (Stage-1b, with their cleaner
event-study surprises and an instrument-mapping design of their own) —
is a NEW pre-registration with its own frozen spec, or it does not run.
The gate consumed one day and ~nothing in costs, versus six months of
forward paper: this is the process working, not failing.

## Stage-1c + Stage-2 shadow: PRE-REGISTERED 2026-08-03, scored 2027-04-30

Registered the same day the Stage-1 earnings leg closed, on prospective
events only — the 347 historical events are burned as a scoring set (we
looked at them five times) and are never scored again. Justification for
Stage-1c is EXTERNAL (the PEAD literature's ~60-trading-day completion
window; cost amortization at one round trip per quarter), explicitly NOT
our S3 numbers — S3 motivated the registration's existence, and may not
motivate its parameters.

**Stage-1c — the literature-standard rule (frozen):**

- Identical to the Stage-1 primary except hold = **60 sessions**.
  Long-only, sign-only, $2,000 notional, timing-ruled entries; costs
  frozen NUMERICALLY as scored today (50bps assumed spread, 2bps
  slippage per leg, the book's commission schedule) so later config
  drift cannot move the bar.
- Event window: roster earnings with report date in
  **[2026-08-04, 2026-12-31]** — strictly future as of registration.
- Scoring: one replay run on **2027-04-30** (every event's 60-session
  window complete). Gate: `evaluate_ibkr_evidence` LCB > 0 with
  **min 20 trips** (the prospective window is smaller than 30 supports);
  if fewer than 20 trips accrue, the window extends through 2027-03-31
  and scores 2027-07-31 — that extension is pre-registered here and is
  the only one.
- **No interim peeks.** Until the scoring date the only permitted query
  is a coverage count (how many events accrued). A peeked replay voids
  the registration.

**Stage-2 shadow — the early-winner filter (frozen protocol):**

- For every roster earnings event from runner deployment onward, an LLM
  produces a same-day verdict: **pick / pass**, direction, one named
  mechanism, confidence — logged append-only with the log timestamp.
- Inputs allowed: the release's numbers, the consensus, pre-release
  context and news. Inputs FORBIDDEN: any post-release price path. A
  verdict logged after the event's Stage-1c entry close is marked LATE
  and excluded from scoring — lateness is measured, never fudged.
- Scoring, same date (2027-04-30): Stage-1c trips partitioned by the
  shadow's pick/pass. The filter shows value iff picked-subset mean net
  exceeds passed-subset mean net AND picked mean net > 0. Sample will be
  small; this is evidence-gathering for the Stage-2 layer, not
  graduation, and it touches no money by construction.
- The runner (detection + prompt + log) is a separate small build; its
  exact prompt is frozen in code at deploy and referenced here by
  commit. Coverage begins at deploy — events before it are simply
  uncovered, never backfilled. NOTE: nine roster names report between
  Aug 4 and Aug 27; a runner deployed this week captures the August
  wave, a later one starts with the autumn wave. Either is valid; the
  log shows which.

## Open questions (for the design period, not blockers today)

- Which venue hosts the cheapest test? FX has 100× lower costs and CPI/
  FOMC are its native events — but the replay harness lacks an FX bars
  path (extension needed). Equities have the earnings surface and the
  widened 16-name roster.
- Consensus-history sourcing without survivorship/lookahead bias — buy,
  scrape-with-care, or restrict to events where a consensus proxy exists.
- ~~Whether the intl book keeps accruing momentum trips as a control~~ —
  ANSWERED same day: its own 5y replay (322 trips, gross +$54.56,
  commission −$644.00, net −$589.44, win 27.6%) fails both contract arms.
  Gross of ~zero means the signal is noise there too; entries were turned
  off in the runtime override 2026-08-03. The widened roster waits for
  this book's Phase 4.
- Extend the replay harness with an FX bars path and replay fx momentum
  before its ~October 30-trip bar — fx costs are ~100× smaller so the
  arithmetic differs, but a zero-IC signal accrues meaningless evidence
  at any cost level.
