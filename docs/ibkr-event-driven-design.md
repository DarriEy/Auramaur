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
