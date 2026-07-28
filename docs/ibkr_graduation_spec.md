# Putting the IBKR books on the graduation ladder

Status: spec, not implemented. Written 2026-07-27.

## Why

The IBKR books cannot graduate today. Not "have not yet" — *cannot*. They sit
structurally outside the ladder, so the $1,099 CAD of real capital in the IBKR
account has no maturation process running against it.

Verified 2026-07-27:

- `strategy_experiments` holds 17 registered prospective clocks. **None is an
  IBKR book.** Every entry is a prediction-market cell (llm, agent_trader ×8,
  platform_consensus, long_horizon ×2, weather_temp, term_structure, vol_anchor,
  bias_harvest).
- `ibkr_multiasset_paper.py` and `ibkr_etf_paper.py` contain **zero** references
  to `ExecutionGateway`, `gateway.submit`, or `record_fill`.
- Entire realized IBKR history: **-$6.00, all commission**, one closed round
  trip at $0.00 gross (`international_equity`).

They are not failing the bar. They were never on the track.

## How a cell actually earns graduation

Two independent data paths feed `GraduationLadder.decide(strategy_source,
category)` (`auramaur/risk/graduation.py:66`):

1. **Realized P&L** — queried from `pnl_ledger` by `strategy_source`
   (`graduation.py:101,111,157,271`). Drives the net-P&L LCB.
2. **Prospective evidence** — `decision_snapshots` (`graduation.py:165`),
   written by `DecisionTracker.capture(...)` from
   `ExecutionGateway._capture_decision` (`execution_gateway.py:221`). This is
   what registers the `strategy_experiments` row, freezes a `strategy_version`
   (sha256 of the strategy config + risk contract), and starts the
   `holdout_warmup_days` clock.

The IBKR books write to `ibkr_paper_ledger` / `ibkr_etf_ledger` and their own
positions tables. Neither is `pnl_ledger`; neither triggers a capture. So both
inputs are empty, permanently.

## What has to change

Three pieces. All three are required — any one alone leaves the books unable to
graduate, and piece 3 without 1 and 2 would gate on evidence that does not
exist.

### 1. Capture a decision at entry

Call `DecisionTracker.capture(...)` on every IBKR entry, mirroring
`_capture_decision`. The IBKR books do not route through `ExecutionGateway`, so
this is inherited for free once the books route through `gateway.submit(...)`
— see "Resolved: route through the gateway" below, which is the chosen path.

Fields that need an IBKR-specific answer:

| Field | Prediction-market source | IBKR equivalent |
|---|---|---|
| `strategy_source` | pillar tag | book cell, e.g. `ibkr_fx_paper` |
| `event_family` | `neg_risk_market_id or market.id` | instrument key, or asset class for correlated legs |
| `cohort_id` | `{venue}:{family}` | `ibkr:{book}` |
| `fair_probability` | model probability | **no analogue** — see below |
| `reference_price` / `executable_price` | market prob / order price | quote mid / fill price |
| `strategy_version` | sha256(strategy config + risk) | same shape, hashing `settings.ibkr` |
| `is_paper` | `order.dry_run` | book's paper flag |

`fair_probability` is the real modelling problem. The ladder's second bar is a
**market-Brier-edge LCB** — it asks whether the strategy's probability beat the
market's. FX and ETF entries are not probabilistic forecasts, so there is no
Brier score to compute. Options:

- **(a)** Exempt directional-price books from the Brier bar and graduate them on
  net-P&L LCB alone. Simplest; weakens the bar to one dimension.
- **(b)** Define a synthetic forecast (e.g. P(price above entry at horizon))
  and score it. Honest but invents a quantity the strategy does not actually
  produce.
- **(c)** Replace the Brier bar for these books with a risk-adjusted return bar
  (Sharpe-like LCB on round-trip returns).

Recommend **(c)**, with **(a)** as an interim. Do not do (b) — a forecast the
strategy never made cannot be evidence about the strategy.

### 2. Book realized P&L where the ladder reads it

Every IBKR close must write a `pnl_ledger` row with `strategy_source` set to the
book cell and `is_paper` set correctly, in addition to (not instead of) the
existing `ibkr_paper_ledger` / `ibkr_etf_ledger` rows, which remain the
venue-native accounting.

Care required on double-counting: `pnl_ledger` feeds the global P&L reporting
and the drawdown gate. Adding IBKR rows changes those aggregates. Decide
explicitly whether IBKR paper P&L should influence the global drawdown latch —
recommend **no**, scoped by `is_paper` and venue, or a phantom-peak repeat is
likely (see the 2026-07-22 drawdown incident).

`ibkr_etf_paper` additionally writes fills by raw SQL with no `record_fill`
call (audit finding #7, 2026-07-26), so it has no registered exposure callsite
at all. That should be fixed as part of this work, not separately.

### 3. Consult the ladder before entering

The books currently never call `decide(...)`. Once 1 and 2 supply evidence, each
book must gate its entries on the returned `CellDecision`, honouring
`force_paper` exactly as `ExecutionGateway.submit` does. Until then, the books
must stay paper by construction — which they are today via `paper: true`, but
that is a config flag, not a ladder decision, and it is what allowed the
mislabelling in PR #375 (`PAPER_SIMULATED` on a book that has a live order
path).

## Sequencing and risk

Land in this order; each step is independently verifiable:

1. **Piece 2 first (booking).** Pure observability, no behaviour change. Lets
   the evidence accumulate while the rest is built, so the holdout clock is not
   waiting on the implementation.
2. **Piece 1 (capture).** Registers clocks. Still no behaviour change: capture
   is a write, not a gate.
3. **Decide the Brier-bar question** with real data in hand from 1 and 2.
4. **Piece 3 (gating).** The only step that changes what trades. Do it last, and
   only after the books have completed round trips — as of today the FX book has
   **10 buy fills and zero sells, ever**, so there is nothing to evaluate.

The IBKR side is where the audit found the least-verified machinery: the
execution mode was mislabelled, the ETF spread gate was disabling stop-losses on
held positions (fixed 2026-07-26), and the multiasset book's `unmarked_positions`
can block a book's entries indefinitely. Treat any change here as
money-path work even though the books are paper today — `IBKRMultiAssetExecution`
has a real live order path behind its gate chain.

## Non-goals

- Not enabling live IBKR trading. This spec ends at "a book *could* graduate";
  arming it remains an operator decision on evidence.
- Not moving capital between venues. The IBKR cash cannot fund the Polymarket
  edge; that is a transfer decision, and transfers are hard-off by config.
- Not changing the prediction-market ladder. Its bars stay as they are.

## Resolved: route through the gateway (corrected 2026-07-27)

An earlier revision of this spec recommended calling `DecisionTracker`
directly, on the grounds that the gateway is "built around prediction-market
orders … fitting FX and equities into it is a significant refactor of a
money-moving component."

**That was wrong, and measuring it disproved it.** The gateway is already
close to venue-agnostic:

- Only **nine lines** in `execution_gateway.py` touch prediction-market
  concepts at all.
- It reads exactly **six fields** off the intent: `market.id`,
  `market.category`, `market.neg_risk_market_id`, `signal.claude_prob`,
  `signal.market_prob`, `signal.strategy_source`.
- Everything else flows through `Order` — already generic (`market_id`,
  `side`, `size`, `price`, `dry_run`, `post_only`, `source`) — and
  `exchange.place_order(order)`, a protocol method **the IBKR clients already
  implement** (`ibkr.py:337`, `ibkr_equity.py:169`).

The three prediction-market couplings degrade benignly rather than breaking:

| Coupling | Behaviour for an IBKR instrument |
|---|---|
| `order.token.value` in the market-cap key | defaults to `YES`, so the cap becomes one per instrument — the correct semantics |
| `orderbook_snapshots` lookup in decision capture | returns nothing; already null-guarded (`book is None`) |
| `neg_risk_market_id` for the event family | optional; falls back to `market.id` |

So routing through the gateway is **both smaller and strictly better** than the
direct-`DecisionTracker` path: it buys decision capture, `pnl_ledger` booking,
the aggregate market-cap guard, the exposure-registry perimeter and the
`force_paper` contract in one move, instead of reimplementing each on the IBKR
side and drifting from the originals.

### The one real gap

`_build_order` obtains an `Order` from, in priority order: a `SmartOrderRouter`,
`exchange.prepare_executable_order(...)`, or `exchange.prepare_order(...)`.
**Neither IBKR client has any of the three.** That is the single missing piece,
and it is the first increment (below).

## Resolved: the Brier bar, and two blockers bigger than it (corrected 2026-07-27)

This spec called `fair_probability` "the real modelling problem" and advised
**(c)** with **(a)** interim, ruling out **(b)** because "a forecast the
strategy never made cannot be evidence about the strategy". Measuring it found
that reasoning sound but **applied to the wrong object**, and found two
structural blockers that no choice of bar could have fixed.

### The IBKR books are not one thing

| | ETF book | multiasset books |
|---|---|---|
| produces a probability? | **yes** — `ibkr_etf_forecasts`, 314 rows | no |
| resolvable binary? | **yes** — adjusted close, 5 sessions, with a resolver | no |
| right bar | **(b)**, on its own real forecast | **(c)** risk-adjusted return |

Option (b) does not invent anything for the ETF arms: they are *asked* for a
calibrated probability, the entry gate is a function of it, and it resolves
against adjusted closes. The prohibition stands for the multiasset books, which
produce no probability anywhere — and which have **1 round trip, ever**, so
their bar cannot be chosen from evidence yet and should wait.

### The benchmark is the drift, not a coin

`reference_price` is what the forecast must beat. 0.5 is wrong: equities rise
more often than not over five sessions, so a constant 0.55 collects a positive
Brier edge for knowing the direction of the last century. The reference is the
instrument's own trailing `horizon_up_rate` over completed closes only
(`risk/ibkr_math.py`), leakage-guarded by `completed_closes(bars, as_of)`.
Scoring against `momentum_control` was rejected: it returns 0.70/0.30 by fiat,
so it is badly calibrated and too easy to beat.

### Two blockers that outrank the bar

Both would have made the ETF book ungraduatable no matter how good its
forecasts were:

1. **`event_family` = `market_id`.** An (arm, symbol) pair was ONE family for
   all time: 28 symbols = 28 families ever, against `min_paired_forecasts: 30`.
   Two short, permanently. The family is now bucketed by ISO week, so
   overlapping same-week forecasts (which share four of five sessions) stay one
   family while genuinely fresh windows count. Carried on
   `neg_risk_market_id`, which `_capture_decision` already reads.
2. **Category cells.** `category = spec.asset_class` fragments one arm across
   16 cells — eleven hold a single symbol, the largest holds five. Every cell
   capped far below 30. The arms are now `strategy_level_strategies`: one
   strategy over a universe, not a separate claim per asset class.

### The bar is 4.24 sigma, not 1.645

`_prospective_stats` Bonferroni-corrects: `0.05 / (50 hypotheses x 90 looks)`
gives z ≈ **4.241**, and `confidence_z: 1.645` is a floor that never binds.
At a 0.02 mean Brier edge (sd 0.10) that needs ~450 paired forecasts; at 0.05
(sd 0.15), ~162. Worth knowing before reading a two-week result as failure.

### Also landed

- `orderbook_snapshots` written from the real IBKR BBO at capture, so fills
  stamp `book_cross` instead of `synthetic` and survive
  `require_executable_fills`. This supplies the data the check reads rather
  than widening `credible_fill_evidence`.
- `market_outcomes` written when a forecast resolves — but see the open
  question immediately below, which this does **not** close.

## OPEN: the ladder models one outcome per market; this book forecasts weekly

Found by audit on 2026-07-27, in the increment-3 work itself, before any
forecast had resolved.

`market_outcomes` is `UNIQUE(venue, market_id)` and `_prospective_stats` joins
`o.event_key = lower(d.venue)||':'||d.market_id`. That is the right model for a
prediction market, which resolves once. An ETF arm forecasts the same symbol
every week, so keying outcomes by the symbol-scoped `market_id` would let the
**first** resolved forecast for (arm, symbol) own the only row that pair can
ever have — and every later week's forecast would be scored against that one
stale outcome. Wrong evidence into a gate is worse than none.

Interim, in effect now: outcomes are keyed by **family** (the weekly bucket),
so each week banks its own row and **nothing joins**. The data accumulates
correctly and no false Brier edge can reach the ladder. The cost is honest —
the ETF book still produces no countable prospective evidence.

**DECIDED 2026-07-27: (iii).** Forecast scoring stays out of the shared ladder;
`auramaur ibkr-calibration` is the forecast-quality instrument. The reason is
not statistical, it is that **the ladder does not gate live IBKR trading at
all** — see the next section. Changing `market_outcomes`'s uniqueness and the
`_prospective_stats` join is a schema-and-query change to machinery 17
prediction-market cells depend on, bought for no trading benefit. Revisit only
if the ETF book is ever given a real order path.

Closing it needs one of:

- **(i)** join prospective evidence on `event_family` rather than `market_id`,
  and relax `UNIQUE(venue, market_id)`. Correct, and touches the shared ladder
  and schema that 17 prediction-market cells depend on.
- **(ii)** give these books the risk-adjusted-return bar (option **c** above)
  and exempt them from `require_market_brier_edge`. Smaller blast radius;
  discards a real, measurable forecast signal.
- **(iii)** decouple forecast scoring from the ladder entirely and keep
  `auramaur ibkr-calibration` as the forecast-quality instrument, letting the
  ladder score execution only.

Note that (iii) is close to what already exists: the calibration report
measures forecast quality directly and does not need the ladder's plumbing.
The unresolved part is which of these the *graduation decision* should rest on.

## Increments

1. **Intent adapter + `prepare_order` on the IBKR client.** Present an
   `InstrumentSpec` as the six fields the gateway reads, and give the client a
   `prepare_order` that builds an `Order` from (spec, quote, side, size).
   Wired behind the existing `paper` flag so **nothing changes about what
   trades**. This is the increment to land first.
2. **Route `_fill` through `gateway.submit(...)`** instead of writing fills
   directly. Booking and capture both arrive as a consequence — pieces 1 and 2
   of "What has to change" are satisfied by this single move.
3. **Decide the Brier-bar question** (see above) with real captured decisions in
   hand.
4. **Gate on `decide(...)`** — last, and only once the books have completed
   round trips.

## The graduation ladder does NOT gate live IBKR trading (measured 2026-07-27)

Worth stating loudly, because increments 1-3 were built on the assumption that
it does. `IBKRMultiAssetExecution.graduated()` never touches
`GraduationLadder`. It runs its own pre-registered contract
(`auramaur/risk/ibkr_evidence.py`) over two book-owned tables:

| gate | source | requirement |
|---|---|---|
| daily marks | `ibkr_paper_daily_marks` | 120 observations, 180 elapsed days, positive 95% LCB, drawdown <= 10% of budget |
| round trips | `ibkr_paper_round_trips` | 30 observations, 180 elapsed days, same LCB and drawdown |

Both must pass, and the operator gate chain
(`multiasset_execution_enabled`, `multiasset_execution_confirm_live`,
`multiasset_execution_books`, `settings.is_live`, `ibkr.environment == "live"`,
no kill switch) sits on top. The ETF book cannot reach any of this: it is
`PAPER_SIMULATED` against a local simulator that refuses a non-`dry_run` order.

So the ladder work matures *evidence quality*; it does not shorten the path to
a real order. That path runs only through the executable multiasset books
(`global_etf`, `futures`, `international_equity`).

### Where that clock actually stands

| book | registry | daily marks | round trips | clock |
|---|---|---|---|---|
| `global_etf` | 35 eligible | 4 | 0/30 | started 2026-07-24 |
| `international_equity` | **2 of 9 eligible** | 4 | 1/30 | started 2026-07-24 |
| `futures` | **0 of 9 eligible — all quarantined** | **0** | 0/30 | **never started** |

Earliest possible live date is therefore **2027-01-20** (first mark + 180 days),
and only for `global_etf`, and only if turnover produces 30 round trips.

Three things bind, in order of how fixable they are:

1. **`futures` is quarantined on market data, not code.** All 9 contracts carry
   `last_error: "no executable BBO"` and `quote_source: none` since 2026-07-23
   — the same missing entitlement behind the `Error 10089` lines in the
   container log. Its 180-day clock cannot start until the CME subscription
   exists. Every day of delay is a day added to the earliest live date.
2. **`international_equity` is running on 2 of 9 instruments** (4
   `qualified_no_live_data`, 3 quarantined), which is why it has managed one
   round trip. It will not reach 30 at that breadth.
3. **The daily-mark streak has almost no slack**: ~124 trading days fall in the
   180-day window against a 120-mark minimum, so only **4 missable days**. A
   week-long outage does not delay graduation by a week — it resets the
   earliest date by however long it takes to accumulate 120 marks.

## Measured 2026-07-27: global_etf has no edge, and exits are not the reason

The forward clock cannot answer "is this worth waiting for" until 2027-01-20.
The book's logic is deterministic, so `auramaur ibkr-backtest` answered it in
minutes. Replay writes nothing to `ibkr_paper_round_trips` /
`ibkr_paper_daily_marks` — those hold the pre-registered forward record.

**Deployed rules, 2021-07-29..2026-07-27, 35 instruments, 1253 sessions:**

| assumed spread | trips | net | mean/trip | win rate |
|---|---|---|---|---|
| 3bps | 463 | **-$968.74** | -$2.09 | 28.1% |
| 25bps | 470 | -$1,308.81 | -$2.78 | 24.9% |

~19% of the $5,000 budget at realistic cost, on 463 observations against a
30-trip minimum. Fails both arms of `evaluate_ibkr_evidence` at every cost
level: LCB not positive, drawdown over budget.

**Exit geometry was the leading hypothesis, and it is refuted.** A 27% win rate
with best +$72 against worst -$56 suggested the 5%-stop / 10%-target geometry
was cutting winners short. `auramaur ibkr-exit-study` searched 48
stop/target/momentum-exit configurations, ranked on trips entered before
2025-01-01, and scored the winner once on trips entered after:

- the best-on-train configuration **is the deployed one** — the search found
  nothing better to pick;
- **no configuration reached a positive train LCB**; the best was -$4.94, in
  sample, on the data it was chosen on;
- that winner scored **-$136.06** on test against a **+$56.22 median** across
  the 48 eligible configurations.

A train ranking that lands *below* the median out of sample is the signature of
a parameter surface with no exploitable structure. The defect is in the entry
signal or the premise, not the exits, and further tuning fits noise.

**Consequence for this spec.** Increments 1-3 put the IBKR books on the ladder
so their evidence could be scored. That work stands and is correct. But for
`global_etf` specifically the evidence is now in, six months early, and it says
the book should not be graduated — it should be replaced or retired. The
remaining ladder work matters for whatever strategy takes its place, not for
this one.

**Fidelity trap, worth remembering.** `IBKRMultiAssetPaperBook` calls
`get_daily_bars(spec)` with its DEFAULT `duration="3 M"` — 61 bars. A replay fed
the full accumulated history unlocks `normalized_momentum`'s 120-session horizon
the live book can never see, and measures a *different, better* strategy. The
first run of the harness did this and reported -$101 instead of -$969.
`SIGNAL_WINDOW = 61` pins it, asserted on the lengths the signal functions
receive rather than on emergent trade counts.

## The entry signal carries no information, and the book is too small to trade

`auramaur ibkr-signal-study` measures the signal's information content rather
than searching thresholds — no selection, nothing to overfit. Forward returns
are EXCESS of the equal-weight universe that session, so market drift cannot
pose as skill, and the t-statistic samples only non-overlapping windows.

**Information coefficient, 41,370 signal/forward pairs, 35 instruments:**

| horizon | IC | t | deployed gate: excess (95% low) |
|---|---|---|---|
| 5 | -0.0074 | -0.57 | -0.010% (-0.041%) |
| 10 | -0.0008 | -0.16 | +0.013% (-0.030%) |
| 21 | +0.0087 | +0.68 | +0.069% (+0.006%) |
| 42 | -0.0291 | -1.31 | -0.045% (-0.140%) |

Zero at every horizon, with the sign flipping. `normalized_momentum` over 61
bars does not predict 10-session excess returns on this universe.

The quintile shape is consistent and it is **not momentum**: Q1 (most negative
momentum) has the HIGHEST forward excess return at every horizon (+0.61% at
h=42), Q3 (the middle) the worst (-0.46%), Q5 recovering modestly. The book buys
momentum >= 0.25 — roughly Q4/Q5 — so it is not merely uninformed, it is
systematically avoiding the best-performing bucket. That U-shape is an
observation on one universe over one 5-year window with a plausible
compositional explanation, and it is NOT a strategy proposal; turning it into
one needs its own pre-registered out-of-sample test.

**Why the book loses is arithmetic, not signal.** Splitting the replay:

| trips | gross | commission | net |
|---|---|---|---|
| 463 | **-$42.74** | **-$926.00** | -$968.74 |

Gross is flat. **96% of the loss is commission.** `_commission` charges
`max(1.0, min(notional*0.001, 10.0))`, and at a $5,000 budget across 6 slots a
position is ~$600-830 — so the $1 MINIMUM binds, at ~12-17bps per leg, ~30bps
round trip, against a gate edge of 1-7bps that is not statistically distinct
from zero. The book cannot size out of the minimum-commission regime because
one position at the marginal rate would need ~$1,000+ and the whole budget is
$5,000.

**So three separate things are true**, and only the first was suspected:

1. exits are not the problem (48 configurations, none with a positive train LCB);
2. the entry signal has no information at any horizon tested;
3. even with a perfect signal, 463 round trips at this position size costs $926
   — the turnover/size economics are broken independently of the strategy.

Fixing commission alone reaches break-even, not profit, because gross is flat.
This book needs a different signal AND a different turnover profile; it is not
a tuning problem.
