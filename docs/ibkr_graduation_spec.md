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
- `market_outcomes` written when a forecast resolves, clearing the INNER JOIN
  in `_prospective_stats` that made every IBKR decision invisible.

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
