# Maker observatory

Status: shadow-only prospective experiment. It has no order-placement interface,
and no observatory field is read by quote formation, selection, sizing, or
execution.

## Configuration lives outside the strategy it observes

The knobs are the top-level `maker_observatory:` section, NOT fields under
`market_maker:`. `ExecutionGateway._capture_decision` hashes
`settings.market_maker` into market_maker's `strategy_version`, and
`_prospective_stats` joins on the LATEST version — so an observatory knob
parked in that section would discard the observed strategy's accrued
prospective evidence and restart its holdout clock every time the instrument
was retuned. An instrument must not be able to re-version its subject.
`tests/test_maker_observatory.py::test_observatory_config_cannot_reversion_the_strategy_it_observes`
pins this.

## Cost on the quoting path

"Shadow-only" is a claim about what the observatory READS, not about what it
costs. `observe()` runs inline in `_quote_market`, inside the
`op_timeout_seconds` watchdog, and every statement takes the process-wide
`Database` serializer that ~30 pillar tasks share.

Markout resolution used to run there too, and 93% of `observe()`'s cost was
that scan: its predicate `unixepoch(?) - unixepoch(filled_at) >= ?` wrapped the
column in a function, so every refresh walked every retained fill for the
market and probed the markouts index once per row. This matters beyond ordinary
latency — slow quotes get picked off, so an observatory that added seconds to
the quote path would have CAUSED the adverse selection it exists to measure.

Resolution now runs on its own bot task, `maker_observatory`, at
`resolve_interval_seconds`. Measured on the same harness and the same
synthetic volumes, per 5-market quoting cycle:

| retained fills / market | before   | after   | offline resolver pass |
|-------------------------|----------|---------|-----------------------|
| 0 (day 0)               | 21.4 ms  | 13.1 ms | 20.6 ms               |
| 40k (day 7)             | 744.6 ms | 13.9 ms | 24.0 ms               |
| 121k (day 21)           | 2.21 s   | 13.6 ms | 21.2 ms               |
| 259k (day 45)           | 4.49 s   | 13.1 ms | 27.3 ms               |

(Three runs; the before column varied 21–35 ms / 0.74–1.03 s / 2.21–2.61 s /
4.49–6.10 s with machine load, the after column never left 13–24 ms at any
volume. The resolver pass is measured against a realistic backlog: one
60 s interval at two fills per 30 s cycle, i.e. four newly due fills.)

The quoting cycle is now flat in retained history — what is left of
`observe()` is `compute_maker_features` (0.14 ms), two indexed volatility
windows and one INSERT, three serialized round-trips in steady state instead
of six. The resolver pass is flat too, and is bounded by outstanding work
rather than by history: `maker_observatory_fills.marks_pending` drives a
partial index (`idx_maker_fills_pending`) holding only fills that still owe a
mark, and the range predicate compares `filled_at` against a precomputed
timestamp instead of wrapping it in `unixepoch()`. `EXPLAIN QUERY PLAN`:

```
SEARCH f USING INDEX idx_maker_fills_pending (filled_at<?)
CORRELATED SCALAR SUBQUERY 1
SEARCH o USING COVERING INDEX idx_maker_obs_market_time (market_id=?)
```

against what it replaced, which could seek to the market and then had to walk
every retained fill in it:

```
SEARCH f USING INDEX idx_maker_fills_market_time (market_id=?)
CORRELATED SCALAR SUBQUERY 1
SEARCH m USING COVERING INDEX sqlite_autoindex_maker_observatory_markouts_1
       (fill_id=? AND horizon_seconds=?)
```

`test_the_due_fill_scan_seeks_an_index_instead_of_scanning_history` asserts
the plan against the shipped SQL.

Moving the scan cannot change what it concludes. A mark is taken from the
first observation at or after `filled_at + horizon` — exactly the book the
inline scan used, because the inline scan ran on the observation that first
found the fill due. Resolving every cycle and resolving once an hour therefore
write identical rows; only the wall-clock moment of the INSERT differs.
`tests/test_maker_observatory.py::test_offline_resolution_reproduces_the_inline_marks`
pins that equivalence, including the late/invalid case.

Storage growth is unchanged and still driven by synthetic paper fills (two per
market per 30 s cycle, ~1.3M rows at 45-day retention), which by this
document's own evidence contract can never be promoted. Shorter retention and
not persisting synthetic fills remain open operator decisions; they are no
longer latency-critical.

## Research question

Can L1 microprice skew, top-five depth imbalance, time-decayed signed aggressor
flow, midpoint-change volatility, and quote persistence identify maker fills
with adverse post-fill markouts?

The observatory records hypotheses; it does not assume those features are
signals. In particular, L1 microprice is kept distinct from top-five depth
imbalance, and reward eligibility remains NULL until trustworthy per-market
reward parameters are available.

## Prospective contract

The feature schema, horizons, and permitted mark lateness are serialized and
hashed into a strategy version. A new definition creates a new version and a
new seven-day warmup. Warmup data fixes each feature's median threshold; only
later holdout fills score the high-versus-low markout effect.

A fill can count toward promotion only when all of these are true:

- it is live and has credible evidence (venue_fill, book_cross, or
  trade_through);
- it belongs to the prospective holdout;
- its mark is taken no more than 45 seconds after the requested horizon;
- the configured horizon has a valid mark.

Synthetic resting paper fills are retained to test plumbing but can never count
as alpha evidence.

## Mark semantics

- Bid: future midpoint minus effective YES fill price.
- Ask/NO: effective YES sale price minus future midpoint.
- Positive is favourable to the maker; negative is toxic.
- Target time, actual mark time, lateness, and validity are immutable.
- The first observed book at or after a horizon is used. A late post-restart
  book is retained and visibly invalid instead of masquerading as a timely
  mark.
- An interrupted resolver pass leaves due marks unresolved for the next pass;
  it never fabricates one. `marks_pending` is cleared only in the same
  transaction that writes a fill's last mark, and every mark is
  `INSERT OR IGNORE` against `(fill_id, horizon_seconds)`.
- Each pass takes at most `resolve_batch_fills` fills, oldest first, and skips
  fills whose market has not been observed as far as their earliest horizon —
  those cannot be marked yet at any horizon, and without that guard a market
  that left the maker's five would strand fills at the head of an oldest-first
  queue and starve every newer mark until retention pruned them. They stay
  pending and are marked (late, `is_valid=0`) if the market returns.
- Paper and live fills are never pooled.
- Report windows are based on fill time, not the later mark time.

## Reporting

Run: auramaur maker-observatory --days 21

The report shows total fills, valid-mark completeness, credible holdout marks,
independent markets, unweighted and size-weighted markouts, market-clustered
bootstrap intervals, toxicity, frozen-threshold feature effects, sampled quote
coverage, and explicit promotion blockers.

Sampled quote coverage means an active quote was present at an observatory
sampling instant. It is deliberately not described as exchange-certified
uptime or reward-qualified time.

## Promotion gate

Every configured horizon must have at least 100 credible holdout marks across
at least five markets, at least 95% timely-mark completeness, and a positive
lower clustered-confidence bound. A candidate feature must additionally show a
stable holdout effect and acceptable quote-coverage and fill-rate trade-offs.

Passing this gate does not authorize a quote change. Any quote policy informed
by these findings is a separate, frozen, prospectively evaluated experiment.
Raw observations are retained for 45 days, pruned on the first cycle after
startup and every 24 hours thereafter (wall-clock, so a stack that restarts
more often than daily still prunes).

## Flow coverage

`signed_flow` is read from `OrderFlowTracker`, which is fed only by the
websocket price-monitor's `on_trade` for the first 20 discovered markets and
keeps at most 50 trades per market. The maker's own five markets are selected
by spread and frequently are not in that set.

The column is therefore **nullable**, and `NULL` means "no trade feed ever
reached this market" — a different fact from "flow was balanced", which is a
genuine 0.0. Coercing the first to the second is the 2026-07-29 qwen3 failure
mode, where a metric incapable of moving looked healthy for a week.

- `OrderFlowTracker.signed_flow` returns `None`, never 0.0, when it has seen no
  trade under any of the market's feed keys (market id, condition id, YES token
  id). A market the feed HAS reached but which saw no aggressive trade inside
  the 300 s window still scores 0.0: that is a measurement.
- `maker_observatory_feature_report` excludes NULLs from the warmup median and
  from both holdout buckets, and every row reports `covered_n` / `marks_n` /
  `coverage` so an absent result cannot be read as a negative one.
- `maker_quote_coverage` reports `flow_samples` (`COUNT(signed_flow)`, which
  skips NULLs by definition) and `flow_coverage`.
- `auramaur maker-observatory` prints a Measured column per feature, the
  trade-feed coverage line, and a warning whenever coverage is below 100%.

## Known measurement gaps

- A fill whose `observation_id` is NULL (its `observe()` raised) is dropped by
  the summary's inner join, so it is invisible rather than counted as an
  incomplete mark.
- Flow coverage is reported, not fixed. Subscribing the websocket to the
  maker's own selected markets is a separate change to the price-monitor task.
