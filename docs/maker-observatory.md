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
`Database` serializer that ~30 pillar tasks share. Measured on an idle SSD, one
`observe()` is 6 serialized round-trips and:

| retained fills / market | observe() | 5-market cycle |
|-------------------------|-----------|----------------|
| 0 (day 0)               | ~7 ms     | ~38 ms         |
| 40k (day 7)             | ~236 ms   | ~1.2 s         |
| 259k (day 45)           | ~917 ms   | ~4.6 s         |

93% of that is `_mark_due`, which rescans every retained fill for the market on
every refresh. The growth is driven by synthetic paper fills (two per market per
30 s cycle = ~1.3M rows at 45-day retention), which by this document's own
evidence contract can never be promoted. Before this instrument runs on a
latency-sensitive live book, the operator should decide between a per-horizon
mark watermark, a shorter retention, and not persisting every synthetic fill.

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
- The first observed book after a horizon is used. A late post-restart book is
  retained and visibly invalid instead of masquerading as a timely mark.
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

## Known measurement gaps

- `signed_flow` is read from `OrderFlowTracker`, which is fed only by the
  websocket price-monitor's `on_trade` for the first 20 discovered markets, and
  keeps at most 50 trades per market. The maker's own five markets are selected
  by spread and frequently are not in that set, so this feature can be
  identically 0.0 for reasons that have nothing to do with flow. The column
  cannot currently distinguish "balanced flow" from "no feed", so a null result
  on it is uninterpretable rather than negative.
- A fill whose `observation_id` is NULL (its `observe()` raised) is dropped by
  the summary's inner join, so it is invisible rather than counted as an
  incomplete mark.
