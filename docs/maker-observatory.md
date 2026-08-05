# Maker observatory

Status: shadow-only prospective experiment. It has no order-placement interface,
and no observatory field is read by quote formation, selection, sizing, or
execution.

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
Raw observations are retained for 45 days.
