# Kalshi hardening and experiment workflow

Kalshi remains paper-first. Live orders still require the environment gate,
configuration gate, and per-order `dry_run=False`; the kill switch and risk
gateway remain authoritative.

## Endpoints

Production uses `https://external-api.kalshi.com/trade-api/v2`, Kalshi's
current API domain (migrated 2026-07 from `api.elections.kalshi.com`, which
serves identical responses and remains a valid fallback). The demo host is
unchanged: `https://demo-api.kalshi.co/trade-api/v2`. If Kalshi retires a
domain, the host literal lives in `KalshiClient._init_api`.

## Fill booking for resting live orders

A live order that rests (`pending`) is not recorded as a position by the
entering strategy. The single-booking contract is: the order monitor records
the confirmed fill (`pnl_tracker.record_fill` → fills / cost basis), and the
next `sync_positions` venue snapshot materializes the `portfolio` row from
venue truth. Strategy-level re-entry is prevented meanwhile by the `signals`
table guard. Do not "fix" the entry gates to record pending orders — that
reintroduces phantom positions.

## Execution contract

- Current fixed-point (`*_fp`, `*_dollars`) fields are canonical. Legacy fields
  are compatibility fallbacks only.
- Prices are quantized using each market's `price_ranges[].step`; quantities
  retain Kalshi's 0.01-contract precision.
- Direction uses `outcome_side` and `book_side` before deprecated side/action
  fields.
- Every directional entry is checked against a fresh order book. Requested
  quantity is capped to executable depth and the limit becomes the marginal
  price needed to fill that quantity.
- The same check applies to paper entries, preventing graduation on impossible
  fills. Decision-time book measurements are stored in
  `kalshi_execution_samples`.
- A live `pending` order is not a position. Only confirmed fill quantities may
  update portfolio or cost basis.

## Accounting contract

- Position snapshots must consume every API cursor before local reconciliation.
  A failed page aborts before the deletion phase.
- Settlement booking reads fixed-point quantities, venue costs, result, revenue,
  and fees. Internal cost basis is preferred; venue cost is the recovery source
  when local cost is unavailable.
- Settlement P&L is `revenue - cost - fee`, and source references make booking
  idempotent.

## Experiment cells

Keep each strategy and parameter variant on a distinct `strategy_source`. Review
resolved net P&L, drawdown, Brier score, executable fill ratio, VWAP versus the
decision midpoint, and fee drag separately. Do not graduate using signal count
or synthetic fill count.

For informed flow, export `kalshi_execution_samples` alongside signal evidence
and resolution outcomes. Stratify by abnormal-size multiple, dominance, time to
close, category, and price band before changing uplift. Long-horizon entries
should be evaluated for maker execution separately from time-sensitive economic
release and settlement-arbitrage entries.

Before increasing live capital, require current-schema fixture tests, a complete
position/settlement reconciliation, no unknown-cost settlements, and positive
resolved net P&L after venue fees.
