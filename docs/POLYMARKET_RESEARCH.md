# Polymarket research and graduation contract

All new mechanisms in `auramaur/research/polymarket_strategies.py` are
paper-only evaluators. They create `strategy_evaluations`; they do not place an
order or bypass the execution gateway.

## Evaluation standard

- Collapse repeated signals and ledger rows to one observation per market;
  cluster related markets by event when event-group metadata is available.
- Freeze parameters before each test window.
- Use chronological expanding walk-forward folds and retain a final untouched
  holdout.
- Evaluate executable bid/ask prices, actual per-market fees, partial fills,
  maker fill rates, rebates, and locked-capital time.
- Require at least 100 independent resolved markets and a positive one-sided
  95% lower confidence bound on mean cost-adjusted P&L.
- Forecast strategies must improve Brier score or log loss over the
  contemporaneous market and show positive executable closing-line value.
- Maker and taker variants are separate strategy cells.

`decision_snapshots` records the order price before submission. The runtime
adds 1-minute, 5-minute, 1-hour, and 24-hour bid/ask marks to
`decision_marks`. Rejected and unfilled decisions remain in the dataset so
fill-selection bias can be measured.

## Evaluators

- `complete_set_arb`: full-partition buy-all/sell-all executable bounds.
- `constraint_arb`: implication, mutual-exclusion, and partition violations;
  semantic verification confidence must be at least 95%.
- `source_latency`: fresh authoritative-source transitions not yet incorporated
  into the CLOB price.
- `normal_bin_probability`: source-native scheduled-release nowcasts.
- `maker_rebate`: spread plus rebate less adverse selection, inventory, and
  cancellation-failure costs.
- `flow_confirmation_multiplier`: bounded confirmation or veto only; flow can
  never originate a position.

Live startup performs Polymarket's geographic eligibility check and fails
closed when eligibility cannot be confirmed. Do not bypass that control.
