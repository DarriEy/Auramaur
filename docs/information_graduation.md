# Information graduation

Official NWS, BLS, BEA, Congress.gov, and EIA observations begin in shadow
mode. They are persisted with source health and point-in-time lineage, but are
withheld from production forecasts.

Cells are keyed by source, category, horizon, and event type. Deterministic
control/treatment assignment and paired forecast/contribution storage are
implemented in `auramaur.information_graduation`. Promotion requires source
health plus positive incremental Brier score, log loss, and paper P&L.

## Explicitly deferred: paired-forecast worker

The automatic worker that spends two model calls to generate control and
treatment forecasts is intentionally not enabled in this PR. Enabling it would
change the model-call budget and operational cadence. A follow-up PR must add:

1. an explicit daily paired-call budget;
2. immutable evidence manifests for both arms;
3. a scheduled paper-only worker;
4. cost-inclusive paper P&L attribution.

Until that worker ships, all new sources remain at zero influence and cannot
graduate accidentally. Operators can inspect registered cells with
`auramaur information-graduation`.
