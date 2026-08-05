# Experiment capacity and wind-down policy

Paper trading protects capital, not attention. Each enabled paper strategy
family consumes candidate flow, API or data-source budget, monitoring surface,
and a future adjudication. Auramaur therefore caps concurrent paper strategy
families at 12. The tracked configuration currently uses 11 slots.

A family consumes one slot when both `enabled` and `paper` are true. Model or
venue arms inside one family share its hypothesis, runtime, and review, so they
do not consume separate slots. Read-only monitors and structurally live services
are outside this count. `Settings.active_paper_trials` is the canonical
inventory; startup fails closed if it exceeds
`experiment_capacity.max_concurrent_paper_trials`.

Every new trial must record, next to its enable flag:

- a start date and a review date or observable sample threshold;
- success and rejection criteria;
- the safe wind-down lever; and
- the condition under which a follow-up trial may be started.

At the review date, stop new observations first. Existing positions continue
through their normal exit/settlement path and historical registrations and
ledger rows remain untouched. Record the verdict before re-enabling; a changed
hypothesis or implementation begins a new dated trial and evidence lineage.

## 2026-08-05 capacity release

Two arms had continued past their preregistered two-week clocks. New collection
is now disabled for `settlement_arb` (clock ended 2026-07-10) and the Kalshi arm
of `resolution_lens` (clock ended 2026-07-17). This is a lifecycle action, not
an inferred performance verdict: their evidence is retained for adjudication.
