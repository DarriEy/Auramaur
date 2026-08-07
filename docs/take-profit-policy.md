# Take-profit policy

The shared portfolio monitor evaluates exits in this order: stop loss,
configurable trailing stop, fee-net lifecycle profit target, binary-market
capital-efficiency rules, then dust cleanup.

## Only the exit fee is charged

The binary profit gate nets off the **exit** fee alone. That is not an
approximation — it is what the accounting does. `broker/pnl.py` realizes
`(fill.price - old_avg_cost) * size - fill.fee` on the SELL branch and builds
the `pnl_ledger` event only there; the BUY branch stores
`total_cost = price * size`, so `avg_cost` is fee-exclusive and the entry fee
never reaches the ledger at all. A gate that charged both legs demanded a cost
the books never record, holding every winner past its target by the entry
leg's drag. The entry fee is still computed and reported on `ExitEconomics`
as a diagnostic, so calibration can see it without it binding the decision.

## Position age

Age resolves in three steps, most specific first:

1. the oldest fill still contributing to the current
   `(market, token, paper/live)` inventory — old round trips and the opposite
   token do not age a re-entry;
2. failing that, the market's first `trades` row, scoped to the same book.
   `trades` has no token column, so this leg cannot be token-scoped; it is
   coarser than the fills path but strictly more information than nothing;
3. failing that, "just entered".

Step 2 exists because step 3 is not free. Positions predating the fills ledger,
and live rows a venue mirror created without ever writing a fill, have no
reverse-inventory answer — and an unknown entry time pins `fraction_remaining`
at exactly 1.0 on every tick, freezing the target in its widest band and making
the near-expiry band permanently unreachable. The fills comparison also carries
a small size tolerance, because summed fill quantities land a few ULPs below
the stored size often enough to erase a position's whole ancestry.

`cost_basis` is deliberately not consulted here. It is the authoritative
holdings table, but its only timestamp is `updated_at`, stamped on every write:
it records when inventory last moved, not when it was entered.

## Trailing stop

`trailing_stop_activation_pct` and `trailing_stop_giveback_fraction` are both
calibratable, and **0 in either disables the tier**. Without that guard zero
meant the opposite of off: a zero activation arms the stop against every
non-negative peak, and a zero giveback reduces the test to `peak > current`, so
"disabled" would have sold every position on its first adverse tick.

## Decision telemetry

Every profit-target evaluation contributes an `exit_decisions` observation
whose `policy_action` describes this policy component, not necessarily the
later capital-efficiency or time-decay decision.

The observations are accumulated during the cycle and written **after** every
exit has been decided, as a single `executemany`. This is deliberate: the HOLD
branch observes every non-exiting position on every tick, so writing inline
took the shared serialized write lock hundreds of times per cycle on the path
that closes positions. A failure in the batch is contained and ignored —
telemetry must never be able to keep a position open.

Rows are pruned to `execution.exit_decision_retention_days` (default 14) by a
bounded per-cycle delete on the indexed time column, following the same pattern
as `candidate_dispositions`. HOLD rows are kept rather than dropped because
they are the counterfactual: a calibration that sees only exits cannot say what
a different threshold would have done. Lower the retention if the volume
outweighs that.

These records are measurement data, not permission to tune against the same
sample. Run `python scripts/calibrate_exit_policy.py auramaur.db`; it uses an
oldest-70% training/newest-30% holdout split and refuses to recommend changes
below the minimum sample. Threshold changes remain manual and reviewable in
`config/defaults.yaml`.

## Venue semantics

Asset semantics remain intentionally distinct: binary venues use the
`rate * price * (1-price)` taker-fee model on the executable exit leg, IBKR
uses the executable bid and its two explicit commissions (both known exactly,
so both are charged), and Kraken retains its percentage fee/slippage model.
