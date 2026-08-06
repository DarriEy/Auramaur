# Take-profit policy

The shared portfolio monitor evaluates exits in this order: stop loss,
configurable trailing stop, fee-net lifecycle profit target, binary-market
capital-efficiency rules, then dust cleanup.

Position age comes from the oldest fill still contributing to the current
`(market, token, paper/live)` inventory. Old round trips and opposite tokens do
not affect a re-entry's lifecycle target. A missing fill history is treated as
a new position, which widens rather than prematurely tightens the target.

Every profit-target evaluation writes an `exit_decisions` observation whose
`policy_action` describes this policy component, not necessarily the later
capital-efficiency or time-decay decision. These
records are measurement data, not permission to tune against the same sample.
Run `python scripts/calibrate_exit_policy.py auramaur.db`; it uses an oldest-70%
training/newest-30% holdout split and refuses to recommend changes below the
minimum sample. Threshold changes remain manual and reviewable in
`config/defaults.yaml`.

Asset semantics remain intentionally distinct: binary venues use the
`rate * price * (1-price)` round-trip taker-fee model, IBKR uses executable bid and explicit
commissions, and Kraken retains its percentage fee/slippage model.
