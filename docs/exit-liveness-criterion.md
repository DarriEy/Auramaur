# Exit-liveness readiness criterion

`readiness.check_exit_liveness` answers one question per strategy book:
**it is still taking entries — is anything still leaving?**

## The failure it exists to catch

The live prediction-market exit loop once raised on every tick. The raise was
caught by a broad `except Exception: log.debug(...)`, so the first position of
each tick aborted the whole tick, and for an extended period the bot took
entries and never exited. Readiness reported clean throughout, because
`check_cycle_health` scores cycles by log level and `debug` is not in
`_ERROR_LEVELS`.

The fix for that bug (#420) addressed the specific attribute error and raised
those handlers to `error`. This criterion addresses the **class**. It fires
regardless of cause — a future exception, a config mistake, a stuck lock, an
exchange adapter silently refusing sells — because it watches outcomes rather
than mechanisms. Nothing about it depends on the exit path logging anything at
all.

## What it measures

A cell is `(venue, book, mode)`. Over a lookback window:

| side | source | why |
| --- | --- | --- |
| entries | `trades` rows with `side='BUY'`, excluding `cancelled`/`rejected` | the gateway's mirror is the only row carrying both the venue and the strategy that *decided* to open |
| exits | `pnl_ledger` rows with `kind IN ('sell','settlement')` | the ledger attributes a realization to the strategy that **opened** the position; the trades mirror attributes the SELL to the exit path itself (`strategy_source='exit'`), which cannot be matched back to a book |

A cell with entries at or above the floor and **zero** exit events FAILs,
naming the cell and both counts. A cell with entries and some exits PASSes.

### Deliberate exemptions

* **Settlements count as exits.** A position can legitimately leave via
  resolution rather than a sell. A long-dated book goes weeks with no SELL
  while positions resolve; counting sells only makes the criterion fire
  permanently on `long_horizon`.
* **`commission` rows are not exits.** They are cash adjustments booked
  against a still-open position. Money moving is not a position leaving, and
  treating them as exits would let an IBKR book mask a stalled exit path.
* **Dormant books never FAIL.** A cell with no entries in the window is not
  evaluated at all. A strategy that stopped trading is not a broken one.
* **Below the entry floor is INSUFFICIENT_DATA.** One entry proves nothing —
  a single position can legitimately be held across the window.
* **A cell that has never produced an exit is not judged.** "Stopped"
  presupposes it was running; a brand-new book's first realization is
  genuinely weeks away. Such cells are reported as *not yet judgeable* in the
  criterion's `detail` rather than silently dropped, so a book that never
  exits at all is still visible to an operator.
* **Attribution buckets are not books.** `''`, `exit`, `order_monitor`,
  `legacy_unattributed`, `adopted_unknown`, `phantom_unattributed` and
  `venue_unattributed` are entry-attribution fallbacks. `broker/ledger.py`
  refuses to credit any of them as the entrant, so their exit count is zero
  *by construction* — judging them is a permanent structural false alarm, not
  a signal.
* **Paper and live are judged separately.** Paper exits kept working
  throughout the incident, which is exactly why nothing looked wrong in
  aggregate. Collapsing the two modes turns the FAIL back into a PASS.

## Calibration

The thresholds were chosen by replaying the criterion over the whole trade
history at six-hour steps and counting the distinct episodes in which it would
have fired, before the outage and during it.

* **Window = 7 days.** Historical false alarms fall as the window widens and
  bottom out at 7 days; no live cell fires anywhere before the outage at 7
  days or longer, and widening further buys nothing. Shorter is better for
  latency, since detection cannot happen before `last_exit + window` — so 7 is
  the shortest window that reaches the false-alarm floor. Windows as long as
  the outage itself never fire at all: the pre-outage exits stay inside the
  window for its entire duration.
* **Entry floor = 3.** At a 7-day window the false-alarm count is already at
  its minimum by 3 entries and does not improve at 4 or 5, while raising the
  floor delays detection by one to two days. Three independent entry decisions
  in a week with zero realizations of any kind is a real signal.

The criterion keeps this window regardless of `--days`. The entry floor is
only meaningful against the window it was measured for, and widening the
report must not quietly change what the criterion means; the `threshold`
string always states the window actually used.

Both knobs are operator-tunable under `monitoring:` in `config/defaults.yaml`
(`exit_liveness_window_days`, `exit_liveness_min_entries`). That section is the
health-check contract, **not** a strategy section: `strategy_version` hashes
only `{strategy_source, that strategy's own config section, risk.min_edge_pct
/ max_spread_pct / confidence_floor}` (see
`broker/execution_gateway.py::_capture_decision`), and `monitoring` is never
resolved as a strategy section. Retuning these therefore cannot reset a 14-day
graduation clock.

### Honest limits

No threshold clears the outage while firing *never* before it. Replayed over
the full history at the chosen settings the criterion produces a small number
of pre-outage episodes, **all of them paper cells and none live**. They are
not obviously false either: each is a book that took double-digit entries
across more than a week with no realization of any kind — the same shape the
criterion is built to report. The residual risk is a paper cell crying wolf,
which is why the failure names the cell and the mode rather than reporting a
single system-wide verdict.

Two live cells that were also entering during the outage are *not* caught,
because both were new enough to have no prior exit at all. That is the
"never exited" exemption doing its job: from outcomes alone, a book that has
never realized is indistinguishable from one whose first realization has not
come due. They surface as *not yet judgeable* instead.
