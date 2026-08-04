# Plan: migrate legacy multi-statement DB writers onto `transaction()` (issue #353)

Planned 2026-08-04. Successor to `docs/plans/db-contention-plan.md`
(implemented 2026-07-20; phase 5 reverted in 18cb6cf after the crit-4
held-long soak failure, later re-landed hardened in 3556a67/#381). Under
the autocommit connection (`isolation_level=None`, database.py:61) plus
the no-op `Database.commit()`/`rollback()` (database.py:1507-1529, commit
eed51b8), every legacy `execute()` is individually durable and
individually committed — so a crash mid-sequence in a multi-statement
writer leaves partial writes. The trailing `commit()` calls in legacy
writers are now dead code; they neither batch nor commit anything.

## Ground rules (hard constraints, all verified in code/history)

1. **Tight spans only.** The phase-5 revert (18cb6cf) was caused by
   transactions held too long. `database.transaction_held_long` fires at
   >250ms (database.py:251-258). Classify/compute/network/LLM **before**
   `BEGIN`; inside the span: writes plus only the DB reads a
   read-modify-write genuinely requires (the `pnl_fill` model,
   pnl.py:68-109). Never a network or LLM await inside a span.
2. **Only the context manager.** `async with db.transaction(owner=...)`
   — never raw `BEGIN`/`COMMIT`/`ROLLBACK` through `db.execute()` in
   application code. The context manager owns orphan recovery and forced
   rollback (the 2026-07-25 45-minute wedge fix, commit 096054e,
   database.py:141-258). Adding any manual transaction path reintroduces
   the wedge class.
3. **Distinct `owner=` per writer**, named `<module>.<helper>` (e.g.
   `owner="ibkr_etf_paper.fill"`), because held-long/queue-wait telemetry
   keys on it. Existing owners: `lineage`, `position_sync`, `pnl_fill`,
   `execution_gateway`, `heartbeat`, `data_edge`,
   `candidate_dispositions`, `experiment_*`, `peak_prune`.
4. **Never wrap a gateway/PnL/calibration call.** `transaction()`
   re-entrancy JOINS the outer span (database.py:122-124): wrapping
   `gateway.submit()` or `pnl.record_fill()` or
   `calibration.record_prediction()` would silently lengthen the outer
   transaction across their work (and `submit()` does network I/O).
   Pillar spans cover only the pillar's own post-fill statements.
5. **Retry policy.** BEGIN-phase transient errors ("cannot start a
   transaction", "database is locked") occur before any write and are
   safe to retry (lineage_observer.py:99-120 is the house model: 3
   attempts, backoff, per-item fallback). Do **not** add blanket retries
   around non-idempotent batches — a COMMIT whose success is ambiguous
   must not be replayed (the record_fill lesson, database.py:74-79).
6. House conventions: type hints, dated WHY comments, tests for anything
   risk-adjacent (CLAUDE.md).

## Classification rubric (applied per writer, not per file)

- **A — genuine logical batch.** ≥2 statements forming one unit whose
  partial completion corrupts state (fill without ledger row, position
  without thesis row, pair entered without its traded-at marker). →
  Adopt `transaction()`.
- **A2 — benign-direction batch.** Multi-statement, but ordering +
  idempotent first statement make the crash-direction partial harmless
  (the `markets` stub + `signals` insert pairs). → Adopt while in the
  file; cheap, and removes the class from future audits.
- **B — independent single-statement writes.** Already atomic and
  durable under autocommit. → No transaction. Delete the dead trailing
  `commit()` only in files touched anyway (with a dated comment); do not
  open untouched files just for that.
- **C — already safe / out of scope.** Idempotent by construction
  (`INSERT OR IGNORE`/`OR REPLACE` upserts, self-healing per-cycle
  mirrors), idempotent DDL, or not on the shared connection at all.

## Verified per-file classification (all 18 survey files)

**False positives — no migration (classification C/B):**

| File | Finding |
|---|---|
| `treasury/transfers.py` | **Out of scope entirely.** Separate ledger file (`transfer_ledger.sqlite3`), its own synchronous `sqlite3` connection (transfers.py:86), and the one batch that matters (`_reserve`, cap-check + insert) already uses explicit `BEGIN IMMEDIATE` (transfers.py:157-177). `Database.transaction()` does not apply. Matches the contention plan's inventory note ("separate file — fine"). Action: dated comment marking it audited, nothing else. |
| `strategy/interim_manager.py` | All five writers are single UPDATE-or-INSERT + dead commit (lines 172-177, 188-191, 259-263, 266-270); `_auto_propose` (344-360) is a loop of **independent** proposal INSERTs — partial completion just means fewer auto-proposals. All B. Live-money file, zero behavior change: classification-only. |
| `broker/onchain.py` | `_record_attempt` (349-374), `_mark_confirmed` (378-382), rejected-update (535-539) — all single-statement. B. |
| `treasury/kraken_pillar.py` | Singles/upserts/deletes throughout (106-128, 175-209, 1244, 1266-1277) — B. The portfolio-mirror loop (659-698) is a self-healing per-cycle rebuild wrapped in a visibility-only try/except — C (optionally batch via one span, low value). Directional fills route through `pnl.record_fill`, which already owns `owner="pnl_fill"`. |

**Genuine batches — adopt `transaction()` (classification A):**

| File | Writer | Batch | Owner label |
|---|---|---|---|
| `strategy/ibkr_etf_paper.py` | `_fill` (391-425) | fill row + commission ledger row + position insert, or realized-trade ledger row + position delete. The canonical orphan genus; currently 13 executes to 1 cycle-end (dead) commit. | `ibkr_etf_paper.fill` |
| `strategy/ibkr_multiasset_paper.py` | `_fill` (216-311) | paper fill + commission ledger + position upsert / trade ledger + round-trip insert + position update-or-delete. (Round-trip already `OR IGNORE` on unique `exit_fill_ref` — keep.) | `ibkr_multiasset.fill` |
| `strategy/agent_trader.py` | entry bookkeeping (`_record_position` portfolio upsert at 695-710 + theses INSERT at 645-654) | position row without its thesis row = held position no arm settles in its book. Restructure so both land in one span after `gateway.submit` returns; calibration call stays outside. Also fix the `_ensure_schema` ALTER+backfill pair (181-188): if the ALTER lands and the UPDATE backfill doesn't, the next run's ALTER raises into `except: pass` and the backfill is skipped forever. | `agent_trader.entry` |
| `strategy/cross_venue_arb.py` | pair bookkeeping (`_record_leg` ×2 at 367-368 + `traded_at` update at 369-372) | crash before `traded_at` leaves `_already_traded` false → the pair can be re-entered → duplicate exposure. One span for both legs' signals rows + verdict update, after `submit_paired` returns. | `cross_venue_arb.entry` |
| `strategy/entailment_arb.py` | `_record_leg` (652-685: markets + signals + portfolio) and the `traded_at` update (632-637) | same genus; wrap the 3-statement helper; fold `traded_at` into the same entry span where the call sites allow. | `entailment_arb.entry` |
| `strategy/resolution_lens.py` | `_record_position` (743-774: markets + signals + portfolio) | markets+signals landing without the portfolio row after a real fill = unmirrored position the resolution tracker never settles. | `resolution_lens.entry` |
| `strategy/term_structure.py` | `_read_family` curve loop (504-515) | one LLM read produces N strike rows; a partial curve poisons the strike-set cache comparison. LLM call completes before the loop — tight span is free. | `term_structure.curve` |

**A2 — adopt while in the file:**

- The eight identical `_persist_signal` helpers (markets `OR IGNORE`
  stub + signals INSERT): `bias_harvest.py:486`, `econ_indicator.py:257`,
  `informed_flow_pillar.py:271`, `long_horizon.py:496`,
  `vol_anchor.py:430`, `weather_temp.py:176`, `agent_trader.py:665`,
  `term_structure.py:967` — owners `<pillar>.signal`. Their
  `_record_position` counterparts are single upserts (B) except
  resolution_lens (A, above).
- `strategy/engine_cycle.py`: the markets+signals pairs inside
  `_persist_cycle_dispositions` (520-543) and `_execute_candidates`
  (860-884) — owner `engine_cycle.signal`.
- `term_structure._trade_curve` observations loop (823-848): independent
  telemetry rows (B), but the data is precomputed, so collect rows and
  land them via one `executemany` in a span — optional, do it while in
  the file.

## Phases

Each phase is one commit; deploy is `docker compose build && up` by the
operator; rollback is `git revert` + rebuild. No schema changes anywhere
in this plan, so images roll both directions against the same DB file.

**Phase 0 — Classification freeze + conventions (S).**
Land this document; add one convention line to CLAUDE.md's architecture
section: multi-statement writes on the shared `Database` use
`async with db.transaction(owner="<module>.<helper>")`; never raw
BEGIN/COMMIT; never a network/LLM await inside a span; never wrap
gateway/PnL/calibration calls. Record the reproducible survey command
from the issue comment so the audit can be re-run at closeout.
Rollback: revert the doc commit — no runtime surface.

**Phase 1 — Shared test scaffolding (S, ~80 LOC).**
A reusable pytest helper (e.g. `tests/txn_helpers.py`):
- `failing_on(db, sql_prefix)` — wraps `Database.execute` to raise
  `sqlite3.OperationalError("injected")` when the target statement of a
  batch is reached (the crash-mid-batch injector).
- `transaction_spy(db, events)` — wraps `Database.transaction` to record
  `(owner, "begin"/"end")` events, for ordering and owner-label
  assertions.
Idioms already proven in `tests/test_database_transaction.py`
(`test_rollback_discards_only_its_own_writes`,
`test_failed_commit_does_not_strand_an_open_transaction`).
Rollback: revert; test-only.

**Phase 2 — Paper-book rehearsal (M, 2 files + tests).**
`ibkr_etf_paper._fill` and `ibkr_multiasset_paper._fill`. Highest
genuine-batch value, zero live-money blast radius (paper ledgers), and
the exact `_record_position`-style shape the issue names — the rehearsal
for phase 5 mechanics and review. Remove the now-dead trailing
`commit()` in touched helpers with a dated comment. Per-writer tests
(pattern below).
Rollback: revert the commit. Worst case regression is paper-book
bookkeeping.

**Phase 3 — Non-money bookkeeping sweep (M, ~10 files, mechanical).**
The eight `_persist_signal` helpers, `engine_cycle`'s two inline pairs,
`term_structure._read_family` (A) + `_trade_curve` observation batching,
and the `agent_trader._ensure_schema` ALTER+backfill fix. One
parametrized test can cover the mirror pillars since the helpers are
byte-similar. These all run pre-risk-gate or as telemetry; a fault here
never strands money state.
Rollback: revert. Signals/markets writes revert to autocommit singles —
the pre-migration behavior exactly.

**Phase 4 — Soak gate (S, ops only; blocks phase 5).**
Minimum 48h paper soak on the deployed image (the db-contention-plan
phase-5 precedent), judged on log telemetry:
- **Zero** `database.transaction_held_long` events for any owner
  introduced in phases 2-3 (this is crit-4, the criterion that reverted
  the last attempt).
- **Zero** `database.transaction_orphan_rolled_back`,
  `transaction_forced_rollback`, `transaction_begin_failed`.
- No growth in `database.transaction_queue_wait` /
  `statement_queue_wait` warning rates vs the pre-deploy week.
- The contention plan's standing metric (locked-errors/day per
  subsystem) not regressed.
Optionally revive the `auramaur-soak` container for a synthetic
write-storm against a **copy** of the DB (paper env, transfers unarmed,
live gates off — the compose file's container gates already default
safe). If revived, decommission it again at closeout or explicitly keep
it; do not leave it half-alive.
Rollback: n/a (no code). A gate failure reverts phase 2/3 commits and
returns to classification.

**Phase 5 — Live-money post-fill batches (M, 4 files + tests, extra
review).**
`agent_trader` entry pair, `cross_venue_arb` pair bookkeeping,
`entailment_arb._record_leg`(+`traded_at`),
`resolution_lens._record_position`. All are post-`submit()`
restructures: the network call completes, then one tight span lands the
pillar's own rows; calibration stays outside. `interim_manager`,
`kraken_pillar`, `onchain`, `transfers` ship
classification-comment-only changes here (dated WHY comments recording
the B/C verdicts so the next audit doesn't re-litigate them). Require a
second review pass on this commit; 7-day live verification window after
deploy (db-contention-plan verification standard), same telemetry
criteria as phase 4 plus zero money-path errors
(`gateway.fill_record_failed`, `order_monitor` fill errors).
Rollback: revert this commit alone; phases 2-3 stand independently.
Trigger: any held-long warning from a phase-5 owner in live, or any
orphan/forced-rollback event attributable to a new span.

**Phase 6 — Closeout (S).**
Re-run the survey command; every remaining
`≥3 execute / ≥1 commit / 0 transaction` file must now carry a dated
classification comment (B/C). Update `docs/plans/db-contention-plan.md`
status to crosslink this plan. Summarize on issue #353 and close.
Record the soak-container decision.

## Per-writer test pattern (required for every A/A2 adoption)

1. **Crash-mid-batch atomicity** — real `Database` on a tmp file;
   `failing_on(db, "INSERT INTO signals")` (i.e., the *second* statement
   of the batch); call the helper; assert the error surfaces per the
   helper's contract **and** zero rows exist in *any* table of the batch
   (no partial). This is the direct analogue of the injected-SQLITE_BUSY
   test the contention plan specified for record_fill.
2. **Tight-span ordering** — mock the network/LLM boundary
   (`gateway.submit`, `_call_model`) and use `transaction_spy`; assert
   every network event precedes the span's `begin` event, and that no
   awaited mock fires between `begin` and `end`. (Lineage of the
   "get_market before first db.execute" test from contention-plan
   phase 1.)
3. **Owner label** — assert the spy saw the writer's distinct `owner=`
   string.
4. **No-strand invariant** — after the injected failure, a subsequent
   unrelated `db.execute` succeeds immediately (proves the
   forced-rollback path cleared the connection; guards against ever
   reintroducing the 2026-07-25 wedge).

## Non-goals

No changes to `db/database.py` (the transaction machinery is complete
and battle-tested); no generalized write-queue actor (rejected in the
contention plan — lineage's queue remains the only queue); no schema
changes; no migration of `transfers.py` off its separate ledger file; no
changes to risk manager, paper gating, or kill switch; no repo-wide
dead-`commit()` sweep outside touched files; no dedup of the eight
mirror `_persist_signal` helpers into a shared base (worthwhile but a
separate, behavior-neutral refactor — keeping this migration mechanical
per file is what makes each phase trivially revertible).
