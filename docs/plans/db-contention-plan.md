# Plan: eliminating chronic SQLite "database is locked" contention

Planned 2026-07-20 after a live investigation (~300 locked errors across 13
subsystems over 2 days; `/proc/locks` evidence; per-subsystem counts in the
git history of the web-dashboard branch).

Status: **implemented 2026-07-20** — core cd24872, phase 1 7833c34, phases
2+3 0897132, phase 4 73bca76, phase 5 3d44cb9. Each phase is its own commit;
rollback is "revert the commit". Deployment: phases 1–4 are low-risk; phase 5
(record_fill/gateway) wants the 48h paper soak below before the live image
picks it up. Verification metric (locked-errors/day per subsystem, 7-day
soak) still applies post-deploy.

## Root-cause model (verified in code)

**M1 — Write transactions held open across network awaits on the shared
connection.** `aiosqlite.connect()` uses default `isolation_level=""`
(`auramaur/db/database.py:20`): the first DML opens an implicit deferred
transaction that stays open until `commit()`. `KalshiClient.sync_positions`
(`auramaur/exchange/kalshi.py:897-1089`) upserts markets/portfolio/cost_basis
inside a per-position loop that awaits `get_market()` over the network between
writes (kalshi.py:944), committing once at the end (kalshi.py:1082) — one
write transaction held for N network round-trips. Every other connection's
write hits SQLITE_BUSY meanwhile. The Polymarket reconciler has a milder
version (`auramaur/broker/reconciler.py:134,216` network calls interleaved
with stub-market inserts at reconciler.py:264-271). This plausibly explains
the majority of the errors.

**M2 — Multiple writer connections in one process.** The bot holds at least
three write-capable connections: the shared `Database` (database.py:20,
busy_timeout=30000), the LineageObserver's second full `Database`
(`auramaur/lineage_observer.py:36-39`, built in `auramaur/composition.py:357-362`),
and call_budget's private sqlite3 conn (`auramaur/nlp/call_budget.py:72`,
fail-fast by design).

**M3 — External openers taking write locks or long read locks.**
- Web dashboard's long-lived mode=ro conn — fixed (per-cycle, commit 6290fd2).
- `auramaur/agentmcp/compare.py:86-92` opens the BOT's DB with the full
  read-write `Database` class from the MCP process — `connect()` runs WAL
  pragma + `executescript(TABLES)` DDL + potential migrations against the
  live trading DB on every `compare_to_bot` call. Worst external opener.
- ~28 CLI `Database()` sites (cli/run.py:82,103; diagnostics; maintenance;
  reporting; redeem.py:361; kraken.py:163,182) run the same DDL connect()
  against the live file on every invocation.

**Bonus correctness hazard (fix in Phase 2/5):** there is NO asyncio lock
around the shared connection's transactions. ~30 pillar tasks interleave
statements on one connection with implicit deferred transactions: task B's
`commit()` (e.g. nlp/cache.py:87) commits task A's half-written transaction,
and `PnLTracker.record_fill`'s error-path `rollback()`
(`auramaur/broker/pnl.py:70-72`) can roll back another task's uncommitted
writes. This is the real reason record_fill is "not idempotent
mid-transaction" (database.py:35-38).

## Connection inventory

| # | Opener | Where | Mode | Notes |
|---|--------|-------|------|-------|
| 1 | Shared bot Database | db/database.py:20 (composition.py:269) | RW WAL busy=30s | hub; M1 victim+cause |
| 2 | LineageObserver Database | lineage_observer.py:37 | RW + DDL | 2nd in-process writer |
| 3 | call_budget conn | nlp/call_budget.py:72 | RW, 0.25s | fail-fast; symptom metric |
| 4 | Treasury ledger | treasury/transfers.py:86 | RW | separate file — fine |
| 5 | CLI ×28 | cli/*.py | RW + DDL | de-fang in Phase 4 |
| 6 | MCP compare bot_db | agentmcp/compare.py:86 | RW + DDL | fix in Phase 4 |
| 7 | MCP MarketData | agentmcp/market_data.py:24 | ro transient | add busy_timeout |
| 8 | MCP agent book | agentmcp/server.py:114,162 | RW | separate agent.db — fine |
| 9 | Web dashboard | web/db.py:38 | ro per-cycle | fixed; policy example |
| 10 | Health probe | health.py:21 | ro transient | fine |
| 11 | Gates dashboard | monitoring/gates.py:21 | ro | fine |
| 12–14 | research scripts / unstick tool / backup.sh | various | ro / offline / backup API | fine |

## Decision

Rejected: (b) splitting ancillary state into separate DB files (breaks
backup.sh/restore.sh single-file assumptions and AURAMAUR_DB_PATH contract;
unnecessary once M1 is fixed); (c) a generalized write-queue actor (huge
signature blast radius; LineageObserver already IS a queue — reuse it);
(d) pragma tuning alone (busy_timeout=30s already loses).

**Chosen: one writer connection + short-transaction discipline + transient-ro
policy.**

## Phases

**Phase 0 — Baseline metric (no code).** Per-subsystem daily locked-error
counts from logs (post-6290fd2 so wins are attributable). Baseline ≈150/day.

**Phase 1 — Kill the long transactions (~120 LOC, 3 files, low risk).**
- kalshi.py sync_positions: fetch all positions AND all get_market results
  into plain lists first (no DB statements), then one short write pass with
  the existing SQL (lines 973-1082 unchanged). Idempotent upserts; same
  single-commit atomicity, now ~ms not minutes.
- reconciler.py: collect stub-market rows during reconcile(), batch-insert at
  the end (INSERT OR IGNORE stays idempotent).
- database.py connect(): add `PRAGMA synchronous=NORMAL` (WAL-safe).
- Test: mock-ordering test asserting all get_market calls complete before the
  first db.execute.

**Phase 2 — `Database.transaction()` guard (~80 LOC + ~10 call sites,
low-medium risk).** asyncio.Lock-guarded async context manager
(`BEGIN IMMEDIATE` → yield → COMMIT, rollback on exception). Adopt first in
non-money frequent writers: broker/sync.py:358-377,414-469,542-596,
bot.py:1182-1228, lineage write methods, resolution_tracker.py:118-144,294-298.
Add a debug warning when a transaction holds the lock >250ms (catches
await-inside-transaction regressions). Autocommit callers keep working.

**Phase 3 — One in-process writer (~30 LOC, 2 files, low risk).**
LineageObserver.create() takes the already-connected shared Database instead
of building its own; wrap its batches in transaction(). Its queue already
decouples producers — no producer changes. Delete the second connection.

**Phase 4 — Read-only policy + CLI de-fanging (~90 LOC, 4 files, low risk).**
- Policy (add to CLAUDE.md): out-of-process consumers open `mode=ro` URIs,
  transient (open→query→close), busy_timeout>=5000, never Database.connect().
- agentmcp/compare.py: replace Database(bot_db) with the transient ro pattern
  from market_data.py:22-26; add busy_timeout there too.
- Database.connect(ensure_schema=...): when stored schema_version already
  equals SCHEMA_VERSION (checked read-only first), skip executescript(TABLES).
  CLI uses the fast path; `auramaur run` keeps full DDL. Caveat: the no-DDL
  guarantee is STEADY-STATE only — a version-skewed CLI (upgraded checkout
  against a not-yet-restarted bot) sees a behind schema_version and will run
  the full init/migrations against the live DB under the running bot. Restart
  the bot before using CLI tooling after upgrading the checkout.

**Phase 5 — Money path, last and guarded (~60 LOC + tests, medium risk).**
Wrap PnLTracker.record_fill/_record_fill_once (pnl.py:39-94+) and
ExecutionGateway._record_result writes (execution_gateway.py:490-576) in
transaction() — makes fill+cost_basis atomic and isolated from other tasks'
commits/rollbacks, which is what makes retry-on-busy safe for the first time
(order_id dedupe at pnl.py:49-66 then guarantees idempotence). Optional one
guarded retry after that lands. Paper soak >=48h before live. Do NOT touch
busy_timeout=30000, exchange/client.py, paper.py, or risk/manager.py.

## Verification

- Metric: locked-errors/day per subsystem. Success: >=95% total reduction and
  ZERO in money-path events (gateway.fill_record_failed,
  order_monitor fill_record_error, kalshi.sync_positions_error,
  position_sync.error) over a 7-day live soak per phase.
- Regression tests: (1) concurrent-task transaction isolation on
  transaction(); (2) sync_positions network-before-write ordering;
  (3) compare.py/market_data.py open bot DB mode=ro only; (4) CLI fast path
  performs no write when schema is current; (5) record_fill under injected
  SQLITE_BUSY leaves zero partial rows (blocker-connection pattern from
  tests/test_call_budget.py).

## Non-goals

No Postgres migration; no schema changes (SCHEMA_VERSION stays 30); no
changes to risk manager, paper gating, live gates, or kill switch; no
call_budget rewrite (its fail-fast design is correct — its error count is a
symptom metric for Phase 1); no busy_timeout lowering; no multi-file split;
web dashboard stays read-only by construction.
