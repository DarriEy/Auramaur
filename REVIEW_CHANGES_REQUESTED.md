# Review: IBKR ETF paper pillar — CHANGES REQUESTED

Reviewed 2026-07-18 (high-effort multi-agent review, 37 agents, all findings
independently verified). The core idea — a structurally paper-only ETF book
comparing LLM arms on IBKR quotes — is welcome and several pieces are good
(no order API on the pillar, fee-per-order modeled, preflight module, test
pins). It is not mergeable in this form. Blocking issues first, then policy
requirements, then scope.

## Blocking correctness issues (all CONFIRMED by independent verification)

1. **Shared paper-wallet corruption** (`strategy/ibkr_etf_paper.py` fill
   recording). ETF fills flow through PnLTracker into the shared
   `is_paper=1` cost_basis/pnl_ledger, which `PaperTrader._compute_balance`
   sums with **no venue scoping**. One ETF-scale BUY drives the shared
   paper wallet's spendable negative → every Polymarket/Kalshi paper entry
   bot-wide rejects with `insufficient_balance`, silently freezing all
   paper books. REQUIRED: partition the ETF book's accounting (its own
   wallet/namespace excluded from `_compute_balance`, or venue-scope that
   computation) + a regression test proving an ETF fill cannot move the
   Polymarket paper wallet.

2. **Full-connection rollback in a finally block**
   (`strategy/engine.py` forecast_snapshots handler; same pattern in
   `data_quality.py`). The unconditional `await self.db.db.rollback()`
   runs even after a successful savepoint rollback, discarding OTHER
   tasks' uncommitted writes on the shared aiosqlite connection — worst
   during the known nightly lock bursts. REQUIRED: roll back the full
   connection only when the savepoint rollback itself raises (the
   aggregator's original pattern).

3. **Fixed-name SAVEPOINT collisions** (`data_sources/aggregator.py`
   `gather()`, and engine's `forecast_observation`). Concurrent tasks
   nest identically-named savepoints; one task's RELEASE/commit swallows
   the other's, the second RELEASE raises, and the fallback full rollback
   destroys unrelated pending writes. REQUIRED: unique savepoint names
   per invocation (uuid suffix) or a serializing lock around the
   savepoint window.

4. **Ungated evidence-ranking change for every strategy**
   (`aggregator.py` ranking formula + `base.py` timestamp_quality).
   Capping recency and weighting by timestamp_quality demotes fresh
   websearch evidence below hours-old RSS for ALL callers — including the
   proven resolution_lens book — shifting verdicts and breaking the
   evaluability of records that restarted 07-08. REQUIRED: revert from
   this PR; if the ranking change has merit, propose it separately behind
   a config gate with an A/B plan.

5. **NaN quotes pass validation** (`exchange/ibkr_equity.py` get_quote).
   `bid <= 0 or ask <= 0 or bid > ask` is False for float('nan') (ib_async
   reports NaN with no market data) → NaN prices enter the paper book.
   REQUIRED: `math.isfinite` checks.

6. **Ships default-ON against repo convention** (defaults.yaml
   `etf_paper_enabled: true`) while its own preflight reports BLOCKED.
   New experiments ship disabled and are turned on deliberately.
   REQUIRED: default off; enablement is the operator's explicit act.

7. **Refresh-budget starvation** (`etf_max_signal_refreshes_per_cycle: 4`
   vs 16 target positions): the book silently caps at ~4 instruments.
   REQUIRED: either a rotation over the symbol set or an honest cap.

8. **Lexical timestamp comparison** in the data-audit path compares
   ISO strings with mixed offsets → audit permanently red. REQUIRED:
   parse to aware datetimes.

9. **Fabricated `risk_checks_passed` rows written outside the risk
   gateway** (direct trades-table INSERT). CLAUDE.md: the risk manager is
   the single gateway; no trade bypasses it. Paper-only does not exempt
   bookkeeping from that invariant — record through the gateway, or write
   rows that do not claim risk checks that never ran.

## Policy requirements (operator decisions, 2026-07-18)

A. **OpenAI API cost must be inside the P&L math.** Book every OpenAI
   call's actual token cost (from the usage field of the response) as a
   FEE into the ETF cell's paper ledger, amortized into the entries that
   the signal produced. The cell's record must answer "does the edge
   clear its own intelligence bill" — cost-exclusive P&L is not
   evaluable evidence here.

B. **The price_history migration ships separately.** The snapshot-key
   column + unique index on the largest table in the production DB (the
   table already implicated in nightly lock bursts) needs its own PR
   with a measured index-build/lock impact on a copy of the real DB, and
   a rollback plan. Do not ride it in with a strategy.

C. **Scale to the shop's measurement regime.** Budget/entries sized like
   the other paper cells (the graduation ladder judges event count and
   sign, not notional). If ETF-scale notional is the point, that is its
   own discussion.

D. **Split the PR.** Pillar+adapter+preflight+its tests = one PR.
   Lineage/data-quality framework = separate proposal. Core-file changes
   (engine/protocols/db) only where the pillar strictly needs them.

## What is already good (keep)

- No order-placement capability on the pillar (capability absence over
  policy) and the PAPER_SIMULATED execution mode is a reasonable protocol
  addition.
- Fee-per-order and spread caps modeled; preflight concept; config
  validation with cross-field checks; test pins.
