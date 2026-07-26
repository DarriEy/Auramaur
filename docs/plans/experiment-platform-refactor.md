# Experiment Platform Refactor

## Decision

Auramaur will keep a small, strict production kernel and gain a separate
experiment layer. Experiments describe hypotheses and desired exposures; they
do not schedule themselves, write accounting records, or place orders.

The production order path remains unchanged:

- directional prediction-market entries use `ExecutionGateway.submit()`;
- paired and quoting strategies use their declared gateway contracts;
- paper trading remains the default and all existing live gates remain binding.

This is an incremental migration, not a rewrite of the running bot.

## Problem

Today a strategy often owns discovery, scheduling, analysis, persistence, risk,
and execution. Adding a hypothesis can therefore require edits to the bot
orchestrator, settings, database, monitoring, and execution code before the
hypothesis has produced useful evidence.

The repository already has valuable but disconnected primitives:

- `strategy.registry` declares production strategy and execution contracts;
- `strategy_experiments` records prospective versions;
- `backtest.walk_forward` provides leakage-resistant evaluation helpers;
- the gateway, ledger, reconciliation, and graduation system protect capital.

The refactor connects these pieces while keeping their responsibilities
separate.

## Target boundaries

### Core

Stable domain types, point-in-time snapshots, target positions, lineage hashes,
and normalized result records. Core has no venue client or database dependency.

### Lab

Disposable hypothesis implementations, feature construction, batch research,
and deterministic replay. A lab experiment consumes an immutable snapshot and
returns target positions. It cannot place orders.

### Runtimes

- **Research** calls the experiment against explicitly supplied snapshots.
- **Replay** applies a declared fill/cost model and produces normalized results.
- **Shadow** records proposed targets without submitting them.

There is intentionally no live runtime in the experiment package. A future
production adapter must live beside the broker and accept only registered,
graduated targets through the existing gateway and its full safety chain.

### Production kernel

Venue connections, scheduling, risk approval, order state, reconciliation,
ledger, settlement, kill switches, and monitoring remain production concerns.

## Experiment contract

Each experiment has immutable metadata:

- key and strategy source;
- hypothesis and economic mechanism;
- implementation version;
- canonical configuration and content-derived lineage ID;
- venues, instruments, and cadence;
- primary metric and baseline;
- sample, holdout, cost, drawdown, and rejection criteria;
- capital eligibility.

A material configuration or implementation-version change creates a new
lineage. Research and production results must never be silently pooled across
lineages.

The behavioral interface is deliberately narrow:

```python
async def evaluate(
    snapshot: MarketSnapshot,
    portfolio: PortfolioSnapshot,
) -> list[TargetPosition]:
    ...
```

Targets express desired exposure, not orders. Execution policy, sizing limits,
and approval remain outside the experiment.

## Migration phases

### Phase 1 — contracts and deterministic lab runtime

- Add the experiment definition, snapshot, target, registry, and runtime APIs.
- Add chronological replay with explicit, versioned fill costs and an in-memory
  shadow sink.
- Bind each definition to its implementation before a runtime can execute it.
- Test complete lineage hashing, deep immutability, feature availability,
  point-in-time ordering, and runtime isolation.

Exit criterion: a new no-order experiment can run in research, replay, and
shadow modes without touching `bot.py`, the database schema, or a venue client.

### Phase 2 — migrate one decision and prove parity

- Extract the deterministic bias-harvest band selector into the experiment
  package.
- Make the existing production pillar delegate to that pure selector.
- Prove parity across boundary and representative in/out-of-band prices.
- Keep discovery, eligibility, sizing, risk, persistence, and execution on the
  existing production path.

Exit criterion: the migrated decision logic is replayable without constructing
the bot, while production order behavior is unchanged.

### Phase 3 — persisted experiment registry and reports

- Persist definitions into the existing `strategy_experiments` table through a
  repository adapter.
- Add data-version and execution-model-version fields through an additive
  migration.
- Produce a standard cost-adjusted report using `backtest.walk_forward`.
- Clearly separate prospective, holdout, paper, and live observations.

Exit criterion: results are reproducible from definition ID, data version, and
execution-model version.

### Phase 4 — migrate complete representative directional strategies

- Extract one simple pillar's signal calculation into the experiment contract.
- Keep its current task as a compatibility adapter.
- Route its targets through the unchanged risk and execution gateway.
- Confirm old and new paths generate equivalent decisions in shadow mode.

Exit criterion: production behavior is equivalent and the strategy logic can be
replayed without constructing the bot.

### Phase 5 — runtime decomposition

- Move data collection, research/replay, trading, reconciliation, and monitoring
  into independently startable services.
- Replace strategy-owned sleep loops with scheduler-owned invocations.
- Keep an append-only event boundary between collection and consumers.

Exit criterion: a failed research worker cannot affect trading or
reconciliation, and a trading restart does not lose source events.

### Phase 6 — retire compatibility paths

- Migrate remaining suitable directional pillars.
- Leave genuinely different machines (market maker, multi-leg arbitrage, IBKR
  multiasset) on specialized contracts.
- Remove obsolete task-specific wiring only after parity and operational
  evidence.

## Non-goals

- No redesign of the live gateway in Phase 1.
- No automatic conversion of a target into an order.
- No claim that all strategies share one cadence or execution model.
- No new live strategy and no relaxation of graduation requirements.
- No database rewrite.

## Guardrails

1. Core and lab modules must not import venue clients, the broker gateway, or
   database implementations.
2. Experiments return data; runtime adapters own side effects.
3. Replay results identify their execution-model version.
4. The experiment package contains no live executor or production-side-effect
   imports. Live adaptation belongs to the production composition root.
5. Production strategy metadata remains authoritative until each strategy is
   deliberately migrated.
6. Every migrated seam gets focused production-delegation parity coverage.
   Full research/replay/shadow parity remains required before removing its
   compatibility path.

## Current implementation status

Phases 1 and 2 and Phase 3 are implemented. The
registered definition is stored in the existing `strategy_experiments` table
using database registration time and its immutable lineage ID. Replay results
are linked by that lineage but stored only as paper `strategy_evaluations`;
they never populate the decision, fill, outcome, or P&L records consumed by
prospective graduation. Registration fails closed on lineage conflicts and on
same-second version ambiguity.

The standardized report requires one explicit execution-model and data-version
cohort, so incompatible replay runs are never blended into one equity curve. It
reports net and gross P&L, costs, turnover, drawdown, event hit rate and profit
factor, independent-market count, average gross exposure, and 0x/1x/2x cost
sensitivity. Retrospective replay metrics and prospective warmup/holdout counts
are separate. Rejection rules without a supported or resolved prospective
evaluator are `not_evaluable`, making the overall result insufficient rather
than silently passing it.

The replay model is deliberately modest: it is
a chronological, immediate-fill portfolio simulator with explicit linear fees
and slippage. It rejects out-of-order or duplicate events, unavailable features,
unpriced holdings, mismatched target reference prices, unavailable cash, and
non-finite financial values. It is not an order-book or queue simulator.

Bias harvest is the migrated proof of concept. Its complete deterministic
candidate-to-proposal decision is portable, while every live-sensitive risk,
gateway, persistence, order-lifecycle, and accounting operation remains in
production. The remaining registry entries are explicitly planned on portable
or specialized tracks.

## Full-set migration sequence

The bias-harvest proof of concept now covers the complete candidate-to-proposal
boundary: band selection, venue/activity/dispute/liquidity/category/time gates,
one-shot state, paper fill haircut, maker-book requirements, direction, fair
probability, attribution, entry price, and capped target. Production maps venue
and database facts into that pure input, then keeps risk and gateway execution
unchanged. Research, replay, and shadow execute the same bound implementation.

The authoritative inventory lives in `auramaur.experiments.migration` and is
conformance-tested against `strategy.registry`. Migration proceeds in waves:

1. Deterministic directional pricing: `vol_anchor`, `weather_temp`,
   `econ_indicator`, `settlement_arb`.
2. Derived-market signals: `term_structure`, `platform_consensus`,
   `informed_flow`, `momentum_coupling`, `interim_manager`, `long_horizon`.
3. Evidence/model strategies: core trading, news reactor, agent-trader lanes,
   and resolution-lens lanes.
4. Specialized contracts: paired arbitrage packages, market-maker quote plans,
   and independently gated Kraken/IBKR asset proposals.

A strategy moves from `planned` to `migrated` after production delegates its
deterministic seam to a pure contract, focused old/new decision coverage passes,
and its existing execution-contract suite still passes. The status does not
claim that the contract is supported by the generic replay runtime.

Bias harvest currently satisfies that decision-seam gate. The following stack
layers migrate portable directional/routing seams, paired and quoting
contracts, and external-asset contracts in independently reviewable changes.
