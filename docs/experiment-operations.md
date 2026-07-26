# Experiment operations

The experiment CLI turns a JSON manifest into one immutable lineage and keeps
research evidence separate from capital-graduation evidence.

```text
auramaur experiment define manifest.json
auramaur experiment register manifest.json --db research.db
auramaur experiment replay manifest.json --db research.db
auramaur experiment shadow manifest.json
auramaur experiment report LINEAGE --db research.db \
  --execution-model linear-v1 --data-version dataset-2026-07
auramaur experiment scoreboard --db research.db
```

`define` validates the complete definition, implementation binding, immutable
snapshots, and initial portfolio before printing the content-derived lineage.
`register` must run before prospective evidence collection. Replay persistence
uses only paper `strategy_evaluations`; it never writes graduation decisions,
fills, outcomes, or P&L ledger rows.

## Manifest shape

Every manifest contains:

- `schema_version` (currently `1`);
- a complete `ExperimentDefinition` under `definition`;
- a pure implementation `binding`;
- chronological, point-in-time `snapshots`;
- `initial_portfolio`;
- an `execution_model` for portable target replay.

Portable bindings name an experiment class and constructor arguments:

```json
{
  "kind": "portable",
  "implementation": "auramaur.experiments.strategies.bias_harvest:BiasHarvestExperiment",
  "constructor_arguments": {"rules": {"band_lo": 0.1, "band_hi": 0.9}}
}
```

The complete rule object is required; abbreviated arguments above are only an
illustration. Bindings are restricted to `auramaur.experiments.strategies` and
cannot dynamically load broker, database, exchange, or production modules.

Specialized bindings call an existing pure proposal function against one named
feature and declare the semantics that must survive replay:

```json
{
  "kind": "specialized",
  "evaluator": "auramaur.experiments.strategies.cross_venue_arb:paired_arb_proposal",
  "feature_name": "cross_venue_pair",
  "proposal_kind": "cross_venue_pair",
  "semantics": "all_or_none",
  "fixed_arguments": {
    "min_confidence": 0.8,
    "required_gap": 0.05,
    "stake_usd": 10
  }
}
```

Supported semantic labels are `all_or_none`, `coupled_quotes`, `routing`,
`manual_action`, and `asset_order`. Specialized replay records immutable
proposal packages but deliberately reports no fill, cost, or P&L metric until
a package-specific execution model exists.

## Scoreboard semantics

The scoreboard emits one row per lineage and portable data/execution-model
cohort. It never pools incompatible cohorts. Specialized proposal replays are
shown as `proposal_replay` with no P&L, while untested registrations are
`no_replay`. Prospective warmup and holdout counts remain separate columns.
