# Interim manager — operator-proposed book management, ladder-evaluated

The graduation ladder correctly paper-forces nearly every strategy while it
earns its record, which leaves live balance idle in the interim. The interim
manager is the disciplined channel for deploying that balance **manually**
(operator or assistant sessions) without abandoning measurement: every
proposal passes the same gauntlet as any strategy entry, books under its own
`strategy_source`, and the ladder judges the manager exactly like the
strategies it stands in for. It is scaffolding with a demolition date built
in.

## What it is

A proposal queue plus a pillar. `auramaur manager propose` records a manual
trade thesis; the pillar validates each pending proposal against the charter
rules below, the risk gateway (all checks), and the graduation ladder (which
paper-forces the manager until IT has earned live), then executes and books
it under `strategy_source = "interim_manager"`. No autonomous signal
generation, no LLM budget consumption, no exemptions.

## The charter — accumulated learnings this manager must obey

Distilled from the project ledger and incidents through 2026-07-19. The
mechanical rules below are enforced in code; the judgment rules bind whoever
writes proposals.

1. **No nameable mispricing mechanism, no trade.** Unexplained disagreement
   with the market defers to the market. Enforced: a proposal must carry a
   thesis of substance, and the thesis is stored immutably beside the fill
   for post-mortem.
2. **Raw model probabilities are compressed and mis-centered.** Six weeks of
   Kraken reads lived in [0.42, 0.56] while resolving 29 points wide; the
   calibrated view, not the raw number, is what a probability means.
   Judgment rule: state fair value AFTER calibration awareness, not model
   output verbatim.
3. **A neutral read from a degraded model is absence of signal, not signal.**
   The 2026-07-19 Gemini-fallback incident produced 114 consecutive 0.5
   reads; the June resolution-lens collapse was the same failure. Never
   propose on analysis produced by fallback routing.
4. **Fees and executable depth decide marginal trades.** Settlement fees,
   taker fees on both venues, deci-cent tick structures, and thin books all
   materially changed recorded P&L this quarter. Enforced: the risk gateway's
   fee-cleared edge requirement and depth-capped sizing apply to every
   proposal; small notionals in thin books.
5. **Mid-band divergence is adverse selection.** The 10–20% divergence bucket
   realized −$5.90/market at 22% win; strong divergence with high confidence
   or no trade.
6. **Respect the ledger's category verdicts.** Do not re-fight a category a
   strategy has already proven negative in with the same class of reasoning
   (news_speed, technical momentum, weather_temp were all retired on
   evidence). The ledger outranks conviction.
7. **Long-dated markets are unresearchable; very short-dated leave no time to
   be right.** The engine's resolution-window filters exist for measured
   reasons; proposals inherit them via the risk gateway.
8. **Venue truth outranks local assumption.** Positions are what the venue
   snapshot says; fills are what the venue confirms; a pending order is not
   a position. Reconcile before acting on state.
9. **Size for information while unproven.** Until the manager's own cells
   graduate, stakes are evidence-gathering stakes (config-capped), not
   P&L-maximizing stakes.

## The generalized decision rule (enforced)

The manager's trade condition is not `fair > market`. It is:

    robust_edge = fair_probability
                  − executable_probability
                  − uncertainty_buffer      (CI half-width, or the default)
                  − taker_fees
                  − slippage_buffer
                  − thin_liquidity_penalty
                  − correlation_penalty     (per open manager position in category)

    execute only if robust_edge ≥ min_robust_edge
                 and entry price ≤ the proposal's max_entry_price
                 and the thesis sunset has not passed
                 and (charter, human-judged) the thesis is falsifiable,
                     resolution is objective, and a catalyst exists

The arithmetic gates are computed and recorded per proposal (`robust_edge`,
`decision_price`, and the full haircut breakdown in the skip reason) — the
manager is deliberately conservative about apparent edges built on weak
estimates. Falsifiability and resolution-objectivity remain human judgments;
the schema records catalyst, invalidation, and sunset so post-mortems can
score whether the judgment was honest.

## Thesis classes (evaluated separately)

Every proposal declares its anchor class; `auramaur manager report` scores
each class independently, because a manager that only picks good classes is
learnable even when individual trades lose:

| class | independent anchor | typical divergence cause |
|---|---|---|
| forecast_divergence | weather/polling/econ forecasts | stale prices, headline anchoring |
| cross_venue | equivalent contracts elsewhere | fragmented liquidity |
| conditional_inconsistency | related markets' implied probabilities | contracts priced independently |
| timeline | known process and deadlines | procedural-timing confusion |
| base_rate | historical reference class | vivid-recent-event attention |
| resolution_language | contract rules vs public reading | ambiguous wording |
| information_lag | newly released official data | slow repricing |
| exclusive_basket | sum of outcome probabilities | bucket-level imbalance |

Priority for the interim period: forecast_divergence,
conditional_inconsistency, exclusive_basket — legible evidence, useful
feedback even in losses.

The mandate, compiled: **find short-lived, falsifiable probability
discrepancies with an external anchor, an explainable mispricing mechanism,
and enough robust edge to survive uncertainty and execution costs.**

## Auto-proposals (charter amendment, 2026-07-20)

The operator amended the "no autonomous signal generation" clause: when
`auto_propose` is armed, the manager may author its own queue entries.
Confidence is OPERATIONALIZED, never judged:

- source signal must carry `auto_min_confidence` (default HIGH) from a
  non-exempt strategy, be fresh (≤6h), for a market neither held nor
  proposed in the last 48h;
- the raw probability is Platt-calibrated per category, and the calibrated
  edge must clear `auto_min_calibrated_edge` (default 0.10);
- the proposal's confidence interval is the configured `auto_ci_width` —
  a deliberately honest haircut the robust-edge gate then charges;
- capped at `auto_daily_cap` per day, `auto_stake_usd` per entry, 24h
  sunset, and the category-delegation rule applies before authorship.

Every auto proposal then faces the IDENTICAL gauntlet as an operator
proposal — nothing about the gates changes with authorship. The `proposer`
tag separates human and auto theses in the scorecard, so the evaluation
contract judges each author on its own record. Tracked default: OFF; arming
is a dated operator override.

## Evaluation contract (pre-registered)

- `strategy_source = "interim_manager"`; **not** in `exempt_strategies` — the
  ladder paper-forces it until each (manager × category) cell clears the same
  bar as everyone: `min_markets` independent markets in the window with a
  positive mean-P&L lower confidence bound. No special thresholds.
- Proposals the pillar skips (delegation, sunset, thesis, expiry, risk
  rejection) are recorded with their reason — the queue is an auditable
  decision log, not just an order channel.
- The scorecard judges reasoning, not only money: fair-value calibration
  (every executed proposal records its prediction), whether the claimed
  mechanism was real, whether invalidation occurred before execution,
  whether the price limit protected the edge, and whether the manager
  selected good thesis classes — variance and bad reasoning are separable.
- Review cadence: alongside the normal graduation review. The manager's
  record is compared per category against the strategies it stands in for;
  if a manager cell and a strategy cell both hold records in a category, the
  strategy's takes precedence at equal evidence.

## Delegation and sunset (enforced in code)

- **Delegation:** before executing any proposal, the pillar checks the
  ladder report. If ANY non-exempt strategy other than the manager holds a
  `live` or `probation` cell in the proposal's category, the proposal is
  skipped with reason `delegated to <strategy>` — the graduate owns the
  category, and the manager stands down there permanently.
- **Sunset:** when the number of graduated (live/probation) non-exempt
  strategy cells reaches `sunset_after_live_cells`, the manager stops
  executing entirely and expires its queue. The interim is over; the
  strategies have the book.
- Open positions are never stranded by either rule: exits flow through the
  engine's position-based exit paths, which the ladder never touches.

## Configuration

`interim_manager` in config (tracked default `enabled: false`; arm it in the
operator's `defaults.local.yaml` with the usual dated rationale block):
`stake_usd` per-proposal cap, `max_entries_per_cycle`, `max_open_positions`,
`proposal_ttl_hours`, `sunset_after_live_cells`, `interval_seconds`, and
`paper` (tracked default `true`; the ladder additionally paper-forces
unproven cells regardless).
