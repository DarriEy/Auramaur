# Polymarket cannot graduate a strategy either (measured 2026-07-28)

Same shape as the IBKR finding, different cause. `GraduationLadder` runs in
`enforce` mode with `prospective_only: true`, and **every one of 112 cells has
n=0 prospective evidence**. Not "not yet" — structurally, today.

## Two independent blockers

### 1. Holdout warmup — by design, no action needed

`DecisionTracker.capture` stamps `is_holdout` once, at capture:

```sql
CASE WHEN datetime('now') >= (SELECT holdout_starts_at ...) THEN 1 ELSE 0 END
```

`holdout_starts_at = registration + holdout_warmup_days (14)`. The oldest clock
(`llm`) registered 2026-07-24, so its holdout opens **2026-08-07**. All 291
decisions captured so far are `is_holdout=0` and, because the flag is stamped
once and never revisited, they can never count. That is the intended
anti-fitting warmup.

Earliest possible graduation is therefore roughly **September**: holdout opens
2026-08-07, then `min_calendar_days: 30` and `min_regime_months: 2` must elapse
on decisions captured after that date.

### 2. Fill evidence — a real defect, and fatal on its own

`require_executable_fills: true` admits only
`credible_fill_evidence: [venue_fill, book_cross, trade_through]`. Of 111
filled decisions:

| evidence | n |
|---|---|
| `synthetic` | 109 |
| `book_cross` | 2 |

`best_ask` is NULL on **all 109**, and `_place_and_record` reads exactly that
field to decide whether a paper fill crossed. So the fills are stamped
`synthetic` and are uncountable forever.

The cause is in `_capture_decision`, which resolves the book by querying a
table someone else populates:

```sql
SELECT best_bid,best_ask FROM orderbook_snapshots
 WHERE market_id=? AND (?='' OR token_id=?) ORDER BY recorded_at DESC LIMIT 1
```

`orderbook_snapshots` is filled by a separate recorder task on its own cadence,
so at decision time the row often does not exist yet: **51 of 111 filled
decisions are on markets the recorder never sampled at all.** The remainder
miss on `token_id` — Kalshi orders carry ticker-style ids
(`KXFDAAPPROVE-MDMA-27JAN01`) while the book holds Polymarket CLOB numerics.

This is the same class of defect fixed for the IBKR books on 2026-07-27: a
decision that depends on another component having already recorded the state it
needs, instead of recording the state it actually used.

## The fix, and its blast radius

The gateway should capture the book the order was BUILT from, not go looking
one up afterwards. Minimal shape: optional `best_bid` / `best_ask` on
`TradeIntent`, preferred by `_capture_decision` when supplied and falling back
to the current lookup when not — so no existing caller changes behaviour — then
supply them from each pillar, which already holds a book at decision time.

That touches shared money-path code used by every prediction-market pillar, so
it wants its own focused pass rather than being appended to other work.

**Until it lands, no Polymarket strategy can graduate on merit**, and the seven
cells trading live today do so through `exempt_strategies` — dated operator
promotions (`agent_trader_opus`, `term_structure`, `interim_manager`,
`arbitrage`, `order_monitor`), not ladder decisions.
