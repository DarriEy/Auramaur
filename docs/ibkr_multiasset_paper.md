# IBKR multi-asset paper books

Auramaur uses Trader Workstation as a **read-only market-data source** and
simulates every fill locally. The multi-asset client has no order-placement
method. A live-account TWS login does not make these books live.

## Books

| Book | Coverage | Contract policy | Paper capital model |
|---|---|---|---|
| `global_etf` | US-listed global equity, rates, credit, commodity, currency and real-estate ETFs | Qualified `STK` with primary exchange | fractional shares at ask/bid |
| `fx` | ten liquid G10 crosses | `CASH` on `IDEALPRO` | 1,000-base-unit lots, USD translation |
| `futures` | micro equity index, rates, energy, metals and agriculture | product-specific roll buffer, then highest reported volume among nearest valid expiries; exact held `conId` retained | conservative per-contract capital plus multiplier P&L |
| `international_equity` | Canada, UK, Europe, Japan, Hong Kong and Australia | native listing, exchange and currency | fractional shares, mark translated to USD |
| `options` | 30–60 DTE ATM calls and puts on liquid ETFs | exact qualified chain contract; exact held `conId` retained | whole contracts, 100 multiplier |
| `bonds` | US Treasury tenors and a US corporate issue | maturity-bounded IBKR bond scanner, exact `conId` | $1,000 face-value units |

The catalog is in `auramaur/exchange/ibkr_instruments.py`. Instrument keys are
unique and every entry declares its security type, exchange, currency, calendar,
multiplier and discovery policy.

## Manifest and qualified registry

The trading space is deliberately hybrid. The version-controlled catalog is the
only authority allowed to introduce economic exposure. IBKR probing qualifies
those declarations into exact broker identities; it never adds a ticker or
scanner result to the universe by itself.

Successful preflight writes `ibkr_contract_registry`, including the manifest
hash, `conId`, local symbol, trading class, exchange, currency, multiplier,
quote provenance, history availability, and validation time. With
`multiasset_registry_required: true`, new entries require a current `eligible`
row whose manifest hash still matches. A catalog edit therefore fails closed
until the instrument is probed again. Existing positions remain manageable by
their persisted specification and original `conId`.

Static listings, FX, futures, options, and government bond policies become
eligible after successful qualification. Corporate bond scanner results need an
operator to approve the exact issue:

```console
auramaur ibkr-contract-approve CORP_5Y --reason "reviewed issuer, maturity and liquidity"
```

Changing the manifest invalidates prior approval. Failed probes are quarantined;
disabled instruments remain outside probing and runtime eligibility.

`multiasset_disabled_instruments` is the explicit entitlement quarantine. The
current username lacks TSEJ data, so `7203.T` remains catalogued but does not
enter runtime or paper evidence until its preflight succeeds.

## Isolation and risk

- TWS connection is always `readonly=True` on a distinct client ID.
- `IBKRReadOnlyMarketData` deliberately exposes no broker order API.
- Strategy execution mode is `PAPER_SIMULATED`.
- Positions, fills, marks and realized P&L live only in `ibkr_paper_*` tables.
- The shared prediction-market paper wallet, `cost_basis`, `portfolio`, and
  `pnl_ledger` are never touched.
- Every book has a separate budget, position cap, deployment cap, daily loss
  limit, stop, take-profit and maximum spread.
- Open positions are marked before new risk is allowed. Realized commissions
  and unrealized losses count toward the daily loss limit.
- Only TWS quotes explicitly tagged `ibkr_live` (`marketDataType=1`) can create
  marks or fills. Frozen, delayed, delayed-frozen, unknown, stale and synthetic
  quotes fail closed.
- A dropped TWS socket clears contract/session caches and reconnects under a
  single connection lock; concurrent book cycles cannot create duplicate API
  sessions.
- Futures/options/bonds are periodically rediscovered, while held derivatives
  continue to be marked and exited using their original `conId`.
- The complete instrument specification is persisted with an opened position.
  Removing or disabling a catalog entry therefore cannot strand it: held
  positions remain markable and exit-manageable.
- Hard stops and take-profits run from a fresh executable BBO before entry-only
  spread, FX-refresh, history and momentum gates. If a current FX cross is
  temporarily missing, exits use the position's stored entry FX conversion.
- `price_source` is stored on every position and fill. Monitoring counts only
  closed `trade` rows in trade count and win rate, while commissions and
  financing remain included in net P&L.
- Schema migrations verify the required columns before advancing
  `schema_version`; lock/busy errors are propagated and cannot leave a database
  falsely stamped as migrated.

## Option-data limitation

The current IBKR username resolves option contracts but TWS returns error 10091
for OPRA/API quotes. Auramaur can calculate a Black–Scholes diagnostic value from
the IBKR underlying and historical volatility, tagged `synthetic_option`, but
the execution gate rejects synthetic values. The options book therefore remains
non-executing until a fresh native IBKR option BBO is available. Synthetic marks
must not be counted as forward paper evidence.

## Operations

TWS must be running with API socket clients enabled, port `7497`, and Read-Only
API selected. Run:

```console
auramaur ibkr-multiasset-preflight
```

The TWS username may be a live-account login when IBKR paper credentials are
unavailable. This changes only the quote session: the socket is still opened
with `readonly=True`, the adapter has no order method, and all simulated
positions/fills remain in Auramaur's local database.

The preflight checks structural isolation, schema, every configured contract,
fresh BBOs, at least 21 daily bars, quote provenance, and per-book forward
evidence. Run it during each venue's session when diagnosing BBO availability;
closed venues are expected to lack executable quotes.

Set `IBKR_MARKET_DATA_TYPE=1` in the compose environment. Keep
`IBKR_QUOTE_ENVIRONMENT=paper`; using a live-account quote login is a separate,
explicit operational review. Values 2–4 may still
qualify contracts and expose delayed/frozen data for diagnostics, but registry
status becomes `qualified_no_live_data` and the runtime will not open risk.

The sector-ETF and OTM option additions roughly doubled the probed contract
count, so preflight wall time scales accordingly at the pacing-safe
concurrency of 2. That is deliberate: the cap protects the session from IBKR
pacing violations; budget extra minutes rather than raising concurrency.

The startup books panel and periodic strategy table report each book separately.
`ibkr_paper_state.last_cycle_at`, `last_success_at`, and `last_error` provide
machine-readable health.

Enable the experiment only in the machine-local override after the PR and
preflight gates pass:

```yaml
ibkr:
  multiasset_paper_enabled: true
```

The tracked default remains `false`, so a fresh checkout cannot start the six
books accidentally. To stop new cycles, set the local override back to `false`;
existing local paper positions remain in the isolated tables for inspection.

## Graduation

There is no live execution implementation for these six books. Adding one is a
separate design and review. At minimum, a book needs 30 closed paper positions,
positive net trade P&L after modeled commissions, profit factor above 1.1, fresh
native quotes for every traded instrument, and asset-specific reconciliation.
No book graduates another book.


## Evidence contract (amended, pre-registered 2026-07-20)

The 200-round-trip / 180-day contract is arithmetically unreachable for
slow-turnover books: an FX book holding 2-6 weeks across 4 slots produces
roughly 15-50 round trips in 180 days. Amended BEFORE any FX evidence
existed:

- **Primary**: `evaluate_ibkr_daily_evidence` over daily marked-to-market
  book P&L (`ibkr_paper_daily_marks`, written idempotently by each cycle) —
  at least 120 daily observations across at least 180 days, positive 95%
  lower confidence bound on the daily mean, drawdown within 10% of budget.
- **Secondary (cost realism)**: at least 30 cost-adjusted round trips with
  their own positive lower bound.
- **Anti-gaming clause**: holding brackets (stops, take-profits, momentum
  exits) must never be shortened merely to manufacture observations; a
  cadence change requires its own pre-registration with rationale.
- **Entry ordering**: when more signals qualify than slots remain, entries
  are taken strongest-normalized-momentum first (not universe order).
- **Carry upgrade path**: `fx_carry_trend` and the book's own trend signal
  are recorded daily, execution-free, to `ibkr_research_signals` (rates via
  FRED OECD immediate-rate series; monthly, lagged — auditable in the
  detail column). Wiring carry into entry RANKING (never gating alone) is
  permitted only after ≥60 recorded days show the blend would not have
  degraded the realized book, and requires its own pre-registered change.
