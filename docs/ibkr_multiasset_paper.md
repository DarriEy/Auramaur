# IBKR multi-asset paper books

Auramaur uses Trader Workstation as a **read-only market-data source** and
simulates every fill locally. The multi-asset client has no order-placement
method. A live-account TWS login does not make these books live.

## Books

| Book | Coverage | Contract policy | Paper capital model |
|---|---|---|---|
| `global_etf` | US-listed global equity, rates, credit, commodity, currency and real-estate ETFs | Qualified `STK` with primary exchange | fractional shares at ask/bid |
| `fx` | ten liquid G10 crosses | `CASH` on `IDEALPRO` | 1,000-base-unit lots, USD translation |
| `futures` | micro equity index, rates, energy, metals and agriculture | nearest contract with >7 days to expiry; exact held `conId` retained | conservative per-contract capital plus multiplier P&L |
| `international_equity` | Canada, UK, Europe, Japan, Hong Kong and Australia | native listing, exchange and currency | fractional shares, mark translated to USD |
| `options` | 30–60 DTE ATM calls and puts on liquid ETFs | exact qualified chain contract; exact held `conId` retained | whole contracts, 100 multiplier |
| `bonds` | US Treasury tenors and a US corporate issue | maturity-bounded IBKR bond scanner, exact `conId` | $1,000 face-value units |

The catalog is in `auramaur/exchange/ibkr_instruments.py`. Instrument keys are
unique and every entry declares its security type, exchange, currency, calendar,
multiplier and discovery policy.

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
- Quotes older than the configured age are rejected.
- Futures/options/bonds are periodically rediscovered, while held derivatives
  continue to be marked and exited using their original `conId`.
- `price_source` is stored on every position and fill.

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

The preflight checks structural isolation, schema, every configured contract,
fresh BBOs, at least 21 daily bars, quote provenance, and per-book forward
evidence. Run it during each venue's session when diagnosing BBO availability;
closed venues are expected to lack executable quotes.

The startup books panel and periodic strategy table report each book separately.
`ibkr_paper_state.last_cycle_at`, `last_success_at`, and `last_error` provide
machine-readable health.

## Graduation

There is no live execution implementation for these six books. Adding one is a
separate design and review. At minimum, a book needs 30 closed paper positions,
positive net trade P&L after modeled commissions, profit factor above 1.1, fresh
native quotes for every traded instrument, and asset-specific reconciliation.
No book graduates another book.
