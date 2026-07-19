# Kraken research and graduation plan

Kraken directional trading remains paper-forced. The live wallet and the
validate-only shadow book are intentionally separate: `kraken_paper_positions`
is authoritative for paper positions and survives restarts.

## Evaluation contract

Every candidate uses the same `PortfolioEvaluator`: one shared $60 portfolio,
$30 slots, executable bid/ask estimates, fees, adverse slippage, forced final
liquidation, drawdown, turnover, time-in-market, per-pair attribution and an
equal-weight buy-and-hold benchmark. Parameter selection must happen only on a
training window; reported results come from sequential untouched test windows.
Warm-up bars are visible to indicators but explicitly non-tradable. The
equal-weight benchmark uses equal dollars per asset and pays the same modeled
spread, slippage, and fees as each challenger.

The registered challengers are LLM plus trend confirmation, cross-sectional
relative strength, residual mean reversion, volatility breakout, confirmed
events, and a passive-liquidity research hook. Passive liquidity is not valid
without order-book replay and queue-aware fills.

## Graduation

`graduation()` fails closed and requires at least 50 closed trades, 90 elapsed
days, positive cost-adjusted expectancy and PnL, benchmark outperformance,
drawdown no greater than 20%, and no pair contributing over 70% of absolute
pair PnL. It also requires a majority of at least three profitable walk-forward
folds, positive stressed-cost results, positive results in at least two declared
regimes, and a positive untouched holdout that beats its benchmark. Passing
authorizes only a $5 canary proposal; it does not change live configuration.

`auramaur kraken research --input FILE` accepts rows shaped as
`[ts, open, high, low, close, volume, bid?, ask?, event_score?,
order_imbalance?]`. `XBTUSDC` is required as the common market factor for
beta-adjusted residual returns. Missing optional fields are neutral rather than
fabricated.

API keys should be IP-restricted and least-privilege. Use Kraken's key-info
preflight to inspect permissions; withdrawal authority belongs on a separate
key and remains governed by the transfer gate.
