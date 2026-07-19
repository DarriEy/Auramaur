# IBKR research signal primitives

`auramaur.research.ibkr_signals` contains pure research-only candidates for:

- short-horizon price-shock mean reversion with a longer trend filter;
- pre-fitted pair residual mean reversion;
- futures curve carry confirmed by trend; and
- FX interest-rate carry confirmed by trend.

Every call requires an explicit timezone-aware `as_of`. Price observations must
be positive, finite, strictly ordered, no later than `as_of`, and optionally no
older than `max_age`. Pair observations must be timestamp-aligned. Futures curve
inputs must be unexpired and expiry-ordered. Invalid or insufficient data raises
`SignalInputError`; evaluators should treat that as **no decision**, never as
permission to trade.

The module performs no fetching, contract selection, sizing, cost modeling, or
execution. Hedge coefficients and interest rates must be point-in-time values
known at `as_of`. Backtests must lag revised macro data, fit pair coefficients on
training data only, construct continuous futures without future roll knowledge,
and apply commissions, spread, slippage, financing, borrow, and roll costs.
