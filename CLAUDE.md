# Auramaur Safety & Development Rules

## ABSOLUTE RULES (never override)
1. **Paper trading is the default.** Real orders require ALL THREE gates:
   - `AURAMAUR_LIVE=true` environment variable
   - `execution.live=true` in config
   - `dry_run=False` per-order flag
2. **Kill switch**: If `./KILL_SWITCH` file exists, halt ALL trading immediately.
3. **Execution contracts are explicit and test-enforced.**
   - Directional entries pass all 15 risk checks through `ExecutionGateway.submit()`.
   - Paired/concurrent arbitrage and market-making use the dedicated gateway
     methods declared by `ExecutionMode`; they must not call an exchange's raw
     order method.
   - Prediction-market exits use the gateway exit contract. The sole
     off-gateway exception is the declared IBKR equity path.
4. **Never hardcode API keys.** All secrets come from environment variables.
5. **Never read `.env` files.** They contain secrets. Use `.env.example` for reference.
6. **Never force-push to main.**

## Commit Attribution
When making git commits, use `Assisted-by: Claude (Anthropic)` in the commit message body instead of `Co-authored-by`. The human author should always be the sole git author of record.

## Architecture
- Prediction-market placements flow through `auramaur/broker/execution_gateway.py`;
  its methods delegate to the appropriate exchange adapter. Strategy pillars
  must not call raw exchange order methods.
- Paper trading interception happens in `auramaur/exchange/paper.py`.
- `auramaur/risk/manager.py` is the approval authority for directional entries.
  Structural strategies use their declared, test-enforced gateway contracts;
  see `auramaur/strategy/protocols.py` and `tests/test_strategy_protocol.py`.
- The web dashboard (`auramaur/web/` + `web/` SPA) is read-only by construction: it opens the DB with SQLite `mode=ro` and must never gain venue credentials or order paths. Keep it that way.
- Out-of-process DB consumers (web, MCP, scripts) open the trading DB via transient `mode=ro` URIs with `busy_timeout>=5000` and never run `Database.connect()`'s DDL against the live file; CLI tooling connects with `ensure_schema=False`.

## Code Style
- Python 3.11+, async-first (asyncio)
- Type hints everywhere
- Pydantic models for all data structures
- structlog for logging (JSON format)
- Tests required for all risk checks

## Risk Defaults
- Max drawdown: 15%
- Max stake per market: $25
- Daily loss limit: $200
- Max open positions: 500
- Minimum edge: 5% after fees
- Kelly fraction: 30%
- Confidence floor: LOW
- Category exposure cap: 60%
- Second opinion divergence max: 0.25
