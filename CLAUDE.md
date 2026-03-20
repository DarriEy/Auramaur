# Auramaur Safety & Development Rules

## ABSOLUTE RULES (never override)
1. **Paper trading is the default.** Real orders require ALL THREE gates:
   - `AURAMAUR_LIVE=true` environment variable
   - `execution.live=true` in config
   - `dry_run=False` per-order flag
2. **Kill switch**: If `./KILL_SWITCH` file exists, halt ALL trading immediately.
3. **Never bypass risk checks.** All 15 checks must pass before any order.
4. **Never hardcode API keys.** All secrets come from environment variables.
5. **Never read `.env` files.** They contain secrets. Use `.env.example` for reference.
6. **Never force-push to main.**

## Architecture
- All money flows through `auramaur/exchange/client.py` — this is the ONLY file that touches the CLOB API for orders.
- Paper trading interception happens in `auramaur/exchange/paper.py`.
- Risk manager in `auramaur/risk/manager.py` is the single gateway — no trade bypasses it.

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
- Max open positions: 15
- Minimum edge: 5% after fees
- Kelly fraction: 25%
