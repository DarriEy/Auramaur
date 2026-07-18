# Auramaur

An autonomous trading bot for prediction markets (Polymarket and Kalshi),
built in six days by a hydrologist using Claude Code.

It is an experiment, not a money machine — paper-trading is the default, live
capital is gated behind a graduation ladder, and most strategies stay on
probation. Treat realized performance as a research result, not a promise.

## What it does

Connects to Polymarket and Kalshi. Scans markets, gathers news from RSS / web
search / Reddit / NewsAPI / FRED / Manifold / Metaculus, asks Claude to
estimate the probability of each outcome, compares against the market price,
and trades when there's edge after fees.

- **NLP analysis** with calibrated probability estimation (Platt scaling on
  resolution feedback) and adversarial second opinions
- **Risk management** with 15 independent checks per trade, geometric Kelly
  position sizing, drawdown limits, and category exposure caps
- **Multi-exchange** order routing with per-exchange fee adjustment
- **Position reconciliation** against on-chain CLOB trade history
- **Resolution tracking** that closes the loop into the calibration system

## Safety constraints (hard-coded)

1. Paper trading is the default. Live orders require **all three gates**:
   `AURAMAUR_LIVE=true`, `execution.live=true`, and per-order `dry_run=False`.
2. A `KILL_SWITCH` file (at the repo root or the working directory) halts all trading.
3. **Directional entries** pass the 15 risk checks via the single `ExecutionGateway`.
   The market maker (resting two-sided quotes) and concurrent arb legs run
   *declared, test-enforced* direct-execution contracts — see `ExecutionMode` in
   `auramaur/strategy/protocols.py` and the conformance guard
   `tests/test_strategy_protocol.py` — not the directional risk path. Exits have
   their own contract. The bypasses are intentional and checked, not implicit.
4. No API keys in code — all secrets come from environment variables.

## Quickstart

```bash
# Install
uv sync

# Configure (copy and fill in)
cp .env.example .env

# Run in paper mode (default). --hybrid runs the full multi-strategy set
# (arb + news-speed + LLM + market-making + bias-harvest + resolution-lens …);
# --agent uses the single agentic analyzer instead.
auramaur run --hybrid

# Run tests
uv run pytest
```

To go live: set `AURAMAUR_LIVE=true` in your environment, set
`execution.live: true` in `config/defaults.yaml`, and accept that you are
trading real money on prediction markets where most participants are now bots
running on the same handful of foundation models as yours.

## Architecture

```
auramaur/
├── exchange/         Polymarket CLOB, Kalshi, IBKR, paper trader
├── data_sources/     News, RSS, Reddit, FRED, Manifold, Metaculus
├── nlp/              Claude analyzer, calibration, prompts
├── risk/             15-check pipeline, Kelly sizer, graduation ladder
├── strategy/         Engine, the strategy pillars, signal detection, resolution
├── broker/           Execution gateway, allocator, syncer, reconciler, PnL ledger
├── treasury/         Cross-venue capital and transfers
├── agentmcp/         MCP bridge exposing the plumbing to an external agent
├── monitoring/       Display, readiness, attribution
└── db/               SQLite schema
```

The CLOB API is touched for orders in one place, `exchange/client.py`; all
placements funnel through the `ExecutionGateway` in `broker/`. Paper-trading
interception happens in `exchange/paper.py`. The risk manager in
`risk/manager.py` is the single path through which directional trades are
approved.

## Status

Runs live on Polymarket and Kalshi with a paper-default posture: most
strategies trade on paper and earn live capital only by clearing the graduation
ladder. Calibration learns from every market resolution. Ongoing research, not a
finished product.

For a common Docker Compose deployment across macOS, WSL, and a single Ubuntu
VM—including a first-class, GUI-authenticated IB Gateway service—see
[`docs/PORTABLE_DEPLOYMENT.md`](docs/PORTABLE_DEPLOYMENT.md).

## License

MIT.
