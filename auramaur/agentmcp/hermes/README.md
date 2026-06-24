# Hermes agent-trader — install

Wires the persistent paper day-trader (Hermes runtime) to Auramaur's plumbing
and gives it a mandate + cadence. Three steps: register the MCP server, install
the mandate skill, accept the schedule.

## 1. Register the `auramaur-trade` MCP server

```bash
hermes mcp add auramaur-trade \
  --command /Users/darri.eythorsson/Auramaur/.venv/bin/python \
  --args -m auramaur.agentmcp.server
hermes mcp test auramaur-trade        # confirms the 6 tools are discoverable
```

Then set the server's environment + (optionally) the read-only allowlist in
`~/.hermes/config.yaml`:

```yaml
mcp_servers:
  auramaur-trade:
    command: "/Users/darri.eythorsson/Auramaur/.venv/bin/python"
    args: ["-m", "auramaur.agentmcp.server"]
    env:
      AGENT_DB_PATH: "/Users/darri.eythorsson/Auramaur/agent.db"
      AURAMAUR_DB_PATH: "/Users/darri.eythorsson/Auramaur/auramaur.db"
      AURAMAUR_LIVE: "false"          # belt-and-suspenders — never live here
    tools:
      include: [scan_markets, get_quote, get_evidence, get_portfolio, place_trade, close_position]
      prompts: false
```

> **Watch-it-think-first option:** register with only the four read tools
> (`scan_markets, get_quote, get_evidence, get_portfolio`) in `include`. The
> agent can reason and build a watchlist but can't trade until you add
> `place_trade` / `close_position`. The allowlist is your config-level write gate
> on top of the paper-only process.

`fastmcp` must be installed in the Auramaur venv:

```bash
cd /Users/darri.eythorsson/Auramaur && uv sync --extra agent
```

## 2. Install the mandate skill

The skill is a local directory; copy it into your Hermes skills tree:

```bash
mkdir -p ~/.hermes/skills/trading/auramaur-agent-trader
cp /Users/darri.eythorsson/Auramaur/auramaur/agentmcp/hermes/SKILL.md \
   ~/.hermes/skills/trading/auramaur-agent-trader/SKILL.md
```

It carries a `blueprint:` block (`schedule: every 4h`), so Hermes registers it as
a **suggested** cron job — it never schedules itself.

## 3. Accept the cadence (opt-in)

```text
/suggestions             # lists the pending auramaur-agent-trader job
/suggestions accept N    # schedules it (creates the cron job)
```

Each run is a fresh, memoryless session: the skill's `blueprint.prompt` triggers
one full trading session, the skill body is the mandate, and continuity comes
from Hermes' own persistent memory (the agent logs its theses there).

## Tuning

- **Cadence:** edit `blueprint.schedule` in `SKILL.md` (`every 4h`, a cron expr
  like `0 13,17,21 * * *`, etc.) and re-accept, or `hermes cron` directly.
- **Bankroll:** the `auramaur.bankroll_usd` config sets the notional the agent
  sizes against.

## Comparison

The agent's realized ledger lives in `agent.db`; the bot's in `auramaur.db`. The
S3 comparison report reads both, segmented identically (`auramaur pnl` shape), to
score agent-vs-bot head-to-head.
