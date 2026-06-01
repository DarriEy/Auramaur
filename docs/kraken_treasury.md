# Kraken treasury & cross-venue transfers

Kraken is wired as a **spot/treasury venue** — not a binary prediction venue, so
it is deliberately NOT part of the trading engine. It does two things:

1. **Treasury / conversion** — buy/sell coins to rebalance (e.g. `POL -> USD`).
2. **Guarded transfers** — move USDC `Kraken -> Polymarket` (Polygon), behind
   layered guardrails.

Directional crypto trading is **off** (`kraken.directional_enabled: false`) until
a validated edge exists — Path-B research found none in the current regime.

## What's automated vs manual

| Lane | Automatable? | How |
| --- | --- | --- |
| Kraken spot orders | yes | `KrakenSpotClient.place_spot_order` / `.convert` |
| Kraken -> Polymarket (USDC/Polygon) | yes, gated | `TransferManager` / `scripts/kraken_transfer.py` |
| Kraken -> Kalshi | **no** | Kalshi is USD bank-rail (ACH/wire) — move funds manually |

## One-time manual setup (you, not the bot)

These cannot be done from code:

1. **Trade-enabled API key** — to place spot orders, the Kraken key needs
   **"Create & Modify Orders"** (read-only "Query Funds" only reads balances).
2. **Whitelist the destination** — Kraken **Funding -> Withdraw -> add address**
   for your Polymarket Polygon USDC deposit address. Give it a key name (e.g.
   `polymarket-usdc`). The API can ONLY send to addresses whitelisted here.
3. **"Withdraw Funds" permission** on the key — enable only after step 2.
4. Put the key name in `config/defaults.yaml -> transfers.allowed_withdraw_keys`.

## The transfer gate ladder (all must hold to move real funds)

1. No `KILL_SWITCH` file.
2. `AURAMAUR_ENABLE_TRANSFERS=true` (env) **and** `transfers.enabled: true` (config)
   → `settings.transfers_armed`.
3. Destination in `transfers.allowed_withdraw_keys` **and** whitelisted in Kraken UI.
4. `min_transfer_usd <= amount <= per_transfer_cap_usd`.
5. today's total + amount <= `daily_cap_usd` (tracked in `data/transfer_ledger.json`).
6. Explicit per-move approval (`--execute` + typed confirmation).

Miss any one → **dry-run preview, nothing moves.**

## Usage

```bash
# read-only wallet
python scripts/kraken_balance.py

# venue + wallet + gate status
python scripts/preflight_venues.py

# preview a transfer (moves nothing)
python scripts/kraken_transfer.py --to polymarket-usdc --amount 50

# execute (only works once the gate ladder above is fully open)
python scripts/kraken_transfer.py --to polymarket-usdc --amount 50 --execute
```

Caps default conservative: per-transfer $100, daily $250 — raise in
`config/defaults.yaml -> transfers` once you trust the flow.
