# IBKR pipeline — from dormant venue to graduation-ready

The IBKR account holds real capital and two strategy families want to earn
their way onto it. This is the ordered checklist from today's state
(venue disabled, no API listener) to "a graduated strategy can trade it".
Everything below the operator steps is already built and tested.

## Current state (verified 2026-07-18)

- IBKR Desktop installed (~/Applications); `~/Jts` config exists from the
  June sessions; `ib_async` in the venv; equity client connects lazily
  (a down Gateway degrades gracefully, no crash loop).
- `ibkr.enabled: false` — the venue is parked. No API port listening.
- IBKR-bound strategies: `oddlot_tender` (paper, EDGAR-scan, manual
  tendering by design) and the `ibkr_etf` paper experiment (#289 — ships
  disabled; OpenAI-armed, cost-in-ledger, isolated tables).

## Operator steps (in order — each gates the next)

1. **Launch IBKR Desktop and log in.** In its settings enable the API
   ("Enable ActiveX and Socket Clients"), note the socket port, and match
   it to config (`ibkr.paper_port` 7497 / `ibkr.live_port` 7496). Start
   with the PAPER environment.
2. **Account-side blockers** (only visible in the IBKR web portal): the
   June CAD-hold and OPRA market-data subscription. Options stay off
   (`options_enabled: false`) until OPRA is active — without it the
   option scanner spams errors by design.
3. **Enable the venue** in `config/defaults.local.yaml`:
   `ibkr: {enabled: true}` (environment stays paper).
4. **Smoke-test**: `python scripts/preflight_venues.py` (has an IBKR
   probe) and, for the ETF experiment, `auramaur ibkr-etf-preflight`.
5. Restart via `./scripts/restart_live.sh`.

## The graduation paths

- **oddlot_tender**: already structured — EDGAR scan runs venue-less;
  a found opportunity is tendered MANUALLY (design decision). IBKR being
  up adds live quotes/positions, not automation.
- **ibkr_etf experiment**: needs `OPENAI_API_KEY` in `.env` +
  `etf_paper_enabled: true` + preflight green. Paper-only by
  construction; its ledger is isolated and cost-inclusive. Judged like
  every cell: events + sign, net of its intelligence bill.
- **Any future graduate**: the ladder's live promotion applies per
  (strategy × category) cell as usual — IBKR is just a venue behind the
  same gates. NOTE the equity client is non-read-only by design (oddlot
  execution); live IBKR orders remain behind the global triple gate.

## Standing cautions

- IBKR Desktop must be running for any of this — it is an interactive
  app with 2FA; the bot cannot start it. If it becomes routine, consider
  IB Gateway (headless-friendlier) + auto-restart, a separate decision.
- Weekly Desktop auto-restarts drop the API for minutes — the lazy
  reconnect handles it; expect gap logs, not incidents.
