# IBKR pipeline — from dormant venue to graduation-ready

The IBKR account holds real capital and two strategy families want to earn
their way onto it. This is the ordered checklist from today's state
(venue and experiments disabled by default) to "a graduated strategy can trade it".
Everything below the operator steps is already built and tested.

## Tracked default state (verified 2026-07-19)

- IBKR Desktop installed (~/Applications); `~/Jts` config exists from the
  June sessions; `ib_async` in the venv; equity client connects lazily
  (a down Gateway degrades gracefully, no crash loop).
- `ibkr.enabled: false` — Auramaur does not connect to the venue. A separately
  launched TWS/Gateway may still listen, but no IBKR strategy task starts.
- IBKR-bound strategies: `oddlot_tender` (paper, EDGAR-scan, manual
  tendering by design), the `ibkr_etf` paper experiment, and the multi-asset
  paper books. All tracked strategy gates ship disabled.

## Operator steps (in order — each gates the next)

1. **Launch IBKR Desktop and log in.** In its settings enable the API
   ("Enable ActiveX and Socket Clients"), note the socket port, and match
   it to config (`ibkr.paper_port` 7497 / `ibkr.live_port` 7496). Start
   with the PAPER environment.
2. **Account-side blockers** (only visible in the IBKR web portal): the
   June CAD-hold and OPRA market-data subscription. Options stay off
   (`options_enabled: false`) until OPRA is active — without it the
   option scanner spams errors by design.
3. **Enable only the venue** in `config/defaults.local.yaml`:
   `ibkr: {enabled: true}` (environment stays paper). In Docker, set
   `IBKR_ENABLED=true`; do not change `IBKR_QUOTE_ENVIRONMENT` from `paper`.
4. **Smoke-test**: `python scripts/preflight_venues.py` (has an IBKR probe),
   then run the relevant `auramaur ibkr-etf-preflight` or
   `auramaur ibkr-multiasset-preflight` command.
5. **Enable exactly one paper experiment** in the local override
   (`etf_paper_enabled: true` or `multiasset_paper_enabled: true`). For Docker,
   use `IBKR_MULTIASSET_PAPER_ENABLED=true` only after preflight. Restart.

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
