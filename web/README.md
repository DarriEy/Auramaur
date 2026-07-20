# Auramaur web dashboard

Read-only browser dashboard over the bot's SQLite state — the web twin of
`auramaur cockpit`. The FastAPI backend lives in `auramaur/web/` (installed via
the `web` extra) and opens the database with SQLite's `mode=ro`, so this stack
is structurally unable to write to the trading DB.

Residual-risk honesty (phase-1 threat model): the container's state mount is
read-write (WAL readers need the `-shm`/`-wal` files), so `mode=ro` +
`query_only` are SQLite-level guarantees, not kernel-level — a compromised
web *process* could still corrupt the DB file or delete `KILL_SWITCH` via the
filesystem. The container is kept off the broker network (it can never reach
ibgateway's trusted-IP API) and publishes on loopback only, which is also the
entirety of its auth: anything that can reach 127.0.0.1:8484 can read the
dashboard. Acceptable for a single-operator LAN deployment; revisit both
before any remote exposure.

## Development

Terminal 1 — the API (needs `pip install -e ".[web]"`; on Windows run the app
factory directly, the `auramaur` CLI imports Unix-only modules):

    auramaur web                       # serves http://127.0.0.1:8484
    # Windows dev:
    python -c "import uvicorn; from auramaur.web.app import create_app; uvicorn.run(create_app(), port=8484)"

Point it at a database with `AURAMAUR_DB_PATH` (defaults to the repo-root
`auramaur.db`).

Terminal 2 — the SPA with hot reload (proxies `/api` to :8484):

    cd web && npm install && npm run dev

## Production

`npm run build` emits `dist/`; the FastAPI app serves it statically when it
exists (`AURAMAUR_WEB_DIST` overrides the location). In Docker this happens in
the `webbuild` stage and the `auramaur-web` compose service publishes the
result on `127.0.0.1:8484` (loopback only, by design).

## Endpoints

Data endpoints serve an envelope — `{ok, error, updated_at, state}` — and
never 500. With a missing or schemaless database the service still comes up,
`ok` is false, `error` says exactly what to fix, and the broker reconnects on
its own once the bot creates real state. `state` keeps the last good snapshot
through transient errors so the UI never goes blank.

- `GET /api/state` — envelope with the full cockpit snapshot (positions, P&L,
  signals, pillars, venues, activity, health)
- `GET /api/stream` — the same envelope as SSE every 2s (`?limit=N` bounds
  it, for curl/tests)
- `GET /api/health` — DB status, kill-switch, live gate

The UI holds itself to the same standard: every failure mode says what is
wrong and what to check — API unreachable (troubleshooting hints), service up
but no usable DB (the error verbatim), stream stalled (stale-data banner over
the last snapshot, with its age).
