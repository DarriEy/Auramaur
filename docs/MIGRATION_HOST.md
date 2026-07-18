# Host migration checklist (DEPRECATED — native launchd path)

> **DEPRECATED 2026-07.** The deployment is the portable Compose stack;
> migrate with `scripts/migrate_portable.sh` per `docs/PORTABLE_DEPLOYMENT.md`
> (private state travels as a `deploy/offsite-bundle.sh` bundle — see
> `docs/SECRETS.md`). Of `scripts/migrate_host.sh`, only `decommission`
> survives, for wiping a native host before hand-back; Phase 3 below remains
> its checklist. Phases 1–2 are historical.

Moving the bot to a new machine (e.g. laptop replacement). The companion
script `scripts/migrate_host.sh` mechanizes each phase; this file is the
authority on the gates between phases.

**The one invariant: never two live bots.** The live venue account is
shared state — two instances double every risk cap and race each other's
orders (the 2026-07 dual-instance incident, across machines this time).
The cutover sequence below is ordered so there is no moment where both
hosts can trade.

Terminology: OLD = the machine currently running the bot; NEW = the target.

---

## Phase 1 — stage the NEW host (safe anytime; zero risk to the live bot)

On NEW:

1. Install prerequisites: git, Python 3.11+, the Claude CLI
   (`~/.local/bin/claude`), and log the CLI in interactively
   (`claude` → `/login`) against the Max+ account.
2. `git clone` the repo, `python -m venv .venv`, `pip install -e .`
   (or the project's documented install).
3. Power settings — the bot must survive lid-closed/idle:
   `sudo pmset -c sleep 0 && sudo pmset -c disablesleep 1` (AC power),
   or run under `caffeinate -i`.
4. **Paper shakedown** (no live credentials on the machine yet):
   `scripts/migrate_host.sh stage` runs the checks, then start the bot
   WITHOUT any `.env` — the env gate stays closed, everything paper-routes.
   Let it run ≥ a few hours.

Gate to Phase 2 — on NEW, all true:
- [ ] bot cycles complete on both venues (scan lines in the log)
- [ ] an LLM call succeeded (`llm.routed` / a lens or arm event in
      `auramaur.log`) — proves the Claude CLI + PATH work
- [ ] no `Instance: auramaur_N.db` fallback line in the startup banner
- [ ] machine survived a lid-close/idle period without the process dying

## Phase 2 — cutover (one sitting, ~1 hour; do NOT split across days)

1. On OLD: `scripts/migrate_host.sh freeze`
   — creates `KILL_SWITCH` (bot halts; launchd will not resurrect),
   waits for the process to exit, then `launchctl bootout`s the agent.
   From this moment NOTHING trades until Phase 2 completes.
2. On NEW: `scripts/migrate_host.sh pull olduser@oldhost:/path/to/Auramaur`
   — rsyncs the private state (see the PRIVATE_STATE list in the script:
   DB + WAL, `.env`, `config/defaults.local.yaml`, logs/, incident
   backups). The repo itself came from git, not rsync.
3. On NEW: install the LaunchAgent —
   `scripts/migrate_host.sh install-agent` (templates the versioned
   example plist with this user's paths, bootstraps it).
4. On NEW: `scripts/migrate_host.sh verify`
   — single instance, primary DB, live mode banner, LLM call, recent
   heartbeat, and NO kill switch present.

Gate to Phase 3 — on NEW, all true:
- [ ] `verify` passes clean
- [ ] first live-mode heartbeat shows the expected cash/positions
      (compare against the venue UI)
- [ ] a full trading cycle completed without `database is locked` storms

Rollback (if NEW misbehaves): `bootout` the agent on NEW, remove
`KILL_SWITCH` on OLD, re-bootstrap the agent on OLD. The DB on OLD is
still authoritative until Phase 3 — nothing on NEW has diverged unless
the NEW bot traded; if it did, rsync the DB BACK before restarting OLD.

## Phase 3 — decommission OLD (before hand-back)

On OLD: `scripts/migrate_host.sh decommission`
— after an interactive confirmation it: bootouts + deletes the
LaunchAgent, securely removes `.env` and `private-key.pem` (Polygon
private key + venue API keys — the real hazard on a returned machine),
removes `secrets/`, the DBs/WAL, logs, `runtime/` (Gateway + Claude
state), local config, and the cached Gateway installer.
Also remember, outside this repo's scope:

- [ ] Straummaur: stop its recorder, rsync `~/Straummaur/data` to NEW,
      restart the recorder there (it is vol_anchor's calibration feed).
      ⚠️ The data directory has NO other backup — it was lost once already
      (2026-07-18, trashed + emptied under a live recorder). Rsync FIRST.
- [ ] Hermes: `~/.hermes` and the (unloaded) `ai.hermes.gateway.plist`
      if still present
- [ ] any shell profiles/keychains holding venue or exchange secrets

---

Post-migration watch (first 48h on NEW): the standard daily check plus
`launchctl print gui/$UID/is.auramaur.bot` (runs count should stay flat),
and one deliberate `kill -9` + respawn test once, off-hours.
