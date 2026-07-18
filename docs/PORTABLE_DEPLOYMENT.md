# Portable workstation and single-VM deployment

Auramaur and IB Gateway are one deployment with separate service lifecycles.
The same x86-64 Compose stack runs on macOS (including Apple Silicon through
Docker emulation), WSL 2, and an Ubuntu VM. Never run
two live Auramaur instances against the same venue accounts.

## Layout

- `auramaur`: trading process, Claude CLI, SQLite, all venue clients
- `ibgateway`: GUI-capable IB Gateway, persistent Jts settings, private API
- `runtime/state`: SQLite, kill switch, verified backups
- `runtime/ibgateway`: durable Gateway configuration
- `runtime/claude`: container Claude login (or bind `CLAUDE_CONFIG_DIR`)
- `runtime/config`: host-local YAML overrides
- `secrets`: PEM/VNC secrets; never committed

IB Gateway's API ports are not published to the host or internet. noVNC binds
only to `127.0.0.1:6080`; on a VM, reach it with Tailscale or:

```bash
ssh -L 6080:127.0.0.1:6080 user@vm
```

Then open `http://127.0.0.1:6080/vnc.html`. IBKR requires interactive GUI
authentication and weekly reauthentication; the deployment never automates or
stores the IBKR login/2FA credentials. Use a dedicated IBKR username.

Compose assigns Auramaur the stable private address `172.18.0.3`, which is the
only non-loopback address that should be added to Gateway's Trusted IP list.
If `172.18.0.0/16` conflicts with another Docker network, change
`AURAMAUR_BROKER_SUBNET`, `AURAMAUR_BOT_IP`, and `IB_GATEWAY_IP` together before
the first start.

## First installation

1. Install Docker Engine/Desktop with Compose.
2. Download IBKR's current offline Linux x86-64 IB Gateway installer into
   `deploy/ibgateway/ibgateway-latest-standalone-linux-x64.sh`.
3. Run `deploy/init-runtime.sh`.
   If the host user is not UID/GID 1000, export `AURAMAUR_UID=$(id -u)` and
   `AURAMAUR_GID=$(id -g)` for every Compose invocation (a Compose env file is
   convenient). This keeps bind-mounted state writable without running the bot
   as root.
4. Copy `.env.example` to `.env`; keep `AURAMAUR_LIVE=false`,
   `AURAMAUR_CONTAINER_LIVE=false`, and
   `AURAMAUR_CONTAINER_ENABLE_TRANSFERS=false`. The Compose-specific gates
   prevent a parallel deployment from inheriting authority from a legacy
   workstation `.env`.
5. Put the Kalshi PEM and other file secrets in `secrets/`; use container paths
   such as `KALSHI_PRIVATE_KEY_PATH=/run/auramaur-kalshi-private.pem`, and set
   `KALSHI_PRIVATE_KEY_SOURCE` to its host path. Existing workstations default
   to `./private-key.pem`; new deployments should use
   `./secrets/kalshi-private.pem`.
6. Copy any deliberate overrides into `runtime/config/defaults.local.yaml`.
7. `docker compose build` and `deploy/start.sh`.
8. Authenticate the dedicated live IBKR quote account through the secured
   noVNC tunnel and configure API port 4002, auto-restart, API access, and
   read-only mode. This live login supplies market data only: keep
   `IBKR__PAPER_TRADE=true`, both Compose arming gates false, and Auramaur's
   local configuration `execution.live=false` throughout the shakedown. The
   Compose default enables the read-only six-book local simulator while
   leaving the order-capable options path and separate ETF/OpenAI experiment
   disabled.
9. Run a multi-hour paper shakedown and `deploy/verify.sh`.

Claude Code credentials persist in `runtime/claude`. Authenticate from the
container with `docker compose exec auramaur claude`; for an existing host login,
set `CLAUDE_CONFIG_DIR` to that host directory before starting Compose.

## Supervisors

On an Ubuntu VM create an `auramaur` system user with Docker access, make it the
owner of `/opt/auramaur`, then install `deploy/systemd/auramaur-compose.service`
under `/etc/systemd/system`. Enable it only after paper verification. On WSL,
systemd does not by itself guarantee the
WSL VM remains active; add a Windows Task Scheduler action at boot that invokes
the distro and starts this unit. The macOS LaunchAgent example is for Docker
Desktop and must not be loaded alongside the legacy native LaunchAgent.

## Backup and restore

`deploy/backup.sh` uses SQLite's online backup API, integrity-checks the result,
compresses it, and retains 14 local copies by default. Copy these encrypted to
off-host object storage. `deploy/offsite-backup.sh` encrypts with an age public
recipient and uploads through rclone; configure only the public recipient and
remote destination on the VM. `deploy/restore.sh BACKUP` refuses to run while
Auramaur is active or overwrite an existing DB.

The supplied systemd monitor timer checks both containers, the SQLite schema,
and the authenticated IB Gateway API socket every five minutes. It can post to
`AURAMAUR_MONITOR_WEBHOOK_URL`. The backup timer runs encrypted off-host backup
daily. Install and enable these units only after their environment files under
`/etc/auramaur` have been reviewed.

## Immutable releases

Tags and manual runs of `.github/workflows/container.yml` publish an amd64
Auramaur image to GHCR with both release and full-commit tags. Set
`AURAMAUR_IMAGE` and an immutable `AURAMAUR_VERSION` on the VM. IB Gateway is
built privately from the operator-downloaded IBKR installer and is never
published by this repository.

## Cutover

1. Stage and paper-test NEW with `scripts/migrate_portable.sh stage`.
2. OLD: `scripts/migrate_portable.sh freeze`, then `export`.
3. Transfer the verified DB backup, `.env`, `secrets/`, local config, Gateway
   settings, and Claude authentication through an encrypted channel.
4. Restore on NEW while stopped. Leave OLD kill-switched.
5. Start NEW, authenticate Gateway if necessary, and run `deploy/verify.sh`.
6. Compare account IDs, cash, positions, open orders, and live/paper mode with
   every venue UI before accepting the cutover.

Rollback requires stopping and kill-switching NEW first. If NEW traded, its DB
is authoritative and must be backed up and restored to OLD before restart.
