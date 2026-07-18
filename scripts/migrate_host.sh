#!/usr/bin/env bash
# DEPRECATED — the native launchd migration path is retired.
#
# Migration and cutover now go through the portable Compose stack:
#   scripts/migrate_portable.sh stage|freeze|export|verify
#   docs/PORTABLE_DEPLOYMENT.md  (procedure)  ·  docs/SECRETS.md  (bundles)
#
# The one surviving command is `decommission`: wiping private state off a
# native host before hand-back. Everything else errors out below so the two
# migration systems can never drift apart again (the old PRIVATE_STATE list
# here had already fallen behind — it omitted secrets/ and runtime/).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"
LABEL="is.auramaur.bot"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
UID_N="$(id -u)"

say()  { printf '\n== %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

cmd_decommission() {
    say "decommission: wiping secrets/state on THIS host (run on OLD, LAST)"
    echo "This deletes .env (private keys!), secrets/, DBs, logs, runtime/,"
    echo "local config, the LaunchAgent, and legacy key files."
    read -r -p "Type the hostname of THIS machine to confirm: " ans
    [ "$ans" = "$(hostname)" ] || fail "confirmation mismatch — aborted"
    pgrep -f 'auramaur run' >/dev/null && fail "bot still running — freeze first"
    if command -v docker >/dev/null 2>&1 \
        && docker compose ps --status running --services 2>/dev/null | grep -qx auramaur; then
        fail "the Compose stack is still running — deploy/kill-switch.sh first"
    fi
    launchctl bootout "gui/${UID_N}/${LABEL}" 2>/dev/null || true
    rm -f "$PLIST"
    rm -P .env 2>/dev/null || rm -f .env
    rm -P private-key.pem 2>/dev/null || rm -f private-key.pem
    rm -rf secrets
    rm -f auramaur*.db auramaur*.db-wal auramaur*.db-shm agent*.db
    rm -rf logs data runtime
    rm -f config/defaults.local.yaml
    rm -f deploy/ibgateway/ibgateway-*-standalone-linux-x64.sh
    echo "Repo-local state wiped. Now, manually:"
    echo "  - rm -rf ~/.claude (CLI credentials) when done using Claude here"
    echo "  - handle Straummaur + ~/.hermes (see MIGRATION_HOST.md Phase 3)"
    echo "  - rclone/keychain/shell-profile secrets outside the repo"
    echo "  - finally: rm -rf ${REPO_DIR}"
}

case "${1:-}" in
    decommission)   cmd_decommission ;;
    stage|freeze|pull|install-agent|verify)
        fail "'$1' is retired — use scripts/migrate_portable.sh (docs/PORTABLE_DEPLOYMENT.md)" ;;
    *) grep '^#' "$0" | sed -n '2,11p'; exit 1 ;;
esac
