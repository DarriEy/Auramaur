#!/usr/bin/env bash
# Host-migration helper — see docs/MIGRATION_HOST.md for the phase gates.
#
#   stage                      (NEW) preflight checks for the paper shakedown
#   freeze                     (OLD) kill-switch + bootout: nothing trades
#   pull  user@oldhost:/path   (NEW) rsync private state from the frozen OLD
#   install-agent              (NEW) template + bootstrap the LaunchAgent
#   verify                     (NEW) post-cutover health gate
#   decommission               (OLD) wipe secrets/state before hand-back
#
# The one invariant: NEVER two live bots. `pull` refuses to run unless the
# source is frozen (KILL_SWITCH present on OLD).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"
LABEL="is.auramaur.bot"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
UID_N="$(id -u)"

# Private state that git does NOT carry — everything the new host needs
# beyond the checkout. Deliberately explicit, not a glob of the repo root.
PRIVATE_STATE=(
    "auramaur.db"
    "auramaur.db-wal"
    "auramaur.db-shm"
    ".env"
    "config/defaults.local.yaml"
    "logs/"
    "agent_db_backup_20260705.db"
)

say()  { printf '\n== %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

cmd_stage() {
    say "stage: preflight for the paper shakedown (run on NEW)"
    command -v git >/dev/null || fail "git missing"
    .venv/bin/python -c 'import sys; assert sys.version_info >= (3,11)' \
        2>/dev/null || fail "no .venv with Python 3.11+ (python -m venv .venv; pip install -e .)"
    command -v claude >/dev/null || fail "claude CLI not on PATH (install + /login first)"
    claude -p "reply with exactly OK" --output-format text 2>/dev/null | grep -q OK \
        || fail "claude CLI call failed — not logged in?"
    [ -f .env ] && fail ".env present — stage phase must run WITHOUT live credentials"
    pmset -g 2>/dev/null | grep -qE ' sleep\s+0' \
        || echo "WARN: system sleep is enabled — set 'pmset -c sleep 0' or run under caffeinate"
    echo "stage preflight OK. Start the bot (paper: no .env) and watch it for a few hours."
    echo "Gate checklist: docs/MIGRATION_HOST.md Phase 1."
}

cmd_freeze() {
    say "freeze: halting the bot on THIS host (run on OLD)"
    touch KILL_SWITCH
    echo "KILL_SWITCH created — waiting for the bot to halt itself..."
    for _ in $(seq 1 60); do
        pgrep -f 'auramaur run' >/dev/null || break
        sleep 2
    done
    pgrep -f 'auramaur run' >/dev/null && fail "bot still running after 120s — investigate before proceeding"
    if launchctl print "gui/${UID_N}/${LABEL}" >/dev/null 2>&1; then
        launchctl bootout "gui/${UID_N}/${LABEL}"
        echo "LaunchAgent booted out."
    fi
    echo "FROZEN. Nothing trades until the new host completes 'verify'."
    echo "Leave KILL_SWITCH in place — it is the rollback anchor."
}

cmd_pull() {
    local src="${1:-}"
    [ -n "$src" ] || fail "usage: migrate_host.sh pull user@oldhost:/path/to/Auramaur"
    say "pull: syncing private state from ${src} (run on NEW)"
    # Refuse unless the source is frozen — the never-two-live-bots gate.
    local host="${src%%:*}" path="${src#*:}"
    ssh "$host" "test -f '${path}/KILL_SWITCH'" \
        || fail "source is NOT frozen (no KILL_SWITCH at ${src}) — run 'freeze' on OLD first"
    ssh "$host" "! pgrep -f 'auramaur run' >/dev/null" \
        || fail "a bot process is still running on the source host"
    for item in "${PRIVATE_STATE[@]}"; do
        rsync -a --info=progress2 "${src}/${item}" "./$(dirname "$item")/" \
            || echo "WARN: ${item} missing on source (ok if it never existed)"
    done
    # The kill switch must NOT travel: the new host starts armed.
    rm -f KILL_SWITCH
    sqlite3 auramaur.db "PRAGMA integrity_check;" | grep -qx ok \
        || fail "pulled auramaur.db fails integrity_check"
    echo "pull complete; DB integrity OK."
}

cmd_install_agent() {
    say "install-agent: templating + bootstrapping the LaunchAgent (run on NEW)"
    sed "s|/Users/YOURUSER|$HOME|g" scripts/launchd/is.auramaur.bot.plist.example > "$PLIST"
    plutil -lint "$PLIST" >/dev/null || fail "templated plist fails lint"
    launchctl bootout "gui/${UID_N}/${LABEL}" 2>/dev/null || true
    launchctl bootstrap "gui/${UID_N}" "$PLIST"
    echo "LaunchAgent bootstrapped. The bot should be starting now."
}

cmd_verify() {
    say "verify: post-cutover health gate (run on NEW)"
    [ -f KILL_SWITCH ] && fail "KILL_SWITCH present on the NEW host — remove it deliberately"
    launchctl print "gui/${UID_N}/${LABEL}" 2>/dev/null | grep -q 'state = running' \
        || fail "LaunchAgent not running"
    [ "$(pgrep -f 'auramaur run' | wc -l | tr -d ' ')" = "1" ] \
        || fail "expected exactly 1 bot process"
    local log="logs/run_live_launchd.out"
    for _ in $(seq 1 60); do
        grep -q 'strategy books' "$log" 2>/dev/null && break
        sleep 2
    done
    grep -q 'strategy books' "$log" || fail "startup banner never appeared in ${log}"
    grep -q 'Instance: auramaur_' "$log" && fail "FALLBACK DB — primary auramaur.db was locked"
    grep -qE 'Mode: LIVE' "$log" || echo "WARN: not in LIVE mode — is .env in place?"
    grep -E 'Build:|Mode:' "$log" | head -2
    echo "verify OK. Watch the first full cycle + compare cash/positions to the venue UI."
}

cmd_decommission() {
    say "decommission: wiping secrets/state on THIS host (run on OLD, LAST)"
    echo "This deletes .env (private keys!), DBs, logs, local config,"
    echo "~/.claude credentials, the LaunchAgent, and this checkout."
    read -r -p "Type the hostname of THIS machine to confirm: " ans
    [ "$ans" = "$(hostname)" ] || fail "confirmation mismatch — aborted"
    pgrep -f 'auramaur run' >/dev/null && fail "bot still running — freeze first"
    launchctl bootout "gui/${UID_N}/${LABEL}" 2>/dev/null || true
    rm -f "$PLIST"
    rm -P .env 2>/dev/null || rm -f .env
    rm -f auramaur*.db auramaur*.db-wal auramaur*.db-shm agent*.db
    rm -rf logs data
    rm -f config/defaults.local.yaml
    echo "Repo-local state wiped. Now, manually:"
    echo "  - rm -rf ~/.claude (CLI credentials) when done using Claude here"
    echo "  - handle Straummaur + ~/.hermes (see MIGRATION_HOST.md Phase 3)"
    echo "  - finally: rm -rf ${REPO_DIR}"
}

case "${1:-}" in
    stage)          cmd_stage ;;
    freeze)         cmd_freeze ;;
    pull)           shift; cmd_pull "$@" ;;
    install-agent)  cmd_install_agent ;;
    verify)         cmd_verify ;;
    decommission)   cmd_decommission ;;
    *) grep '^#' "$0" | sed -n '2,12p'; exit 1 ;;
esac
