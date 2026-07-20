#!/bin/sh
set -eu

# Web-dashboard twin of auramaur-entrypoint.sh, with the bot-only guards
# removed on purpose:
#   - NO KILL_SWITCH refusal — the dashboard must stay up during a halt,
#     precisely so the operator can see the KILL SWITCH banner and state.
#   - NO writability checks on logs or ~/.claude — this process only reads
#     state and tails logs; its mounts are read-only where possible.
runtime_uid=${AURAMAUR_RUNTIME_UID:-1000}
runtime_gid=${AURAMAUR_RUNTIME_GID:-1000}
groupmod -o -g "$runtime_gid" auramaur
usermod -o -u "$runtime_uid" -g "$runtime_gid" auramaur

db_path=${AURAMAUR_DB_PATH:-/app/state/auramaur.db}
if ! gosu auramaur test -r "$db_path"; then
    echo "Database not readable by host UID $runtime_uid: $db_path" >&2
    echo "Start the bot once to create it, or fix AURAMAUR_UID/AURAMAUR_GID." >&2
    exit 1
fi

exec gosu auramaur "$@"
