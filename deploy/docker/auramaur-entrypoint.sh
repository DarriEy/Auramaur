#!/bin/sh
set -eu

mkdir -p "${AURAMAUR_STATE_DIR:-/app/state}" "${AURAMAUR_LOG_DIR:-/app/logs}"

if [ -f "${AURAMAUR_KILL_SWITCH_PATH:-/app/state/KILL_SWITCH}" ]; then
    echo "KILL_SWITCH present; refusing to start Auramaur" >&2
    exit 0
fi

runtime_uid=${AURAMAUR_RUNTIME_UID:-1000}
runtime_gid=${AURAMAUR_RUNTIME_GID:-1000}
groupmod -o -g "$runtime_gid" auramaur
usermod -o -u "$runtime_uid" -g "$runtime_gid" auramaur
for dir in "${AURAMAUR_STATE_DIR:-/app/state}" "${AURAMAUR_LOG_DIR:-/app/logs}" "$HOME/.claude"; do
    if ! gosu auramaur test -w "$dir"; then
        echo "Runtime path is not writable by host UID $runtime_uid: $dir" >&2
        echo "Set AURAMAUR_UID/AURAMAUR_GID to the host owner and re-run." >&2
        exit 1
    fi
done

exec gosu auramaur "$@"
