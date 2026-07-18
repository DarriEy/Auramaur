#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

mkdir -p runtime/state/backups runtime/logs runtime/config runtime/claude \
    runtime/ibgateway secrets

for dir in runtime/state runtime/logs runtime/config runtime/claude runtime/ibgateway secrets; do
    chmod 0700 "$dir"
done

if [ ! -f secrets/ibgateway-vnc-password.txt ]; then
    umask 077
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -base64 24 > secrets/ibgateway-vnc-password.txt
    else
        echo "Create a strong password in secrets/ibgateway-vnc-password.txt" >&2
        exit 1
    fi
fi

if [ ! -f runtime/config/defaults.local.yaml ]; then
    printf '%s\n' '# Host-local overrides. Keep execution.live false through shakedown.' \
        'execution:' '  live: false' > runtime/config/defaults.local.yaml
    chmod 0600 runtime/config/defaults.local.yaml
fi

echo "Runtime directories initialized."
echo "Compose defaults to UID:GID 1000:1000. If this host differs, export"
echo "AURAMAUR_UID=$(id -u) and AURAMAUR_GID=$(id -g) before Compose commands."
echo "Install venue secrets under ./secrets and review .env paths before starting."
