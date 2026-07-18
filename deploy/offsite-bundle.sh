#!/bin/sh
# Full private-state bundle: everything a bare host needs beyond git + the DB.
# The DB has its own daily path (deploy/offsite-backup.sh); run THIS after any
# credential/config change and at least monthly. See docs/SECRETS.md.
set -eu
umask 077

cd "$(dirname "$0")/.."

: "${AURAMAUR_BACKUP_AGE_RECIPIENT:?Set one or more age recipients (age1..., space-separated)}"
command -v age >/dev/null || { echo "age is required" >&2; exit 1; }
if [ "${AURAMAUR_BUNDLE_NO_UPLOAD:-0}" != "1" ]; then
    : "${AURAMAUR_BACKUP_REMOTE:?Set an rclone destination, or AURAMAUR_BUNDLE_NO_UPLOAD=1}"
    command -v rclone >/dev/null || { echo "rclone is required" >&2; exit 1; }
fi

# Private state beyond the checkout and the SQLite backups. Deliberately
# explicit, mirrors docs/PORTABLE_DEPLOYMENT.md "Cutover" item 3.
members=""
for m in .env secrets config/defaults.local.yaml runtime/config \
         runtime/ibgateway runtime/claude; do
    if [ -e "$m" ]; then
        members="$members $m"
    else
        echo "WARN: $m missing — not bundled (ok if it never existed)" >&2
    fi
done
case " $members " in
    *" .env "*|*" secrets "*) ;;
    *) echo "Nothing secret to bundle (no .env, no secrets/) — aborting" >&2; exit 1 ;;
esac

out_dir=${AURAMAUR_BACKUP_DIR:-runtime/state/backups}
mkdir -p "$out_dir"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
plain="$out_dir/auramaur-bundle-$timestamp.tar.gz"
encrypted="$plain.age"
trap 'rm -f "$plain"' EXIT

# Gateway logs are bulky and operational — never leave the host.
# shellcheck disable=SC2086 # members is a space-separated repo-relative list
tar --exclude '*.log' --exclude '*.log.*' -czf "$plain" $members

set --
for r in $AURAMAUR_BACKUP_AGE_RECIPIENT; do set -- "$@" -r "$r"; done
age "$@" -o "$encrypted" "$plain"
chmod 0600 "$encrypted"

echo "Bundle members:$members"
if [ "${AURAMAUR_BUNDLE_NO_UPLOAD:-0}" = "1" ]; then
    echo "Encrypted bundle (NOT uploaded): $encrypted"
else
    rclone copyto "$encrypted" "$AURAMAUR_BACKUP_REMOTE/$(basename "$encrypted")"
    echo "Encrypted bundle uploaded: $AURAMAUR_BACKUP_REMOTE/$(basename "$encrypted")"
fi
