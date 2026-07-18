#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

: "${AURAMAUR_BACKUP_AGE_RECIPIENT:?Set one or more age recipients (age1..., space-separated)}"
: "${AURAMAUR_BACKUP_REMOTE:?Set an rclone destination, e.g. remote:auramaur}"
command -v age >/dev/null || { echo "age is required" >&2; exit 1; }
command -v rclone >/dev/null || { echo "rclone is required" >&2; exit 1; }

backup=$(deploy/backup.sh | tail -1)
encrypted="$backup.age"
trap 'rm -f "$encrypted"' EXIT

set --
for r in $AURAMAUR_BACKUP_AGE_RECIPIENT; do set -- "$@" -r "$r"; done
age "$@" -o "$encrypted" "$backup"
rclone copyto "$encrypted" "$AURAMAUR_BACKUP_REMOTE/$(basename "$encrypted")"
echo "Encrypted backup uploaded: $AURAMAUR_BACKUP_REMOTE/$(basename "$encrypted")"
