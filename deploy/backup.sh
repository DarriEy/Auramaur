#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
db=${AURAMAUR_DB_PATH:-runtime/state/auramaur.db}
backup_dir=${AURAMAUR_BACKUP_DIR:-runtime/state/backups}
keep=${AURAMAUR_BACKUP_KEEP:-14}

if [ ! -f "$db" ]; then
    echo "Database not found: $db" >&2
    exit 1
fi

mkdir -p "$backup_dir"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
target="$backup_dir/auramaur-$timestamp.db"

sqlite3 "$db" ".timeout 30000" ".backup '$target'"
result=$(sqlite3 "$target" "PRAGMA integrity_check;")
if [ "$result" != "ok" ]; then
    echo "Backup integrity check failed: $result" >&2
    exit 1
fi
gzip "$target"
chmod 0600 "$target.gz"

# Retention is explicit and scoped to timestamped backups in this directory.
find "$backup_dir" -type f -name 'auramaur-*.db.gz' -print \
    | sort -r \
    | awk "NR > $keep" \
    | while IFS= read -r old; do rm -f -- "$old"; done

echo "$target.gz"
