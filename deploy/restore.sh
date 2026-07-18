#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
source_backup=${1:-}
target=${AURAMAUR_DB_PATH:-runtime/state/auramaur.db}

if [ -z "$source_backup" ] || [ ! -f "$source_backup" ]; then
    echo "usage: deploy/restore.sh path/to/auramaur-TIMESTAMP.db[.gz]" >&2
    exit 1
fi
if docker compose ps --status running --services 2>/dev/null | grep -qx auramaur; then
    echo "Auramaur is running; stop it before restoring state." >&2
    exit 1
fi

mkdir -p "$(dirname "$target")"
tmp=$(mktemp "${TMPDIR:-/tmp}/auramaur-restore.XXXXXX.db")
trap 'rm -f "$tmp"' EXIT
case "$source_backup" in
    *.gz) gzip -cd "$source_backup" > "$tmp" ;;
    *) cp "$source_backup" "$tmp" ;;
esac

result=$(sqlite3 "$tmp" "PRAGMA integrity_check;")
if [ "$result" != "ok" ]; then
    echo "Restore source fails integrity check: $result" >&2
    exit 1
fi
if [ -e "$target" ]; then
    echo "Refusing to overwrite existing database: $target" >&2
    echo "Move it aside deliberately, then retry." >&2
    exit 1
fi
mv "$tmp" "$target"
chmod 0600 "$target"
echo "Restored $target"
