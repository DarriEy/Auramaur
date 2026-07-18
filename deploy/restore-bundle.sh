#!/bin/sh
# Decrypt and stage a private-state bundle made by deploy/offsite-bundle.sh.
# Extracts into a fresh staging directory — never directly over a live tree —
# so the operator diffs/moves files deliberately. See docs/SECRETS.md.
set -eu
umask 077

cd "$(dirname "$0")/.."

bundle=${1:-}
dest=${2:-runtime/restore-$(date -u +%Y%m%dT%H%M%SZ)}
identity=${AURAMAUR_BACKUP_AGE_IDENTITY:-}

if [ -z "$bundle" ] || [ ! -f "$bundle" ]; then
    echo "usage: deploy/restore-bundle.sh bundle.tar.gz.age [dest-dir]" >&2
    echo "       AURAMAUR_BACKUP_AGE_IDENTITY=path/to/age-identity.txt" >&2
    exit 1
fi
[ -n "$identity" ] && [ -f "$identity" ] \
    || { echo "AURAMAUR_BACKUP_AGE_IDENTITY must point at an age identity file" >&2; exit 1; }
command -v age >/dev/null || { echo "age is required" >&2; exit 1; }
if [ -e "$dest" ]; then
    echo "Refusing to extract into existing path: $dest" >&2
    exit 1
fi

plain=$(mktemp "${TMPDIR:-/tmp}/auramaur-bundle.XXXXXX.tar.gz")
trap 'rm -f "$plain"' EXIT
age -d -i "$identity" -o "$plain" "$bundle"

# No absolute paths, no traversal — a tampered bundle must not write outside dest.
if tar -tzf "$plain" | grep -qE '^/|(^|/)\.\.(/|$)'; then
    echo "Bundle contains absolute or traversal paths — refusing to extract" >&2
    exit 1
fi
for expected in .env secrets/kalshi-private.pem; do
    tar -tzf "$plain" | grep -qx "$expected" \
        || echo "WARN: expected member missing from bundle: $expected" >&2
done

mkdir -p "$dest"
tar -xzf "$plain" -C "$dest"
chmod -R go-rwx "$dest"

echo "Staged into: $dest"
echo "Review, then move members into place (paths are repo-relative)."
echo "The database is separate: deploy/restore.sh <auramaur-*.db.gz>"
