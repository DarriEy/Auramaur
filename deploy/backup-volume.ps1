# Daily backup of the auramaur-state volume DB (untracked ops script).
# Created 2026-07-23 after /app/state moved to the named volume: the old
# WSL systemd backup timer points at the dead pre-migration paths.
#
# Method: SQLite online-backup API inside the running container (consistent
# under concurrent writes) -> docker cp to runtime\backups -> prune to 14.
# Scheduled via Task Scheduler task "AuramaurVolumeBackup" (daily 07:15).
# Revert: schtasks /Delete /TN AuramaurVolumeBackup /F ; delete this file.

$ErrorActionPreference = "Stop"
$repo = "C:\Users\darey\repos\Auramaur"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$dest = Join-Path $repo "runtime\backups\volume-daily"
New-Item -ItemType Directory -Force $dest | Out-Null

docker exec auramaur-auramaur-1 python3 -c @"
import sqlite3
src = sqlite3.connect('file:/app/state/auramaur.db?mode=ro', uri=True, timeout=60)
dst = sqlite3.connect('/tmp/backup.db')
src.backup(dst)
dst.close(); src.close()
print('in-container backup complete')
"@
if ($LASTEXITCODE -ne 0) { throw "in-container sqlite backup failed" }

docker cp auramaur-auramaur-1:/tmp/backup.db (Join-Path $dest "auramaur-$stamp.db")
if ($LASTEXITCODE -ne 0) { throw "docker cp failed" }
docker exec auramaur-auramaur-1 rm -f /tmp/backup.db

# Prune to the newest 14
Get-ChildItem $dest -Filter "auramaur-*.db" | Sort-Object Name -Descending |
    Select-Object -Skip 14 | Remove-Item -Force

Write-Output "backup ok: auramaur-$stamp.db ($([math]::Round((Get-Item (Join-Path $dest "auramaur-$stamp.db")).Length/1MB)) MB)"
