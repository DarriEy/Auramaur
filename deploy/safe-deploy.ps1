# Gated deploy for the Windows-repo compose stack (untracked ops script).
# Port of the WSL deploy/safe-deploy.sh gates after the 2026-07-23 migration
# (that day an image built from a mid-edit tree shipped a partial #351).
# Usage:  pwsh deploy/safe-deploy.ps1 [-RunTests]
param([switch]$RunTests)
$ErrorActionPreference = "Stop"
$repo = "C:\Users\darey\repos\Auramaur"
Set-Location $repo

# Gate 1: no merge/rebase in progress
foreach ($marker in @(".git\MERGE_HEAD", ".git\rebase-merge", ".git\rebase-apply")) {
    if (Test-Path $marker) { throw "GATE: merge/rebase in progress ($marker) — resolve first" }
}

# Gate 2: tracked tree must be clean (untracked ops files are fine)
$dirty = git status --porcelain | Where-Object { $_ -notmatch "^\?\?" }
if ($dirty) { throw "GATE: tracked files modified — commit or stash first:`n$($dirty -join "`n")" }

# Gate 3: whole-tree python parse
python -m compileall -q auramaur
if ($LASTEXITCODE -ne 0) { throw "GATE: python parse failure in auramaur/" }

# Gate 4 (optional): test suite
if ($RunTests) {
    python -m pytest tests/ -q -x --ignore=tests/test_web
    if ($LASTEXITCODE -ne 0) { throw "GATE: tests failed" }
}

# Gate 5: compose config valid (includes compose.override.yaml merge)
docker compose config --quiet
if ($LASTEXITCODE -ne 0) { throw "GATE: compose config invalid" }

# Build + deploy
docker compose build auramaur
if ($LASTEXITCODE -ne 0) { throw "build failed" }
docker compose up -d auramaur auramaur-web
if ($LASTEXITCODE -ne 0) { throw "compose up failed" }

# Gate 6: health wait (up to 90s)
$deadline = (Get-Date).AddSeconds(90)
do {
    Start-Sleep -Seconds 5
    $health = docker inspect auramaur-auramaur-1 --format '{{.State.Health.Status}}'
} while ($health -ne "healthy" -and (Get-Date) -lt $deadline)
if ($health -ne "healthy") { throw "container not healthy after 90s (status: $health)" }

# Post-deploy: py-spy for stall forensics (lost on every recreate)
docker exec -u root auramaur-auramaur-1 sh -c "pip install -q py-spy 2>/dev/null" | Out-Null

$mode = docker logs auramaur-auramaur-1 --since 2m 2>&1 | Select-String "Mode:" | Select-Object -First 1
if ($mode) { Write-Output "deploy ok — $($mode.Line.Trim())" }
else { Write-Output "deploy ok (healthy; banner not yet in log window)" }
