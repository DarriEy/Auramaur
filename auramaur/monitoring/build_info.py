"""Which build is this process actually running?

The 2026-06 category leak kept trading for a full day after the fix merged
(#93, 2026-06-10) because the long-lived bot process was never restarted —
the code on disk and the code in memory diverged silently. This module reads
the git HEAD once at import time (≈ process start) and again on demand, so
the bot can announce its build at startup and a periodic guard can warn when
the on-disk tree has moved past the running process.

Pure file reads, no git subprocess: it must be cheap and safe to call from a
periodic tick, and must never take the bot down — every failure path returns
"" and the guard treats that as "unknown, stay quiet".
"""

from __future__ import annotations

import time
from pathlib import Path

import structlog

log = structlog.get_logger()

_SHA_LEN = 12


def _repo_root(start: Path | None = None) -> Path | None:
    """Walk up from this file (or *start*) to the directory holding .git."""
    p = (start or Path(__file__)).resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def read_git_head(root: Path | None = None) -> str:
    """Return the short SHA the checkout currently points at, '' if unknown.

    Handles plain checkouts, detached HEAD, packed refs, and linked
    worktrees (where .git is a *file* pointing at the real gitdir and refs
    live in the shared common dir).
    """
    try:
        repo = root or _repo_root()
        if repo is None:
            return ""
        git = repo / ".git"
        if git.is_file():  # linked worktree: "gitdir: /path/.git/worktrees/x"
            target = git.read_text().strip().partition(":")[2].strip()
            if not target:
                return ""
            git = (repo / target).resolve() if not Path(target).is_absolute() \
                else Path(target)
        common = git
        commondir = git / "commondir"
        if commondir.exists():  # worktree gitdir: refs live in the common dir
            common = (git / commondir.read_text().strip()).resolve()

        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head[:_SHA_LEN]  # detached HEAD: literal SHA
        ref = head[5:].strip()
        ref_path = common / ref
        if ref_path.exists():
            return ref_path.read_text().strip()[:_SHA_LEN]
        packed = common / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                line = line.strip()
                if line.endswith(f" {ref}"):
                    return line.split(" ", 1)[0][:_SHA_LEN]
        return ""
    except OSError:
        return ""


# Captured when the process first imports this module — i.e. the build the
# running interpreter actually loaded, regardless of what lands on disk later.
STARTUP_SHA: str = read_git_head()


class BuildStalenessGuard:
    """Warns when the on-disk checkout has moved past the running process.

    ``check()`` is called from a periodic bot task. It compares the SHA
    captured at process start with the current on-disk HEAD; on mismatch it
    alerts (loudly, but rate-limited to once per ``realert_seconds`` per the
    quiet-feed rules) and logs a structured warning. Returns True when stale
    so callers/tests can branch on it.
    """

    def __init__(self, alert=None, realert_seconds: float = 3600.0,
                 startup_sha: str | None = None,
                 read_head=read_git_head,
                 clock=time.monotonic):
        self._alert = alert or (lambda msg: None)
        self._realert_seconds = realert_seconds
        self._startup_sha = (STARTUP_SHA if startup_sha is None
                             else startup_sha)
        self._read_head = read_head
        self._clock = clock
        self._last_alert_at: float | None = None
        self._last_alerted_sha = ""

    def check(self) -> bool:
        disk_sha = self._read_head()
        if not self._startup_sha or not disk_sha:
            return False  # unknown build (no .git, packaged install): stay quiet
        if disk_sha == self._startup_sha:
            self._last_alert_at = None
            self._last_alerted_sha = ""
            return False
        now = self._clock()
        due = (self._last_alert_at is None
               or disk_sha != self._last_alerted_sha
               or now - self._last_alert_at >= self._realert_seconds)
        if due:
            msg = (f"running build {self._startup_sha} but the checkout is "
                   f"now at {disk_sha} — restart the bot to pick up merged "
                   f"fixes")
            self._alert(msg)
            log.warning("bot.stale_build", running=self._startup_sha,
                        on_disk=disk_sha)
            self._last_alert_at = now
            self._last_alerted_sha = disk_sha
        return True
