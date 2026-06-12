"""Tests for the build-staleness guard (process build vs on-disk checkout).

Regression context: the 2026-06 category leak traded for a day on pre-fix
code because the merged fix sat on disk while the long-lived bot process
kept running the old build.
"""

from __future__ import annotations

from pathlib import Path

from auramaur.monitoring.build_info import (
    BuildStalenessGuard, read_git_head,
)

SHA_A = "a" * 40
SHA_B = "b" * 40


def _make_repo(tmp_path: Path, sha: str = SHA_A, *, packed: bool = False,
               detached: bool = False) -> Path:
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    if detached:
        (git / "HEAD").write_text(f"{sha}\n")
    else:
        (git / "HEAD").write_text("ref: refs/heads/main\n")
        if packed:
            (git / "packed-refs").write_text(
                "# pack-refs with: peeled fully-peeled sorted\n"
                f"{sha} refs/heads/main\n")
        else:
            (git / "refs" / "heads" / "main").write_text(f"{sha}\n")
    return tmp_path


def test_read_git_head_ref(tmp_path):
    assert read_git_head(_make_repo(tmp_path)) == SHA_A[:12]


def test_read_git_head_packed_refs(tmp_path):
    assert read_git_head(_make_repo(tmp_path, packed=True)) == SHA_A[:12]


def test_read_git_head_detached(tmp_path):
    assert read_git_head(_make_repo(tmp_path, detached=True)) == SHA_A[:12]


def test_read_git_head_linked_worktree(tmp_path):
    """In a linked worktree .git is a FILE pointing at the real gitdir, and
    refs live in the shared common dir."""
    main = _make_repo(tmp_path / "main")
    wt_gitdir = tmp_path / "main" / ".git" / "worktrees" / "wt"
    wt_gitdir.mkdir(parents=True)
    (wt_gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    (wt_gitdir / "commondir").write_text("../..\n")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {wt_gitdir}\n")
    assert read_git_head(wt) == SHA_A[:12]


def test_read_git_head_no_repo(tmp_path):
    assert read_git_head(tmp_path) == ""


def test_guard_quiet_when_build_matches():
    alerts: list[str] = []
    guard = BuildStalenessGuard(alert=alerts.append, startup_sha=SHA_A[:12],
                                read_head=lambda: SHA_A[:12])
    assert guard.check() is False
    assert alerts == []


def test_guard_alerts_on_mismatch_and_rate_limits():
    alerts: list[str] = []
    now = [0.0]
    guard = BuildStalenessGuard(alert=alerts.append, startup_sha=SHA_A[:12],
                                read_head=lambda: SHA_B[:12],
                                realert_seconds=3600.0,
                                clock=lambda: now[0])
    assert guard.check() is True
    assert len(alerts) == 1
    assert SHA_A[:12] in alerts[0] and SHA_B[:12] in alerts[0]

    # Repeated checks within the window stay quiet (still stale though).
    now[0] = 300.0
    assert guard.check() is True
    assert len(alerts) == 1

    # After the re-alert window it speaks again.
    now[0] = 3601.0
    assert guard.check() is True
    assert len(alerts) == 2


def test_guard_realerts_immediately_on_new_sha():
    """A SECOND merge while already stale is new information — no rate limit."""
    alerts: list[str] = []
    disk = [SHA_B[:12]]
    guard = BuildStalenessGuard(alert=alerts.append, startup_sha=SHA_A[:12],
                                read_head=lambda: disk[0],
                                realert_seconds=3600.0,
                                clock=lambda: 0.0)
    assert guard.check() is True
    disk[0] = "c" * 12
    assert guard.check() is True
    assert len(alerts) == 2


def test_guard_quiet_when_build_unknown():
    """No .git (packaged install) → unknown build → never alert."""
    alerts: list[str] = []
    guard = BuildStalenessGuard(alert=alerts.append, startup_sha="",
                                read_head=lambda: SHA_B[:12])
    assert guard.check() is False
    guard2 = BuildStalenessGuard(alert=alerts.append, startup_sha=SHA_A[:12],
                                 read_head=lambda: "")
    assert guard2.check() is False
    assert alerts == []


def test_guard_resets_after_restart_catchup():
    """If disk returns to the running SHA (e.g. revert), the alarm clears."""
    alerts: list[str] = []
    disk = [SHA_B[:12]]
    guard = BuildStalenessGuard(alert=alerts.append, startup_sha=SHA_A[:12],
                                read_head=lambda: disk[0],
                                clock=lambda: 0.0)
    assert guard.check() is True
    disk[0] = SHA_A[:12]
    assert guard.check() is False
    # Going stale again re-alerts immediately (state was reset).
    disk[0] = SHA_B[:12]
    assert guard.check() is True
    assert len(alerts) == 2
