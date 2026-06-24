"""The shared root-aware KILL_SWITCH check.

CRITICAL: these tests must NEVER create a KILL_SWITCH at the real repo root —
that would halt the running bot. Every test monkeypatches KILL_SWITCH_PATH to a
tmp dir and chdirs into a tmp dir, so no real switch is ever touched.
"""

from __future__ import annotations

import auramaur.killswitch as ks


def test_absent_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(ks, "KILL_SWITCH_PATH", tmp_path / "root" / "KILL_SWITCH")
    monkeypatch.chdir(tmp_path)
    assert ks.kill_switch_present() is False


def test_root_anchor_found_even_when_cwd_lacks_it(tmp_path, monkeypatch):
    """The bug this fixes: a switch at the repo root must be detected even when
    the process CWD has no KILL_SWITCH (the bare CWD check would miss it)."""
    root = tmp_path / "root"; root.mkdir()
    cwd = tmp_path / "cwd"; cwd.mkdir()
    monkeypatch.setattr(ks, "KILL_SWITCH_PATH", root / "KILL_SWITCH")
    monkeypatch.chdir(cwd)  # CWD deliberately has no switch
    assert ks.kill_switch_present() is False
    (root / "KILL_SWITCH").touch()  # arm at the "repo root" only
    assert ks.kill_switch_present() is True


def test_cwd_branch_still_honored(tmp_path, monkeypatch):
    """A CWD-relative ./KILL_SWITCH is still honored (back-compat with tooling)."""
    monkeypatch.setattr(ks, "KILL_SWITCH_PATH", tmp_path / "root" / "KILL_SWITCH")
    monkeypatch.chdir(tmp_path)
    assert ks.kill_switch_present() is False
    (tmp_path / "KILL_SWITCH").touch()
    assert ks.kill_switch_present() is True
