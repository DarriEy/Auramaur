"""Single source of truth for the KILL_SWITCH safety check.

CLAUDE.md absolute rule #2: if the KILL_SWITCH file exists, halt ALL trading.
Many call sites used a bare ``Path("KILL_SWITCH")``, which is resolved relative
to the process CWD — so a launch from a directory other than the repo root would
silently miss the switch. Anchor the check to the repo root (and still honor the
CWD for backward-compat with tooling that touches ``./KILL_SWITCH``), and route
every site through this one function so the behavior can't drift.
"""

from __future__ import annotations

from pathlib import Path

# Repo root = one level up from this file's package dir (auramaur/ -> repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent

# The canonical kill-switch path. Arm/disarm tooling should create/remove THIS
# file so the check below reliably finds it regardless of launch directory.
KILL_SWITCH_PATH = _REPO_ROOT / "KILL_SWITCH"


def kill_switch_present() -> bool:
    """True if the KILL_SWITCH file exists at the repo root OR the current CWD."""
    return KILL_SWITCH_PATH.exists() or Path("KILL_SWITCH").exists()
