"""Auramaur CLI package — thin entrypoint.

Binds the shared ``main`` group + ``console`` from ._base and ``AuramaurBot``
(re-exported so ``patch("auramaur.cli.AuramaurBot")`` still works), then imports
the command-group submodules for their @main.command registration side effects.
"""

from __future__ import annotations

from auramaur.bot import AuramaurBot  # noqa: F401  (re-exported for test patching)
from auramaur.cli._base import console, log, main  # noqa: F401

# Importing these registers their commands onto `main` (side effect).
from auramaur.cli import (  # noqa: E402,F401
    diagnostics, intel, kraken, maintenance, manager, redeem, reporting, run,
    web,
)

if __name__ == "__main__":
    main()
