"""Auramaur CLI — shared base: warnings setup, console, and the click group.

A leaf module (imports nothing from the cli submodules) so every command module
can ``from auramaur.cli._base import main, console`` without an import cycle.
"""

from __future__ import annotations

import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::RuntimeWarning"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import click
import structlog
from rich.console import Console

console = Console()
log = structlog.get_logger()


@click.group()
def main():
    """Auramaur — Polymarket prediction market trading bot."""
    pass
