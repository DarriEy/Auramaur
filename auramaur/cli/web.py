"""Auramaur CLI — web dashboard server."""

from __future__ import annotations

import click

from auramaur.cli._base import console, main


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Bind address. Keep loopback unless the LAN should see it.")
@click.option("--port", default=8484, show_default=True, type=int)
def web(host: str, port: int):
    """Serve the read-only web dashboard (FastAPI + the built React app)."""
    try:
        import uvicorn

        from auramaur.web.app import create_app
    except ImportError as exc:
        raise click.ClickException(
            f"web dependencies not installed ({exc}) — run: pip install -e '.[web]'"
        ) from exc

    console.print(f"[bold blue]Auramaur dashboard[/] on http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
