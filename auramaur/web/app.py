"""FastAPI app for the Auramaur web dashboard.

Phase 1 is read-only monitoring: the same fused view as ``auramaur cockpit``,
served as JSON plus an SSE stream, with the built React app mounted at ``/``.

Every data endpoint serves the broker's envelope — ``{ok, error, updated_at,
state}`` — never a 500: a dashboard that goes blank when something is wrong
fails at exactly the moment it matters most. ``state`` holds the last GOOD
snapshot even while ``error`` is set, so the operator keeps context during a
blip instead of losing the screen.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from auramaur.killswitch import kill_switch_present
from auramaur.runtime import REPO_ROOT
from auramaur.web.broker import StateBroker
from auramaur.web.db import ReadOnlyDatabase
from config.settings import Settings


def _dist_dir() -> Path:
    return Path(os.environ.get("AURAMAUR_WEB_DIST", REPO_ROOT / "web" / "dist"))


def create_app(
    db_path: str | None = None,
    settings: Settings | None = None,
    refresh_seconds: float = 2.0,
) -> FastAPI:
    settings = settings or Settings()
    broker = StateBroker(ReadOnlyDatabase(db_path), settings, refresh_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        broker.start()
        yield
        await broker.stop()

    app = FastAPI(title="Auramaur dashboard", lifespan=lifespan)

    @app.get("/api/state")
    async def state() -> dict:
        await broker.wait_first()
        return broker.envelope()

    @app.get("/api/health")
    async def health() -> dict:
        await broker.wait_first()
        env = broker.envelope()
        return {
            "database": {"ok": env["ok"], "detail": env["error"] or "ok"},
            "updated_at": env["updated_at"],
            "kill_switch": kill_switch_present(),
            "is_live": settings.is_live,
        }

    @app.get("/api/stream")
    async def stream(request: Request, limit: int | None = None) -> StreamingResponse:
        # `limit` bounds the stream (tests, curl debugging); browsers omit it.
        async def events():
            await broker.wait_first()
            sent = 0
            while not await request.is_disconnected():
                yield f"event: state\ndata: {json.dumps(broker.envelope())}\n\n"
                sent += 1
                if limit is not None and sent >= limit:
                    return
                await asyncio.sleep(broker.refresh_seconds)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Built React app, if present (dev mode uses the Vite server + proxy
    # instead). Mounted last so /api routes take precedence.
    dist = _dist_dir()
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="app")

    return app
