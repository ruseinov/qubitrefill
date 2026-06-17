"""FastAPI application factory.

Wires the HTTP + WS routers, CORS for the MVP origins, and a lifespan that runs
the MTM scheduler for the life of the server. Run locally with:

    uvicorn backend.api.app:app --reload --workers 1

(``--workers 1`` while persistence is in-memory — see TODO.md.)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import config
from ..events.bus import get_bus
from ..orchestration.scheduler import run_mtm_loop
from . import routes, ws

log = logging.getLogger(__name__)


def _check_market_source() -> None:
    """Announce the active data source; a missing assets-api should fail loudly."""
    from ..financial.prices.assets_api import AssetsApiSource
    from ..financial.prices.source import get_source

    source = get_source()
    log.info("market data source: %s", type(source).__name__)
    if isinstance(source, AssetsApiSource):
        try:
            source.health()
            log.info("assets-api reachable at %s", config.ASSETS_API_BASE_URL)
        except Exception as e:
            log.error(
                "assets-api unreachable at %s — solves and MTM will fail until it is up "
                "(or set MARKET_DATA_SOURCE=synthetic): %s",
                config.ASSETS_API_BASE_URL,
                e,
            )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _check_market_source()
    stop = asyncio.Event()
    task = asyncio.create_task(run_mtm_loop(get_bus(), stop))
    try:
        yield
    finally:
        stop.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    app = FastAPI(title="QTW 2026 Trading Game", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.CORS_ORIGINS),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(routes.router)
    app.include_router(ws.router)
    return app


app = create_app()
