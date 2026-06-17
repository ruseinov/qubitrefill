"""HTTP routes — the 4 request/response endpoints mirroring mvp/src/api/mocks.ts.

The WebSocket channel (subscribeAgent) lives in ws.py. The heavy optimize
pipeline runs in a worker thread so the event loop stays responsive; the
resulting events are published on the loop afterwards.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from .. import config
from ..events.bus import get_bus
from ..financial.basket import validate_basket
from ..orchestration.job import run_optimization
from ..persistence.agents import get_agent_store
from ..persistence.leaderboard import build_leaderboard
from ..solvers.types import SolverFailed
from .schemas import (
    AgentConfig,
    LeaderboardEntry,
    OptimizeRequest,
    RoutingResult,
    SubmitAgentResponse,
)

router = APIRouter()


@router.post("/agents", response_model=SubmitAgentResponse)
async def create_agent(config_in: AgentConfig) -> SubmitAgentResponse:
    try:
        validate_basket(config_in.assets)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    record = get_agent_store().create(config_in, bankroll=config.BANKROLL_USD)
    return SubmitAgentResponse(
        agent_id=record.id,
        qr_url=f"{config.QR_BASE_URL}/p/{record.id}",
        bankroll=record.bankroll,
    )


@router.get("/agents/{agent_id}", response_model=AgentConfig)
async def get_agent(agent_id: str) -> AgentConfig:
    record = get_agent_store().get(agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return record.to_config()


@router.post("/agents/{agent_id}/optimize", response_model=RoutingResult)
async def optimize(agent_id: str, body: OptimizeRequest | None = None) -> RoutingResult:
    sliders = body.sliders if body is not None else None
    assets = body.assets if body is not None else None
    try:
        outcome = await asyncio.to_thread(run_optimization, agent_id, sliders, assets)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="agent not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SolverFailed as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    bus = get_bus()
    for event in outcome.events:
        bus.publish(event.channel, event.payload)
    return outcome.result


@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def leaderboard() -> list[LeaderboardEntry]:
    return build_leaderboard()
