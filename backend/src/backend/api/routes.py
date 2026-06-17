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
from ..financial.basket import get_asset, validate_basket
from ..financial.estimators.expected_return import expected_return
from ..financial.prices.source import get_source
from ..orchestration.job import run_optimization
from ..persistence.agents import AgentRecord, get_agent_store
from ..persistence.leaderboard import build_leaderboard
from ..solvers.types import SolverFailed
from .schemas import (
    AgentConfig,
    LeaderboardEntry,
    MarketAsset,
    MarketResult,
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


@router.get("/agents/{agent_id}/market", response_model=MarketResult)
async def market(agent_id: str) -> MarketResult:
    record = get_agent_store().get(agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return await asyncio.to_thread(_compute_market, record)


def _compute_market(record: AgentRecord) -> MarketResult:
    tickers = list(record.holdings_units) or validate_basket(record.assets)
    source = get_source()
    returns = source.hourly_returns(tickers, config.SIGMA_WINDOW_HOURS)
    mu = expected_return(returns, config.MU_WINDOW_HOURS)
    spot = source.spot_prices(tickers)
    assets = []
    for i, t in enumerate(tickers):
        meta = get_asset(t)
        units = record.holdings_units.get(t, 0.0)
        assets.append(
            MarketAsset(
                ticker=t,
                name=meta.name,
                asset_class=meta.asset_class,
                mu=float(mu[i]),
                units=units,
                usd=units * spot[t],
            )
        )
    return MarketResult(agent_id=record.id, assets=assets)


@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def leaderboard() -> list[LeaderboardEntry]:
    return build_leaderboard()
