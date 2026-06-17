"""End-to-end optimization job for one request.

basket → slider_map → estimates(μ, Σ) → solver race → allocate → persist.
Every solve is a fresh allocation: a retune liquidates all holdings at spot
and reallocates the full value over the (possibly re-selected) basket. Returns
the result plus the events to publish so the API layer can run this in a
worker thread and publish on the event loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from ..api.schemas import RoutingResult, SliderValues
from ..financial.basket import validate_basket
from ..financial.estimators.covariance import covariance
from ..financial.estimators.expected_return import expected_return
from ..financial.pnl import mark_to_market
from ..financial.prices.base import MarketDataSource
from ..financial.prices.source import get_source
from ..financial.qubo_decoder import weights_to_portfolio
from ..financial.slider_map import map_sliders
from ..financial.types import PortfolioProblem
from ..persistence.agents import AgentStore, get_agent_store
from ..persistence.jobs import JobStore, get_job_store
from ..solvers.router import race
from ..solvers.types import ProviderProvenance, Solution

_PROVIDER_LABELS = {
    "gurobi": "Gurobi",
    "sa": "Simulated Annealing",
    "dwave": "D-Wave Advantage",
}


@dataclass(frozen=True)
class Event:
    channel: str
    payload: dict


@dataclass(frozen=True)
class OptimizeOutcome:
    result: RoutingResult
    events: list[Event]
    solver_results: list[Solution]
    winner_provider: str


def run_optimization(
    agent_id: str,
    sliders: SliderValues | None = None,
    assets: list[str] | None = None,
    *,
    agents: AgentStore | None = None,
    jobs: JobStore | None = None,
    market: MarketDataSource | None = None,
    deadline_s: float | None = None,
) -> OptimizeOutcome:
    """Run one job. Raises KeyError for unknown agents, ValueError for a bad
    basket, SolverFailed when the race produces nothing feasible."""
    agents = agents if agents is not None else get_agent_store()
    jobs = jobs if jobs is not None else get_job_store()
    market = market if market is not None else get_source()

    agent = agents.get(agent_id)
    if agent is None:
        raise KeyError(f"unknown agent {agent_id!r}")

    if sliders is not None:
        agents.update_sliders(agent_id, sliders)
    if assets is not None:
        agents.update_assets(agent_id, validate_basket(assets))
    agent = agents.get(agent_id)

    tickers = validate_basket(agent.assets)
    params = map_sliders(agent.sliders, len(tickers))

    # Σ over the fixed 720h window, μ over the fixed lookback within it.
    returns = market.hourly_returns(tickers, config.SIGMA_WINDOW_HOURS)
    problem = PortfolioProblem(
        mu=expected_return(returns, config.MU_WINDOW_HOURS),
        Sigma=covariance(returns),
        gamma=params.gamma,
        w_max=params.w_max,
        w_min=params.w_min,
        asset_tickers=tickers,
    )

    race_result = race(problem, deadline_s=deadline_s)
    winner = race_result.winner

    # Liquidate everything at spot, reallocate the full value by the winner's
    # weights. The spot snapshot covers old and new holdings alike.
    spot = market.spot_prices(list({*tickers, *agent.holdings_units}))
    is_first = not agent.holdings_units
    portfolio_value = (
        agent.bankroll
        if is_first
        else sum(units * spot[t] for t, units in agent.holdings_units.items())
    )
    holdings_units = {
        t: w * portfolio_value / spot[t]
        for t, w in zip(tickers, winner.weights, strict=True)
        if w > 0.0
    }

    agents.apply_solve(
        agent_id,
        holdings_units=holdings_units,
        total=portfolio_value,
        provider_type=winner.provider_role,
    )
    provenance = ProviderProvenance(
        provider=winner.provider,
        provider_role=winner.provider_role,
        q_hash=race_result.q_hash,
        deadline_s=deadline_s if deadline_s is not None else config.RACE_OVERALL_DEADLINE_S,
        solve_time_s=winner.solve_time_s,
        feasible=winner.feasible,
    )
    job = jobs.record(agent_id, provenance)

    result = RoutingResult(
        provider=_PROVIDER_LABELS.get(winner.provider, winner.provider),
        provider_type=winner.provider_role,
        solve_time=winner.solve_time_s,
        vs_classical=race_result.vs_classical,
        portfolio=weights_to_portfolio(winner.weights, tickers, portfolio_value),
        kind="first" if is_first else "retune",
        job_id=job.id,
        solved_at=job.solved_at,
    )

    update = mark_to_market(holdings_units, spot, agent.bankroll)
    events = [Event(channel=f"agent:{agent_id}", payload=update.model_dump(by_alias=True))]
    if is_first:
        events.append(
            Event(
                channel="tv",
                payload={
                    "type": "new-agent",
                    "agentId": agent_id,
                    "name": agent.name,
                    "handle": agent.handle,
                },
            )
        )
    return OptimizeOutcome(
        result=result,
        events=events,
        solver_results=race_result.all_results,
        winner_provider=winner.provider,
    )
