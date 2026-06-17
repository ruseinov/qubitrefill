"""Background schedulers — the MTM loop.

`run_mtm_loop` is an async task started by the API lifespan. Every `MTM_TICK_S`
it revalues every agent's holdings against a fresh spot snapshot, persists the
new valuation, and publishes an AgentUpdate per agent. It never trades — pure
revaluation until the user retunes (CLAUDE.md §5.5). Publishing happens on the
event loop, so the bus queues stay loop-safe.

"""

from __future__ import annotations

import asyncio
import logging

from .. import config
from ..events.bus import EventBus
from ..financial import basket
from ..financial.pnl import mark_to_market
from ..financial.prices.base import MarketDataSource
from ..financial.prices.source import get_source
from ..persistence.agents import AgentStore, get_agent_store


async def run_mtm_loop(
    bus: EventBus,
    stop: asyncio.Event,
    *,
    agents: AgentStore | None = None,
    market: MarketDataSource | None = None,
    tick_s: float | None = None,
) -> None:
    """Revalue holdings and push AgentUpdates until ``stop`` is set."""
    agents = agents if agents is not None else get_agent_store()
    market = market if market is not None else get_source()
    tick = tick_s if tick_s is not None else config.MTM_TICK_S
    tickers = list(basket.TICKERS)

    log = logging.getLogger(__name__)
    while not stop.is_set():
        try:
            spot = market.spot_prices(tickers)
            for agent in agents.all():
                if not agent.holdings_units:
                    continue
                update = mark_to_market(agent.holdings_units, spot, agent.bankroll)
                agents.set_valuation(agent.id, update)
                bus.publish(f"agent:{agent.id}", update.model_dump(by_alias=True))
        except Exception:
            # A flaky data source must not kill the loop; skip this tick.
            log.exception("MTM tick failed")
        # Sleep one tick, but wake immediately when asked to stop.
        try:
            await asyncio.wait_for(stop.wait(), timeout=tick)
        except TimeoutError:
            pass
