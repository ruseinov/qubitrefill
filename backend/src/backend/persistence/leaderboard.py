"""Leaderboard — agents ranked by total = bankroll + MTM P&L.

Derived from the agent store (no separate state). Ties break by agent id for a
stable ordering.
"""

from __future__ import annotations

from ..api.schemas import LeaderboardEntry
from .agents import AgentStore, get_agent_store


def build_leaderboard(agents: AgentStore | None = None) -> list[LeaderboardEntry]:
    store = agents if agents is not None else get_agent_store()
    records = sorted(store.all(), key=lambda r: (-r.total, r.id))
    return [
        LeaderboardEntry(
            rank=index + 1,
            agent_id=record.id,
            name=record.name,
            handle=record.handle,
            total=record.total,
            pl_usd=record.pl_usd,
            pl_pct=record.pl_pct,
            jobs_solved=record.jobs_solved,
            primary_provider=record.primary_provider,
        )
        for index, record in enumerate(records)
    ]
