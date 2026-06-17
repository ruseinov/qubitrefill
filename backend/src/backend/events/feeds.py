"""Channel naming conventions shared by publishers and the WS handlers."""

from __future__ import annotations

TV_CHANNEL = "tv"


def agent_channel(agent_id: str) -> str:
    return f"agent:{agent_id}"
