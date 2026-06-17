"""In-process async pub/sub event bus.

Backs the live push channels: per-agent MTM updates (`agent:{id}`) and TV events
(`tv`). Each subscriber gets its own `asyncio.Queue`; `publish` is non-blocking
and drops to a full queue (a slow consumer just misses a tick — the next MTM
update supersedes it).

V0 is single-process and in-memory. Fan-out across processes (multiple uvicorn
workers) would need Redis pub/sub or Postgres LISTEN/NOTIFY here — see TODO.md.
All publish/subscribe calls happen on the server event loop, so the queues stay
loop-safe without extra locking.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)

    def publish(self, channel: str, payload: object) -> None:
        for queue in tuple(self._subscribers.get(channel, ())):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass  # slow consumer; drop — the next update supersedes this one

    def subscribe(self, channel: str, maxsize: int = 128) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers[channel].add(queue)
        return queue

    def unsubscribe(self, channel: str, queue: asyncio.Queue) -> None:
        subscribers = self._subscribers.get(channel)
        if subscribers:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(channel, None)


_bus = EventBus()


def get_bus() -> EventBus:
    """Return the process-wide event bus singleton."""
    return _bus
