"""Agent store — config + holdings + valuation + retune history.

> In-memory only: a process-local dict guarded by a lock. State is lost on
> restart and not shared across workers (run uvicorn with --workers 1). Holdings
> are stored as token **units** (not USD) so mark-to-market is just units × spot.
> Production durability would swap this class for SQLite/Postgres — see TODO.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from uuid import uuid4

from ..api.schemas import AgentConfig, AgentUpdate, SliderValues


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class AgentRecord:
    id: str
    name: str
    handle: str | None
    email: str | None
    reach_out: list[str] | None
    updates_opt_in: bool | None
    sliders: SliderValues
    assets: list[str] | None  # the player's basket (re-selectable on retune)
    bankroll: float
    holdings_units: dict[str, float] = field(default_factory=dict)
    total: float = 0.0  # current mark-to-market value
    pl_usd: float = 0.0
    pl_pct: float = 0.0
    jobs_solved: int = 0
    primary_provider: str = "CPU"  # ProviderType of the latest winning solve
    created_at: str = ""

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            handle=self.handle,
            email=self.email,
            reach_out=self.reach_out,
            updates_opt_in=self.updates_opt_in,
            sliders=self.sliders,
            assets=self.assets,
        )


class AgentStore:
    def __init__(self) -> None:
        self._agents: dict[str, AgentRecord] = {}
        self._lock = RLock()

    def create(self, config: AgentConfig, bankroll: float) -> AgentRecord:
        with self._lock:
            agent_id = uuid4().hex[:8]
            record = AgentRecord(
                id=agent_id,
                name=config.name,
                handle=config.handle,
                email=config.email,
                reach_out=config.reach_out,
                updates_opt_in=config.updates_opt_in,
                sliders=config.sliders,
                assets=list(config.assets) if config.assets else None,
                bankroll=bankroll,
                total=bankroll,
                created_at=_now_iso(),
            )
            self._agents[agent_id] = record
            return record

    def get(self, agent_id: str) -> AgentRecord | None:
        with self._lock:
            return self._agents.get(agent_id)

    def all(self) -> list[AgentRecord]:
        with self._lock:
            return list(self._agents.values())

    def update_sliders(self, agent_id: str, sliders: SliderValues) -> None:
        with self._lock:
            record = self._agents[agent_id]
            record.sliders = sliders

    def update_assets(self, agent_id: str, assets: list[str]) -> None:
        with self._lock:
            record = self._agents[agent_id]
            record.assets = list(assets)

    def apply_solve(
        self,
        agent_id: str,
        holdings_units: dict[str, float],
        total: float,
        provider_type: str,
    ) -> None:
        """Record the outcome of a solve: new holdings, valuation, provider, count."""
        with self._lock:
            record = self._agents[agent_id]
            record.holdings_units = dict(holdings_units)
            record.total = total
            record.pl_usd = total - record.bankroll
            record.pl_pct = (record.pl_usd / record.bankroll * 100.0) if record.bankroll else 0.0
            record.jobs_solved += 1
            record.primary_provider = provider_type

    def set_valuation(self, agent_id: str, update: AgentUpdate) -> None:
        """Update the mark-to-market valuation from the MTM loop."""
        with self._lock:
            record = self._agents.get(agent_id)
            if record is None:
                return
            record.total = update.total
            record.pl_usd = update.pl_usd
            record.pl_pct = update.pl_pct

    def reset(self) -> None:
        with self._lock:
            self._agents.clear()


_store = AgentStore()


def get_agent_store() -> AgentStore:
    """Return the process-wide agent store singleton."""
    return _store
