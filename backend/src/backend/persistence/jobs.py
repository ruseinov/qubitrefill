"""Job audit log — one record per solved Q hash.

> In-memory only (process-local dict). Persists `ProviderProvenance` for every
> race so outcomes are auditable. Production would back this with an append-only
> table — see TODO.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock
from uuid import uuid4

from ..solvers.types import ProviderProvenance


@dataclass(frozen=True)
class JobRecord:
    id: str
    agent_id: str
    q_hash: str
    provider: str
    provider_role: str
    solve_time_s: float
    deadline_s: float
    feasible: bool
    solved_at: str


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = RLock()

    def record(self, agent_id: str, provenance: ProviderProvenance) -> JobRecord:
        with self._lock:
            job = JobRecord(
                id=uuid4().hex[:12],
                agent_id=agent_id,
                q_hash=provenance.q_hash,
                provider=provenance.provider,
                provider_role=provenance.provider_role,
                solve_time_s=provenance.solve_time_s,
                deadline_s=provenance.deadline_s,
                feasible=provenance.feasible,
                solved_at=datetime.now(UTC).isoformat(),
            )
            self._jobs[job.id] = job
            return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[JobRecord]:
        with self._lock:
            return list(self._jobs.values())

    def reset(self) -> None:
        with self._lock:
            self._jobs.clear()


_store = JobStore()


def get_job_store() -> JobStore:
    """Return the process-wide job store singleton."""
    return _store
