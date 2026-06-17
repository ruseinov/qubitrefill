"""D-Wave Advantage QPU provider — solves the QUBO via the Ocean SDK.

Joins the race only when DWAVE_API_TOKEN is set (explicit opt-in — QPU time
costs real money; see router.build_providers). Uses DWaveCliqueSampler: our
QUBO is dense (the budget penalty couples every pair of bits), and the clique
sampler reuses precomputed clique embeddings instead of re-running a minutes-
long minor-embedding search per solve. The reported solve time is QPU access
time, not wall clock; the race winner is still decided by arrival order.
"""

from __future__ import annotations

import os
import time
from functools import partial
from threading import Lock

from ... import config
from ...financial.types import PortfolioProblem
from ..sampling import select_solution
from ..types import QuboMatrix, Solution, SolverFailed

_sampler = None
_sampler_lock = Lock()


def is_configured() -> bool:
    return bool(os.environ.get("DWAVE_API_TOKEN"))


def sample_kwargs(num_reads: int) -> dict:
    """Shared QPU sampling parameters (also used by the verify-dwave CLI).

    The budget penalty couples every pair of bits, so per-qubit coupling sums
    are large and the sampler's default chain strength under-protects chains —
    uniform torque compensation with a prefactor keeps them intact.
    """
    kwargs: dict = {"num_reads": num_reads, "annealing_time": config.DWAVE_ANNEAL_TIME_US}
    try:
        from dwave.embedding.chain_strength import uniform_torque_compensation

        kwargs["chain_strength"] = partial(
            uniform_torque_compensation, prefactor=config.DWAVE_CHAIN_STRENGTH_PREFACTOR
        )
    except ImportError:
        pass  # fake samplers in tests don't need it
    return kwargs


def _get_sampler():
    """Build the clique sampler once (the Leap handshake is slow)."""
    global _sampler
    with _sampler_lock:
        if _sampler is None:
            try:
                from dwave.system import DWaveCliqueSampler
            except ImportError as e:
                raise SolverFailed("dwave-system not installed") from e
            try:
                _sampler = DWaveCliqueSampler()
            except Exception as e:
                raise SolverFailed(f"D-Wave unavailable: {e}") from e
        return _sampler


class DWaveProvider:
    name = "dwave"
    role = "QPU"

    def __init__(self, sampler=None, num_reads: int | None = None) -> None:
        self._sampler = sampler  # injectable for tests
        self._num_reads = num_reads if num_reads is not None else config.DWAVE_NUM_READS

    def solve_qubo(
        self, qubo: QuboMatrix, problem: PortfolioProblem, deadline_s: float
    ) -> Solution:
        sampler = self._sampler or _get_sampler()
        t0 = time.perf_counter()
        try:
            response = sampler.sample_qubo(
                qubo.to_dict(), label="qtw-tradinggame", **sample_kwargs(self._num_reads)
            )
        except Exception as e:
            raise SolverFailed(f"D-Wave sampling failed: {e}") from e
        wall = time.perf_counter() - t0

        weights, bits = select_solution(response, qubo, problem)
        qpu_access_us = response.info.get("timing", {}).get("qpu_access_time")
        return Solution(
            weights=weights,
            objective=problem.objective(weights),
            solve_time_s=qpu_access_us / 1e6 if qpu_access_us else wall,
            provider="dwave",
            provider_role="QPU",
            feasible=False,  # set by the router's feasibility gate
            raw_bitstring=bits,
        )

    def solve_qp(self, problem: PortfolioProblem, deadline_s: float) -> Solution:
        raise NotImplementedError("D-Wave solves QUBO, not the continuous QP")
