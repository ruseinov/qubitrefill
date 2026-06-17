"""Simulated Annealing provider — solves the QUBO via dwave-neal."""

from __future__ import annotations

import time

from ...financial.types import PortfolioProblem
from ..sampling import select_solution
from ..types import QuboMatrix, Solution, SolverFailed


class SAProvider:
    name = "sa"
    role = "CPU"

    def __init__(self, num_reads: int = 500, num_sweeps: int = 500) -> None:
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps

    def solve_qubo(
        self, qubo: QuboMatrix, problem: PortfolioProblem, deadline_s: float
    ) -> Solution:
        try:
            import neal
        except ImportError as e:
            raise SolverFailed("dwave-neal not installed") from e

        sampler = neal.SimulatedAnnealingSampler()
        t0 = time.perf_counter()
        response = sampler.sample_qubo(
            qubo.to_dict(), num_reads=self.num_reads, num_sweeps=self.num_sweeps
        )
        elapsed = time.perf_counter() - t0

        weights, bits = select_solution(response, qubo, problem)

        return Solution(
            weights=weights,
            objective=problem.objective(weights),
            solve_time_s=elapsed,
            provider="sa",
            provider_role="CPU",
            feasible=False,  # set by the router's feasibility gate
            raw_bitstring=bits,
        )

    def solve_qp(self, problem: PortfolioProblem, deadline_s: float) -> Solution:
        raise NotImplementedError("SA solves QUBO; use solve_qubo with the encoded matrix")
