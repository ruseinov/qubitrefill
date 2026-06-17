"""Pick the best decodable solution from a sampler response.

A QUBO sampler's lowest-energy sample can still violate the budget (penalties
compete with the objective, and QPU noise/chain breaks blur small coefficients).
Scanning every read for the best *feasible* decode costs microseconds and
recovers solutions the energy ordering would discard.
"""

from __future__ import annotations

import numpy as np

from ..financial.qubo_decoder import decode_bitstring
from ..financial.types import PortfolioProblem
from .feasibility import check_feasibility
from .types import QuboMatrix


def select_solution(
    response, qubo: QuboMatrix, problem: PortfolioProblem
) -> tuple[np.ndarray, np.ndarray]:
    """Return (weights, bits) of the best feasible sample by true objective,
    falling back to the lowest-energy sample when none is feasible."""
    best_objective = None
    best: tuple[np.ndarray, np.ndarray] | None = None
    fallback: tuple[np.ndarray, np.ndarray] | None = None

    for sample in _samples(response):
        bits = np.array([sample[i] for i in range(qubo.n)], dtype=np.int8)
        weights = decode_bitstring(bits, qubo.decode_meta, normalize=True)
        if fallback is None:
            fallback = (weights, bits)
        if check_feasibility(weights, problem.w_max, problem.w_min).feasible:
            objective = problem.objective(weights)
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best = (weights, bits)

    return best if best is not None else fallback


def _samples(response):
    """Samples in energy-ascending order; tolerates minimal fake samplers."""
    try:
        return response.samples()
    except AttributeError:
        return [response.first.sample]
