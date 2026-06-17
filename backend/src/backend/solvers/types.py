"""Shared solver dataclasses — what providers consume and produce."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ProviderName = Literal["dwave", "sa", "gurobi"]
ProviderRole = Literal["QPU", "CPU"]


@dataclass(frozen=True)
class DecodeMeta:
    """Information needed to decode a QUBO bitstring back to portfolio weights.

    Bit layout (i-major): position p holds weight bit x_{i,k} with
    i = p // b, k = p % b. Every asset carries w_min plus a bit-encoded
    increment spanning [w_min, w_max] — there is no indicator block (the
    cardinality constraint was dropped; the basket decides participation).
    """

    n_assets: int
    bits_per_asset: int
    w_max: float
    asset_tickers: list[str]
    w_min: float = 0.0

    @property
    def n_total_bits(self) -> int:
        return self.n_assets * self.bits_per_asset

    @property
    def weight_coef(self) -> float:
        """Per-bit increment above w_min (bits span [w_min, w_max])."""
        return (self.w_max - self.w_min) / (2**self.bits_per_asset - 1)


@dataclass
class QuboMatrix:
    """Symmetric QUBO: minimize xᵀQx subject to x ∈ {0,1}ⁿ."""

    Q: np.ndarray  # symmetric, shape (n, n)
    decode_meta: DecodeMeta

    @property
    def n(self) -> int:
        return self.Q.shape[0]

    def to_dict(self) -> dict[tuple[int, int], float]:
        """Upper-triangle dict form for Ocean samplers (combines symmetric halves)."""
        qdict: dict[tuple[int, int], float] = {}
        for i in range(self.n):
            qdict[(i, i)] = float(self.Q[i, i])
            for j in range(i + 1, self.n):
                v = float(self.Q[i, j] + self.Q[j, i])
                if v != 0.0:
                    qdict[(i, j)] = v
        return qdict


@dataclass
class Solution:
    """A solver's output."""

    weights: np.ndarray  # shape (N,), should sum to ~1 if feasible
    objective: float  # the mean-variance objective value at these weights
    solve_time_s: float  # wall-clock time spent in solver
    provider: ProviderName
    provider_role: ProviderRole
    feasible: bool  # set by feasibility checker, not the provider itself
    raw_bitstring: np.ndarray | None = None  # for QUBO solvers; None for Gurobi QP


@dataclass
class ProviderProvenance:
    """Audit metadata preserved per job (currently lost at the API boundary,
    but persisted in jobs.py for the audit log).
    """

    provider: ProviderName
    provider_role: ProviderRole
    q_hash: str  # SHA256 of QUBO matrix (for race-result audit)
    deadline_s: float
    solve_time_s: float
    feasible: bool


class SolverFailed(Exception):
    """Raised when a solver returns no usable result before its deadline."""

    pass
