"""Internal dataclasses for the financial pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SliderParams:
    """Physical parameters derived from `SliderValues` via slider_map."""

    gamma: float  # risk aversion (objective coefficient)
    w_max: float  # per-asset cap, relative to the basket (1/n → W_MAX_CEILING)
    w_min: float  # participation floor — every basket asset is held at least this
    rebalance_hours: int  # scheduled re-optimization cadence (24h → 1h cap)


@dataclass
class PortfolioProblem:
    """Mean-variance allocation over the player's basket — a box-constrained QP.

    min  (γ/2) wᵀΣw  -  μᵀw
    s.t. Σwᵢ = 1
         w_min ≤ wᵢ ≤ w_max

    Every solve is a fresh allocation: a retune liquidates all holdings at spot
    and reallocates the full value over the (possibly re-selected) basket, so
    there is no turnover anchor and no cardinality constraint — the basket
    decides which assets participate. Gurobi solves the QP natively; SA/D-Wave
    solve the bit-discretized QUBO encoded from it.
    """

    mu: np.ndarray  # shape (N,)
    Sigma: np.ndarray  # shape (N, N), symmetric PSD
    gamma: float
    w_max: float
    asset_tickers: list[str]  # the player's basket (subset of the universe)
    w_min: float = 0.0

    @property
    def N(self) -> int:
        return self.mu.shape[0]

    def objective(self, weights: np.ndarray) -> float:
        """Mean-variance objective value at the given weights."""
        return float(0.5 * self.gamma * weights @ self.Sigma @ weights - self.mu @ weights)

    def __post_init__(self) -> None:
        assert self.Sigma.shape == (self.N, self.N), "Sigma must be (N, N)"
        assert len(self.asset_tickers) == self.N, "asset_tickers length mismatch"
        assert 0 < self.w_max <= 1.0, "w_max must be in (0, 1]"
        assert 0.0 <= self.w_min <= self.w_max, "w_min must be in [0, w_max]"
        # Budget must be reachable: N·w_min ≤ 1 ≤ N·w_max.
        assert self.N * self.w_min <= 1.0 + 1e-9, "N·w_min must be ≤ 1"
        assert self.N * self.w_max >= 1.0 - 1e-9, "N·w_max must be ≥ 1"
