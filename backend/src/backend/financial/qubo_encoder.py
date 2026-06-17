"""Encode the mean-variance QP as a symmetric QUBO via bit discretization.

DERIVATION
----------

QP form (continuous w over the player's n-asset basket):

    min   (γ/2) wᵀΣw  -  μᵀw
    s.t.  Σ w_i = 1                  (budget)
          w_min ≤ w_i ≤ w_max        (box)

There is no cardinality constraint — basket selection already decides which
assets participate — so the encoding needs no indicator variables and no
coupling penalties. Each weight is w_min plus a b-bit increment:

    w_i = w_min + c · Σ_k 2^k · x_{i,k},   x_{i,k} ∈ {0,1},
    c = (w_max - w_min) / (2^b - 1)

In matrix form w = w_min·1 + D x with D ∈ ℝ^(n × nb), D[i, i·b + k] = c·2^k.
The box constraint is implicit in the encoding (all-zero bits → w_min,
all-one bits → w_max).

QUBO objective contributions (z := x, length nb)
------------------------------------------------

Substituting w = m + Dx (m := w_min·1) into the smooth objective:

1. Quadratic:  xᵀ [ (γ/2) DᵀΣD ] x
2. Linear:     Dᵀ [ γ Σ m - μ ] · x
   (constants in m alone are dropped — they don't affect the argmin)

Budget penalty λ_sum (Σw - 1)² with Σw = n·w_min + uᵀx, u = Dᵀ1:
    = λ_sum (uᵀx - r)²,  r := 1 - n·w_min
    → λ_sum u uᵀ on the quadratic, -2 λ_sum r u on the linear.

CONVENTION
----------

Q is SYMMETRIC. For binary x, x_i² = x_i, so the diagonal stores all linear
coefficients. For off-diagonal pairs (i, j), the coefficient of x_i x_j in the
expanded objective is 2 · Q[i, j].
"""

from __future__ import annotations

import hashlib

import numpy as np

from .. import config
from ..solvers.types import DecodeMeta, QuboMatrix
from .types import PortfolioProblem


def bits_for_basket(n_assets: int) -> int:
    """Full precision while the QUBO stays small; 3 bits for large baskets
    (see config — QPU coupler dynamic range, not solver capacity, is the limit)."""
    if n_assets * config.BIT_PRECISION <= config.QUBO_PREFERRED_MAX_VARS:
        return config.BIT_PRECISION
    return config.BIT_PRECISION_LARGE


def encode_qubo(
    problem: PortfolioProblem,
    bits_per_asset: int | None = None,
    penalty_mult_budget: float | None = None,
) -> QuboMatrix:
    """Convert the box-constrained QP → QUBO. See module docstring."""

    b = bits_per_asset if bits_per_asset is not None else bits_for_basket(problem.N)
    pmult_budget = (
        penalty_mult_budget if penalty_mult_budget is not None else config.PENALTY_MULT_BUDGET
    )

    N = problem.N
    w_min = problem.w_min
    c = (problem.w_max - w_min) / (2**b - 1)
    n_bits = N * b

    # Build D such that w = w_min·1 + D x  (shape (N, n_bits))
    D = np.zeros((N, n_bits))
    for i in range(N):
        for k in range(b):
            D[i, i * b + k] = c * (2**k)

    m = np.full(N, w_min)  # the constant offset vector

    A_obj = (problem.gamma / 2.0) * D.T @ problem.Sigma @ D
    b_obj = D.T @ (problem.gamma * (problem.Sigma @ m) - problem.mu)

    # Penalty weight tracks the largest objective coefficient (tiny epsilon
    # guard only — a hard floor would blow the penalty:objective ratio to ~10⁶
    # and erase the objective below QPU precision after coupler auto-scaling).
    obj_scale = max(float(np.abs(A_obj).max()), float(np.abs(b_obj).max()), 1e-12)
    lambda_sum = pmult_budget * obj_scale

    # -------------------------------------------------------------------------
    # Budget penalty: λ_sum (uᵀx - r)²,  u = Dᵀ1,  r = 1 - N·w_min
    # -------------------------------------------------------------------------
    u = D.T @ np.ones(N)
    r = 1.0 - N * w_min
    A_budget = lambda_sum * np.outer(u, u)
    b_budget = -2.0 * lambda_sum * r * u

    # Assemble Q: quadratic blocks, then linear onto the diagonal.
    Q = A_obj + A_budget
    diag = b_obj + b_budget
    for p in range(n_bits):
        Q[p, p] += diag[p]

    assert np.allclose(Q, Q.T), "QUBO matrix must be symmetric"

    decode_meta = DecodeMeta(
        n_assets=N,
        bits_per_asset=b,
        w_max=problem.w_max,
        w_min=w_min,
        asset_tickers=list(problem.asset_tickers),
    )
    return QuboMatrix(Q=Q, decode_meta=decode_meta)


def qubo_hash(qubo: QuboMatrix) -> str:
    """Stable SHA-256 of the QUBO matrix for audit logs."""
    return hashlib.sha256(qubo.Q.tobytes()).hexdigest()
