"""V0 quality bar: budget + box feasibility.

No cardinality check — the basket decides which assets participate, and the
w_min floor guarantees every selected asset is held.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config


@dataclass(frozen=True)
class FeasibilityResult:
    feasible: bool
    budget_residual: float  # Σw_i − 1, signed
    box_violation: float  # max violation of the [w_min, w_max] bounds
    reason: str = ""


def check_feasibility(
    weights: np.ndarray,
    w_max: float,
    w_min: float = 0.0,
    eps_budget: float | None = None,
    eps_box: float | None = None,
) -> FeasibilityResult:
    eps_b = eps_budget if eps_budget is not None else config.EPS_BUDGET
    eps_x = eps_box if eps_box is not None else config.EPS_BOX

    budget_res = float(weights.sum() - 1.0)
    box_v = max(
        float(max(0.0, (weights - w_max).max())),
        float(max(0.0, (w_min - weights).max())),
    )

    reasons = []
    if abs(budget_res) > eps_b:
        reasons.append(f"budget |Σw-1|={abs(budget_res):.4g} > {eps_b}")
    if box_v > eps_x:
        reasons.append(f"box violation {box_v:.4g} > {eps_x}")

    return FeasibilityResult(
        feasible=not reasons,
        budget_residual=budget_res,
        box_violation=box_v,
        reason="; ".join(reasons) if reasons else "ok",
    )
