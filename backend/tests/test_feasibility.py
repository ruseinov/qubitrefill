"""feasibility: budget and box checks."""

from __future__ import annotations

import numpy as np

from backend.solvers.feasibility import check_feasibility


def test_feasible_solution_passes():
    result = check_feasibility(np.array([0.5, 0.3, 0.2]), w_max=0.6, w_min=0.1)
    assert result.feasible
    assert result.reason == "ok"


def test_budget_violation_detected():
    result = check_feasibility(np.array([0.5, 0.3, 0.1]), w_max=0.6, w_min=0.1)
    assert not result.feasible
    assert "budget" in result.reason


def test_cap_violation_detected():
    result = check_feasibility(np.array([0.7, 0.2, 0.1]), w_max=0.6, w_min=0.1)
    assert not result.feasible
    assert "box" in result.reason


def test_floor_violation_detected():
    result = check_feasibility(np.array([0.55, 0.42, 0.03]), w_max=0.6, w_min=0.1)
    assert not result.feasible
    assert "box" in result.reason


def test_negative_weight_is_a_box_violation():
    result = check_feasibility(np.array([0.8, 0.4, -0.2]), w_max=0.6, w_min=0.0)
    assert not result.feasible
    assert "box" in result.reason
