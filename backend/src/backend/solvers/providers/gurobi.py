"""Gurobi provider — solves the box-constrained QP natively.

Also serves as the offline oracle for validating QUBO penalty weights.
Lazy import so the rest of the codebase works without a Gurobi install.
"""

from __future__ import annotations

import time

import numpy as np

from ...financial.types import PortfolioProblem
from ..types import QuboMatrix, Solution, SolverFailed


class GurobiProvider:
    name = "gurobi"
    role = "CPU"

    def solve_qp(self, problem: PortfolioProblem, deadline_s: float) -> Solution:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as e:
            raise SolverFailed("gurobipy not installed") from e

        N = problem.N
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("portfolio", env=env)
        model.setParam("TimeLimit", deadline_s)

        w = model.addVars(N, lb=problem.w_min, ub=problem.w_max, name="w")
        model.addConstr(gp.quicksum(w[i] for i in range(N)) == 1.0)

        risk = gp.quicksum(
            0.5 * problem.gamma * problem.Sigma[i, j] * w[i] * w[j]
            for i in range(N)
            for j in range(N)
        )
        ret = gp.quicksum(problem.mu[i] * w[i] for i in range(N))
        model.setObjective(risk - ret, GRB.MINIMIZE)

        t0 = time.perf_counter()
        model.optimize()
        elapsed = time.perf_counter() - t0

        if model.SolCount == 0:
            raise SolverFailed(f"Gurobi returned no solution (status {model.Status})")

        return Solution(
            weights=np.array([w[i].X for i in range(N)]),
            objective=float(model.ObjVal),
            solve_time_s=elapsed,
            provider="gurobi",
            provider_role="CPU",
            feasible=True,  # tentative; the router re-checks
            raw_bitstring=None,
        )

    def solve_qubo(
        self, qubo: QuboMatrix, problem: PortfolioProblem, deadline_s: float
    ) -> Solution:
        return self.solve_qp(problem, deadline_s)
