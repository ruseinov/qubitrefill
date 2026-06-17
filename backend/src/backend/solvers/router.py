"""Parallel solver race — first feasible solution wins.

Each provider gets the problem in its native form: Gurobi the continuous QP,
SA/D-Wave the bit-discretized QUBO. The solve work releases the GIL, so a
thread pool gives real parallelism. All results are kept for the audit log
and the vsClassical baseline (decision Q7).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .. import config
from ..financial.qubo_encoder import encode_qubo, qubo_hash
from ..financial.types import PortfolioProblem
from .feasibility import check_feasibility
from .providers import dwave
from .providers.gurobi import GurobiProvider
from .providers.sa import SAProvider
from .types import QuboMatrix, Solution, SolverFailed


@dataclass
class RaceResult:
    winner: Solution
    runner_up_classical: Solution | None
    all_results: list[Solution]
    q_hash: str

    @property
    def vs_classical(self) -> float:
        """How many times faster the winner was than the classical runner-up
        (runner_up_time / winner_time, ≥1 when the winner is faster). The
        frontend renders this as "N× vs classical". 1.0 if no baseline."""
        if self.runner_up_classical is None or self.winner.solve_time_s <= 0.0:
            return 1.0
        return self.runner_up_classical.solve_time_s / self.winner.solve_time_s


def build_providers() -> list:
    """The race field: SA always, Gurobi unless disabled for production,
    the D-Wave QPU when a Leap token is set."""
    providers: list = [GurobiProvider()] if config.GUROBI_IN_RACE else []
    providers.append(SAProvider())
    if dwave.is_configured():
        providers.append(dwave.DWaveProvider())
    return providers


def _dispatch(
    provider: object, problem: PortfolioProblem, qubo: QuboMatrix, deadline_s: float
) -> Solution:
    if provider.name == "gurobi":  # type: ignore[attr-defined]
        return provider.solve_qp(problem, deadline_s)  # type: ignore[attr-defined]
    return provider.solve_qubo(qubo, problem, deadline_s)  # type: ignore[attr-defined]


def race(problem: PortfolioProblem, deadline_s: float | None = None) -> RaceResult:
    """Run the race; raise SolverFailed if nothing feasible arrives in time."""
    overall_deadline = deadline_s if deadline_s is not None else config.RACE_OVERALL_DEADLINE_S
    per_solver_deadline = config.SOLVER_DEADLINE_S

    qubo = encode_qubo(problem)
    q_h = qubo_hash(qubo)
    providers = build_providers()

    results: list[Solution] = []
    winner: Solution | None = None

    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = {
            executor.submit(_dispatch, p, problem, qubo, per_solver_deadline): p.name
            for p in providers
        }
        try:
            for future in as_completed(futures, timeout=overall_deadline):
                try:
                    solution = future.result()
                except SolverFailed:
                    continue
                feas = check_feasibility(solution.weights, problem.w_max, problem.w_min)
                solution.feasible = feas.feasible
                results.append(solution)
                if solution.feasible and winner is None:
                    winner = solution  # keep collecting for the vsClassical baseline
        except TimeoutError:
            pass  # deadline hit; proceed with whatever finished

    if winner is None:
        raise SolverFailed("no feasible solution from any provider before deadline")

    return RaceResult(
        winner=winner,
        runner_up_classical=_pick_runner_up_classical(winner, results),
        all_results=results,
        q_hash=q_h,
    )


def _pick_runner_up_classical(winner: Solution, results: list[Solution]) -> Solution | None:
    """Fastest classical (CPU) solution that isn't the winner, preferring feasible ones."""
    others = [s for s in results if s is not winner and s.provider_role == "CPU"]
    if not others:
        return None
    feasible = [s for s in others if s.feasible]
    return min(feasible or others, key=lambda s: s.solve_time_s)
