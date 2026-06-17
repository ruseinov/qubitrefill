"""D-Wave provider: decode path, best-of-reads selection, the token gate."""

from __future__ import annotations

import numpy as np
import pytest

from backend.financial.qubo_encoder import encode_qubo
from backend.solvers.providers.dwave import DWaveProvider
from backend.solvers.router import build_providers


def _bits_for_levels(levels: list[int], n_bits: int, b: int = 4) -> np.ndarray:
    bits = np.zeros(n_bits, dtype=np.int8)
    for i, level in enumerate(levels):
        for k in range(b):
            bits[i * b + k] = (level >> k) & 1
    return bits


class FakeResponse:
    """Minimal stand-in for a dimod SampleSet (energy-ascending samples)."""

    def __init__(self, samples: list[dict[int, int]], qpu_access_us: float):
        self._samples = samples
        self.first = type("Best", (), {"sample": samples[0]})()
        self.info = {"timing": {"qpu_access_time": qpu_access_us}}

    def samples(self):
        return self._samples


class FakeSampler:
    def __init__(self, samples: list[dict[int, int]], qpu_access_us: float = 16_000.0):
        self._samples = samples
        self._qpu_access_us = qpu_access_us
        self.last_qubo: dict | None = None

    def sample_qubo(self, qdict, **kwargs):
        self.last_qubo = qdict
        return FakeResponse(self._samples, self._qpu_access_us)


def _as_sample(bits: np.ndarray) -> dict[int, int]:
    return {i: int(b) for i, b in enumerate(bits)}


def test_solve_qubo_decodes_and_reports_qpu_time(synthetic_problem_3assets):
    qubo = encode_qubo(synthetic_problem_3assets)
    # levels [8, 8, 5] sum exactly to 1.0 for w_min=0.1, w_max=0.6, b=4
    bits = _bits_for_levels([8, 8, 5], qubo.n)

    sampler = FakeSampler([_as_sample(bits)])
    solution = DWaveProvider(sampler=sampler).solve_qubo(
        qubo, synthetic_problem_3assets, deadline_s=2.0
    )

    assert solution.provider == "dwave"
    assert solution.provider_role == "QPU"
    assert solution.weights.sum() == pytest.approx(1.0)
    assert solution.solve_time_s == pytest.approx(0.016)  # QPU access, not wall clock
    assert sampler.last_qubo == qubo.to_dict()


def test_best_feasible_read_beats_lowest_energy(synthetic_problem_3assets):
    qubo = encode_qubo(synthetic_problem_3assets)
    # lowest-energy sample badly violates the budget; a later read is feasible
    infeasible = _bits_for_levels([15, 15, 15], qubo.n)  # Σw = 1.8
    feasible = _bits_for_levels([8, 8, 5], qubo.n)  # Σw = 1.0

    sampler = FakeSampler([_as_sample(infeasible), _as_sample(feasible)])
    solution = DWaveProvider(sampler=sampler).solve_qubo(
        qubo, synthetic_problem_3assets, deadline_s=2.0
    )

    assert solution.weights.sum() == pytest.approx(1.0)
    assert np.array_equal(solution.raw_bitstring, feasible)


def test_race_field_requires_a_leap_token(monkeypatch):
    monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
    assert [p.name for p in build_providers()] == ["gurobi", "sa"]

    monkeypatch.setenv("DWAVE_API_TOKEN", "token")
    assert [p.name for p in build_providers()] == ["gurobi", "sa", "dwave"]


def test_production_race_excludes_gurobi(monkeypatch):
    from backend import config

    monkeypatch.setattr(config, "GUROBI_IN_RACE", False)
    monkeypatch.setenv("DWAVE_API_TOKEN", "token")
    assert [p.name for p in build_providers()] == ["sa", "dwave"]
