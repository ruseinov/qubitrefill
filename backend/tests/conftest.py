from __future__ import annotations

import numpy as np
import pytest

from backend.financial.prices.source import set_source
from backend.financial.prices.synthetic import SyntheticMarketSource
from backend.financial.types import PortfolioProblem
from backend.persistence.agents import get_agent_store
from backend.persistence.jobs import get_job_store


@pytest.fixture(autouse=True)
def no_dwave_token(monkeypatch):
    """Keep tests hermetic — never let a configured Leap token reach the QPU."""
    monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def isolated_state():
    """Reset stores and pin a fixed-clock market so pipeline math is reproducible."""
    get_agent_store().reset()
    get_job_store().reset()
    set_source(SyntheticMarketSource(clock=lambda: 1000.0))
    yield
    set_source(None)
    get_agent_store().reset()
    get_job_store().reset()


@pytest.fixture
def synthetic_returns() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((720, 4)) * 0.01


@pytest.fixture
def synthetic_problem_3assets() -> PortfolioProblem:
    mu = np.array([0.02, 0.01, 0.005])
    Sigma = np.array(
        [
            [0.04, 0.01, 0.005],
            [0.01, 0.03, 0.002],
            [0.005, 0.002, 0.01],
        ]
    )
    return PortfolioProblem(
        mu=mu,
        Sigma=Sigma,
        gamma=2.0,
        w_max=0.6,
        w_min=0.1,
        asset_tickers=["A", "B", "C"],
    )
