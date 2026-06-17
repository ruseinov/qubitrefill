"""End-to-end optimize pipeline on the synthetic market (first solve + retune)."""

from __future__ import annotations

import importlib.util

import pytest

from backend.api.schemas import AgentConfig, SliderValues
from backend.orchestration.job import run_optimization
from backend.persistence.agents import get_agent_store

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gurobipy") is None, reason="gurobipy not installed"
)

BASKET = ["BTC", "ETH", "SOL", "USDC", "IONQ", "QBTS", "RGTI"]


def _config(assets: list[str] | None = BASKET, **sliders: int) -> AgentConfig:
    base = {"rebalanceFrequency": 50, "riskPreference": 50, "maxPositionSize": 50}
    base.update(sliders)
    return AgentConfig(
        name="Quanta",
        handle="q",
        email="q@example.com",
        sliders=SliderValues(**base),
        assets=assets,
    )


def test_first_solve_allocates_bankroll_over_the_basket():
    store = get_agent_store()
    record = store.create(_config(), bankroll=10_000.0)

    outcome = run_optimization(record.id)
    result = outcome.result

    assert result.kind == "first"
    # every basket asset is held (w_min floor), nothing outside the basket
    assert {e.ticker for e in result.portfolio} == set(BASKET)
    assert sum(e.pct for e in result.portfolio) == pytest.approx(100.0, abs=1e-6)
    assert sum(e.usd for e in result.portfolio) == pytest.approx(10_000.0, abs=1e-3)
    assert result.job_id and result.solved_at

    persisted = store.get(record.id)
    assert persisted.jobs_solved == 1
    assert set(persisted.holdings_units) == set(BASKET)

    channels = {e.channel for e in outcome.events}
    assert f"agent:{record.id}" in channels
    assert "tv" in channels  # new-agent splash on first solve


def test_retune_is_value_neutral_and_keeps_the_basket():
    store = get_agent_store()
    record = store.create(_config(), bankroll=10_000.0)
    run_optimization(record.id)

    new_sliders = _config(riskPreference=90, maxPositionSize=80).sliders
    outcome = run_optimization(record.id, sliders=new_sliders)
    result = outcome.result

    assert result.kind == "retune"
    assert {e.ticker for e in result.portfolio} == set(
        BASKET
    )  # basket unchanged unless re-selected
    # fixed-clock market → stationary spot → retune conserves value
    assert sum(e.usd for e in result.portfolio) == pytest.approx(10_000.0, abs=1e-2)
    assert store.get(record.id).jobs_solved == 2
    assert {e.channel for e in outcome.events} == {f"agent:{record.id}"}


def test_retune_can_select_a_new_basket():
    """A retune liquidates everything and reallocates over the new basket."""
    store = get_agent_store()
    record = store.create(_config(), bankroll=10_000.0)
    run_optimization(record.id)

    new_basket = ["HON", "GOOGL", "IBM", "QBTS"]
    result = run_optimization(record.id, assets=new_basket).result

    assert result.kind == "retune"
    assert {e.ticker for e in result.portfolio} == set(new_basket)
    # fixed-clock market → liquidation conserves value
    assert sum(e.usd for e in result.portfolio) == pytest.approx(10_000.0, abs=1e-2)
    assert store.get(record.id).assets == new_basket


def test_no_basket_defaults_to_full_universe():
    from backend.financial.basket import TICKERS

    store = get_agent_store()
    record = store.create(_config(assets=None), bankroll=10_000.0)
    result = run_optimization(record.id).result
    assert {e.ticker for e in result.portfolio} == set(TICKERS)
