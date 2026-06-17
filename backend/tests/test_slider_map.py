"""slider_map: the risk inversion, basket-relative caps, rebalance tiers."""

from __future__ import annotations

import pytest

from backend import config
from backend.api.schemas import SliderValues
from backend.financial.slider_map import map_sliders, max_position_cap, rebalance_every_hours


def _sliders(**overrides: float) -> SliderValues:
    base = {"rebalanceFrequency": 50, "riskPreference": 50, "maxPositionSize": 50}
    base.update(overrides)
    return SliderValues(**base)


def test_risk_preference_is_inverted():
    aggressive = map_sliders(_sliders(riskPreference=100), basket_size=10)
    conservative = map_sliders(_sliders(riskPreference=0), basket_size=10)
    gamma_lo, gamma_hi = config.GAMMA_RANGE
    assert aggressive.gamma == pytest.approx(gamma_lo, rel=1e-6)
    assert conservative.gamma == pytest.approx(gamma_hi, rel=1e-6)


def test_max_position_is_relative_to_basket():
    n = 10
    low = map_sliders(_sliders(maxPositionSize=0), basket_size=n)
    high = map_sliders(_sliders(maxPositionSize=100), basket_size=n)
    assert low.w_max == pytest.approx(1.0 / n)  # equal weight
    assert high.w_max == pytest.approx(config.W_MAX_CEILING)


def test_single_asset_basket_cap_is_100pct():
    assert max_position_cap(1, 0) == pytest.approx(1.0)
    assert max_position_cap(1, 100) == pytest.approx(1.0)


def test_rebalance_tiers_are_discrete_with_hourly_cap():
    assert rebalance_every_hours(0) == 24
    assert rebalance_every_hours(19) == 24
    assert rebalance_every_hours(20) == 8
    assert rebalance_every_hours(50) == 4
    assert rebalance_every_hours(99) == 1
    assert rebalance_every_hours(100) == 1  # hard cap


def test_w_min_keeps_budget_feasible():
    for n in (1, 4, 25):
        params = map_sliders(_sliders(), basket_size=n)
        assert params.w_min == pytest.approx(config.MIN_POSITION_FRACTION / n)
        assert n * params.w_min <= 1.0
        assert n * params.w_max >= 1.0 - 1e-9


def test_min_position_floor_is_quarter_of_equal_weight():
    params = map_sliders(_sliders(), basket_size=8)
    assert params.w_min == pytest.approx(0.25 / 8)


def test_basket_below_minimum_raises():
    from backend.financial.basket import TICKERS, validate_basket

    with pytest.raises(ValueError, match="at least"):
        validate_basket(["BTC", "ETH"])
    assert validate_basket(["BTC", "ETH", "IONQ"]) == ["BTC", "ETH", "IONQ"]
    assert len(validate_basket(None)) == len(TICKERS)
