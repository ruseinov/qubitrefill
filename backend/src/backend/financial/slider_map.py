"""SliderValues (0–100) → SliderParams (γ, w_max, w_min, rebalance cadence).

SINGLE SOURCE OF TRUTH for the slider → physical-parameter mapping.
Frontend never sees physical params; it only sends 0–100 sliders.

The three sliders (mirrors mvp/src/api/types.ts and mvp/src/utils/strategy.ts):
- Risk Preference 100    → aggressive/speculative → low γ (inverted, log-scaled)
- Max Position Size 100  → heavy concentration → high w_max (relative to basket)
- Rebalance Frequency 100→ hourly scheduled re-optimization (hard cap)

The dropped sliders' roles moved elsewhere: diversification → the player's
basket selection; holding style → fixed lookbacks in config. There is no
turnover term — a retune liquidates and reallocates from scratch.
"""

from __future__ import annotations

import math

from .. import config
from ..api.schemas import SliderValues
from .types import SliderParams


def _lerp(t: float, lo: float, hi: float) -> float:
    """Linear interpolation. t in [0, 1]."""
    return lo + (hi - lo) * t


def _log_lerp(t: float, lo: float, hi: float) -> float:
    """Log-scaled interpolation. t in [0, 1]."""
    return math.exp(_lerp(t, math.log(lo), math.log(hi)))


def max_position_cap(basket_size: int, value: float) -> float:
    """Per-asset cap as a fraction, RELATIVE to the basket.

    Mirrors mvp/src/utils/strategy.ts::maxPositionCapPct: an absolute cap below
    1/n is infeasible, and with a big basket a large absolute cap never binds.
    Sweeps from equal weight (1/n) up to W_MAX_CEILING; a 1-asset basket is
    always 100%.
    """
    n = max(1, basket_size)
    floor = 1.0 / n
    ceiling = max(config.W_MAX_CEILING, floor)
    return floor + (value / 100.0) * (ceiling - floor)


def rebalance_every_hours(value: float) -> int:
    """Rebalance slider → scheduled cadence in hours (tiers, hourly hard cap).

    Mirrors mvp/src/utils/strategy.ts::rebalanceEveryHours.
    """
    tiers = config.REBALANCE_TIERS_HOURS
    index = min(len(tiers) - 1, int(value // 20))
    return tiers[index]


def map_sliders(sliders: SliderValues, basket_size: int) -> SliderParams:
    """Map 0–100 slider values to physical optimization parameters.

    `basket_size` is the number of assets the player selected — w_max and
    w_min are defined relative to it.
    """

    # Risk Preference: high slider → aggressive → low γ (inverted, log-scaled)
    risk_t = 1.0 - sliders.risk_preference / 100.0
    gamma = _log_lerp(risk_t, *config.GAMMA_RANGE)

    # Max Position Size: relative cap, equal-weight → W_MAX_CEILING
    w_max = max_position_cap(basket_size, sliders.max_position_size)

    # Minimum position: every selected asset is held (no dust, no cardinality).
    w_min = config.MIN_POSITION_FRACTION / max(1, basket_size)

    # Rebalance Frequency: scheduled re-optimization cadence
    rebalance_hours = rebalance_every_hours(sliders.rebalance_frequency)

    return SliderParams(
        gamma=gamma,
        w_max=w_max,
        w_min=w_min,
        rebalance_hours=rebalance_hours,
    )
