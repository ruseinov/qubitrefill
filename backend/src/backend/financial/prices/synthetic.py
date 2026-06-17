"""Deterministic synthetic market data — the V0 stand-in for live data.

Implements `MarketDataSource` so the rest of the pipeline is unaware it isn't
live. History is a seeded one-factor model (correlated, full-rank covariance);
spot prices oscillate smoothly with wall-clock time so mark-to-market P&L
animates on the booth screens. Swap for the assets-api source by flipping
``config.MARKET_DATA_SOURCE``.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable

import numpy as np

from ... import config

# Stablecoins: pinned ~$1, negligible variance, no market beta.
_STABLES = frozenset({"USDC", "USDT", "DAI"})

# Plausible spot levels for the known basket so the demo reads credibly. Price
# level is cosmetic — only returns drive μ/Σ — so unknown tickers fall back to a
# seeded value.
_BASE_PRICES: dict[str, float] = {
    # crypto
    "BTC": 65_000.0,
    "ETH": 3_500.0,
    "BNB": 600.0,
    "USDC": 1.0,
    "XRP": 0.60,
    "SOL": 150.0,
    "HYPE": 30.0,
    "DOGE": 0.15,
    "USDT": 1.0,
    "ZEC": 40.0,
    "ALGO": 0.25,
    "STRK": 0.50,
    "FIL": 5.0,
    "RENDER": 5.0,
    # stocks
    "IONQ": 35.0,
    "QBTS": 15.0,
    "RGTI": 12.0,
    "QUBT": 8.0,
    "HON": 220.0,
    "SAF": 230.0,
    "SPCX": 150.0,
    "IBM": 290.0,
    "LAES": 4.0,
    "ARQQ": 25.0,
    "GOOGL": 175.0,
    "NVDA": 180.0,
    "MSFT": 450.0,
    "AMZN": 220.0,
}


def _ticker_seed(ticker: str) -> int:
    """Stable 32-bit seed derived from the ticker symbol."""
    return int.from_bytes(hashlib.sha256(ticker.encode()).digest()[:4], "big")


class SyntheticMarketSource:
    """Seeded, network-free `MarketDataSource`."""

    def __init__(self, seed: int | None = None, clock: Callable[[], float] = time.time) -> None:
        self._seed = seed if seed is not None else config.SYNTHETIC_SEED
        self._clock = clock

    def _params(self, ticker: str) -> dict[str, float]:
        """Per-asset model parameters, deterministic in (seed, ticker)."""
        rng = np.random.default_rng(self._seed ^ _ticker_seed(ticker))
        is_stable = ticker in _STABLES
        base = _BASE_PRICES.get(ticker) or float(rng.uniform(5.0, 500.0))
        return {
            "base": base,
            "hourly_vol": 0.0008 if is_stable else float(rng.uniform(0.005, 0.02)),
            "beta": 0.0 if is_stable else float(rng.uniform(0.4, 1.4)),
            "drift": 0.0 if is_stable else float(rng.uniform(-5e-5, 2e-4)),
            "amplitude": 0.0 if is_stable else float(rng.uniform(0.02, 0.08)),
            "period_s": float(rng.uniform(180.0, 1200.0)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }

    def hourly_returns(self, tickers: list[str], window_hours: int) -> np.ndarray:
        """One-factor model: rᵢ = βᵢ·factor + idioᵢ + driftᵢ → (T, N), full rank."""
        factor_rng = np.random.default_rng(self._seed ^ (0x9E3779B1 * window_hours & 0xFFFFFFFF))
        factor = factor_rng.standard_normal(window_hours) * 0.01
        columns = []
        for ticker in tickers:
            p = self._params(ticker)
            idio_rng = np.random.default_rng(
                self._seed ^ _ticker_seed(ticker) ^ (window_hours << 1)
            )
            idiosyncratic = idio_rng.standard_normal(window_hours) * p["hourly_vol"]
            columns.append(p["beta"] * factor + idiosyncratic + p["drift"])
        return np.column_stack(columns)

    def spot_prices(self, tickers: list[str]) -> dict[str, float]:
        """Smoothly oscillating spot so MTM moves between retunes."""
        now = self._clock()
        prices: dict[str, float] = {}
        for ticker in tickers:
            p = self._params(ticker)
            oscillation = p["amplitude"] * math.sin(
                2.0 * math.pi * now / p["period_s"] + p["phase"]
            )
            prices[ticker] = float(p["base"] * (1.0 + oscillation))
        return prices
