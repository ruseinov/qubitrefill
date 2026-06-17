"""Synthetic market source: determinism, full-rank Σ, stable USDC, moving spot."""

from __future__ import annotations

import numpy as np

from backend.financial.estimators.covariance import covariance
from backend.financial.prices.synthetic import SyntheticMarketSource

TICKERS = ["BTC", "ETH", "SOL", "USDC", "IONQ", "QBTS"]


def test_history_is_deterministic():
    a = SyntheticMarketSource(seed=7).hourly_returns(TICKERS, 720)
    b = SyntheticMarketSource(seed=7).hourly_returns(TICKERS, 720)
    assert a.shape == (720, len(TICKERS))
    assert np.array_equal(a, b)


def test_covariance_is_full_rank():
    returns = SyntheticMarketSource(seed=7).hourly_returns(TICKERS, 720)
    sigma = covariance(returns)
    assert np.linalg.matrix_rank(sigma) == len(TICKERS)
    assert np.all(np.diag(sigma) > 0)


def test_usdc_is_pinned_and_low_variance():
    source = SyntheticMarketSource(seed=7, clock=lambda: 0.0)
    spot = source.spot_prices(TICKERS)
    assert spot["USDC"] == 1.0  # zero amplitude → pinned
    returns = source.hourly_returns(TICKERS, 720)
    usdc_var = returns[:, TICKERS.index("USDC")].var()
    btc_var = returns[:, TICKERS.index("BTC")].var()
    assert usdc_var < btc_var


def test_spot_moves_with_the_clock():
    now = {"t": 0.0}
    source = SyntheticMarketSource(seed=7, clock=lambda: now["t"])
    first = source.spot_prices(TICKERS)["BTC"]
    now["t"] = 90.0
    later = source.spot_prices(TICKERS)["BTC"]
    assert first != later
