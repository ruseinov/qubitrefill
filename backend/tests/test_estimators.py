"""estimators: covariance shape/symmetry and τ-window mean behavior."""

from __future__ import annotations

import numpy as np
import pytest

from backend.financial.estimators.covariance import covariance
from backend.financial.estimators.expected_return import expected_return


def test_covariance_shape_and_symmetry(synthetic_returns):
    n_assets = synthetic_returns.shape[1]
    sigma = covariance(synthetic_returns)
    assert sigma.shape == (n_assets, n_assets)
    assert np.allclose(sigma, sigma.T)
    # Diagonal variances are strictly positive for non-degenerate returns.
    assert np.all(np.diag(sigma) > 0)


def test_covariance_rejects_too_few_observations():
    with pytest.raises(ValueError):
        covariance(np.zeros((1, 4)))


def test_expected_return_uses_only_the_tau_window():
    # 10 hours: first 5 all +0.10, last 5 all -0.02, single asset.
    returns = np.array([[0.10]] * 5 + [[-0.02]] * 5)
    mu_short = expected_return(returns, tau_hours=5)
    mu_full = expected_return(returns, tau_hours=10)
    assert mu_short[0] == pytest.approx(-0.02)
    assert mu_full[0] == pytest.approx(0.04)


def test_expected_return_clamps_tau_to_available_history():
    returns = np.array([[0.01, 0.02]] * 3)
    mu = expected_return(returns, tau_hours=999)
    assert mu == pytest.approx([0.01, 0.02])


def test_expected_return_rejects_nonpositive_tau(synthetic_returns):
    with pytest.raises(ValueError):
        expected_return(synthetic_returns, tau_hours=0)
