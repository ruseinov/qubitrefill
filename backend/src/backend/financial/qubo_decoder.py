"""Decode a QUBO bitstring back to portfolio weights and PortfolioEntry[]."""

from __future__ import annotations

import numpy as np

from .. import config
from ..api.schemas import PortfolioEntry
from ..solvers.types import DecodeMeta


def decode_bitstring(
    bits: np.ndarray,
    meta: DecodeMeta,
    normalize: bool = False,
) -> np.ndarray:
    """Return the weight vector w (shape (N,)) from a QUBO bitstring.

    QUBO weights live on a discrete grid, so Σw=1 is only reachable to within
    ~half a grid step. ``normalize=True`` rescales a near-budget solution onto
    the simplex; sums outside QUBO_NORMALIZE_TOL are left for the feasibility
    gate to reject.
    """
    if bits.shape != (meta.n_total_bits,):
        raise ValueError(f"bitstring length {bits.shape[0]} != expected {meta.n_total_bits}")

    b = meta.bits_per_asset
    place_values = meta.weight_coef * (2 ** np.arange(b))
    weights = meta.w_min + bits.reshape(meta.n_assets, b) @ place_values

    if normalize:
        total = float(weights.sum())
        if total > 0 and abs(total - 1.0) <= config.QUBO_NORMALIZE_TOL:
            weights = weights / total
    return weights


def weights_to_portfolio(
    weights: np.ndarray,
    tickers: list[str],
    bankroll_usd: float,
) -> list[PortfolioEntry]:
    """Convert weights → PortfolioEntry list, sorted descending by pct."""
    entries = [
        PortfolioEntry(ticker=ticker, pct=float(w) * 100.0, usd=float(w) * bankroll_usd)
        for ticker, w in zip(tickers, weights, strict=True)
        if w > 0.0
    ]
    entries.sort(key=lambda e: e.pct, reverse=True)
    return entries
