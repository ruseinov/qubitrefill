"""Decoder: bit-grid round-trips, w_min offset, simplex normalization."""

from __future__ import annotations

import numpy as np
import pytest

from backend.financial.qubo_decoder import decode_bitstring
from backend.solvers.types import DecodeMeta


@pytest.fixture
def meta() -> DecodeMeta:
    return DecodeMeta(
        n_assets=3, bits_per_asset=4, w_max=0.6, w_min=0.1, asset_tickers=["A", "B", "C"]
    )


def _bits_for_levels(levels: list[int], meta: DecodeMeta) -> np.ndarray:
    bits = np.zeros(meta.n_total_bits, dtype=np.int8)
    for i, level in enumerate(levels):
        for k in range(meta.bits_per_asset):
            bits[i * meta.bits_per_asset + k] = (level >> k) & 1
    return bits


def test_zero_bits_decode_to_w_min(meta):
    weights = decode_bitstring(np.zeros(meta.n_total_bits, dtype=np.int8), meta)
    assert np.allclose(weights, meta.w_min)


def test_all_bits_decode_to_w_max(meta):
    weights = decode_bitstring(np.ones(meta.n_total_bits, dtype=np.int8), meta)
    assert np.allclose(weights, meta.w_max)


def test_decode_matches_integer_levels(meta):
    levels = [5, 0, 15]
    weights = decode_bitstring(_bits_for_levels(levels, meta), meta)
    expected = [meta.w_min + level * meta.weight_coef for level in levels]
    assert weights == pytest.approx(expected)


def test_normalize_projects_near_budget_onto_simplex(meta):
    # levels chosen so the raw sum is close to but not exactly 1
    bits = _bits_for_levels([8, 8, 6], meta)
    raw = decode_bitstring(bits, meta)
    assert abs(raw.sum() - 1.0) > 1e-6

    normalized = decode_bitstring(bits, meta, normalize=True)
    assert normalized.sum() == pytest.approx(1.0)


def test_normalize_leaves_bad_sums_alone(meta):
    bits = np.ones(meta.n_total_bits, dtype=np.int8)  # sums to 1.8 — way off
    raw = decode_bitstring(bits, meta, normalize=True)
    assert raw.sum() == pytest.approx(1.8)


def test_wrong_length_bitstring_raises(meta):
    with pytest.raises(ValueError):
        decode_bitstring(np.zeros(meta.n_total_bits - 1, dtype=np.int8), meta)
