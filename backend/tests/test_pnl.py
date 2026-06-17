"""pnl.mark_to_market: holdings × spot → rolled-up AgentUpdate."""

from __future__ import annotations

import pytest

from backend.financial.pnl import mark_to_market


def test_flat_when_value_equals_bankroll():
    # 0.1 BTC @ 50k + 1 ETH @ 5k = 10_000 = bankroll → zero P&L.
    update = mark_to_market({"BTC": 0.1, "ETH": 1.0}, {"BTC": 50_000.0, "ETH": 5_000.0}, 10_000.0)
    assert update.total == pytest.approx(10_000.0)
    assert update.pl_usd == pytest.approx(0.0)
    assert update.pl_pct == pytest.approx(0.0)


def test_gain_is_reported_in_usd_and_pct():
    # BTC up 20% → holding worth 6_000, total 11_000 on a 10_000 bankroll.
    update = mark_to_market({"BTC": 0.1, "ETH": 1.0}, {"BTC": 60_000.0, "ETH": 5_000.0}, 10_000.0)
    assert update.total == pytest.approx(11_000.0)
    assert update.pl_usd == pytest.approx(1_000.0)
    assert update.pl_pct == pytest.approx(10.0)
