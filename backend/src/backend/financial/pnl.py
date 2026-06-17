"""Mark-to-market P&L.

The MTM loop revalues each agent's held positions against live spot prices and
emits an AgentUpdate. It never trades: token holdings are fixed between retunes,
so weights drift purely as prices move (see CLAUDE.md §5.5). Pure math given the
holdings and a spot snapshot — the market source is decided upstream.
"""

from __future__ import annotations

from ..api.schemas import AgentUpdate


def mark_to_market(
    holdings_units: dict[str, float],
    spot_prices: dict[str, float],
    bankroll_usd: float,
) -> AgentUpdate:
    """Revalue holdings at spot and return the rolled-up AgentUpdate.

    Args:
        holdings_units: token units held per ticker.
        spot_prices: current USD price per ticker (must cover every holding).
        bankroll_usd: the agent's starting bankroll.

    total = Σ units[i] × spot[i];  plUSD = total − bankroll;  plPct = plUSD/bankroll × 100.
    """
    total = sum(units * spot_prices[ticker] for ticker, units in holdings_units.items())
    pl_usd = total - bankroll_usd
    pl_pct = (pl_usd / bankroll_usd * 100.0) if bankroll_usd else 0.0
    return AgentUpdate(pl_usd=pl_usd, pl_pct=pl_pct, total=total)
