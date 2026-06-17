"""Select the active market-data source from config (process-wide singleton)."""

from __future__ import annotations

from ... import config
from .base import MarketDataSource
from .synthetic import SyntheticMarketSource

_source: MarketDataSource | None = None


def get_source() -> MarketDataSource:
    global _source
    if _source is None:
        _source = _build()
    return _source


def _build() -> MarketDataSource:
    name = config.MARKET_DATA_SOURCE
    if name == "synthetic":
        return SyntheticMarketSource()
    if name == "assets-api":
        from .assets_api import AssetsApiSource

        return AssetsApiSource()
    raise ValueError(f"unknown MARKET_DATA_SOURCE: {name!r}")


def set_source(source: MarketDataSource | None) -> None:
    """Override the active source (tests); None resets to the config default."""
    global _source
    _source = source
