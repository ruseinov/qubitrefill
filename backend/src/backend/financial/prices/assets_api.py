"""MarketDataSource backed by the assets-api service.

assets-api (gitlab.com/quip.network/assets-api) polls upstream providers
(Alpaca / Massive / CoinGecko) into SQLite and serves hourly OHLCV bars and
spot prices over REST; its response shapes are contractual with this protocol.

Stocks trade ~6.5h on weekdays, so their bars have gaps; series are
forward-filled onto the union hourly grid before computing returns (flat
price → zero return over closed hours).
"""

from __future__ import annotations

import httpx
import numpy as np

from ... import config


class AssetsApiError(RuntimeError):
    pass


class AssetsApiSource:
    def __init__(self, base_url: str | None = None, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            base_url=(base_url or config.ASSETS_API_BASE_URL).rstrip("/"),
            timeout=config.ASSETS_API_TIMEOUT_S,
        )

    def _get(self, path: str, params: dict) -> dict:
        try:
            response = self._client.get(path, params=params)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AssetsApiError(f"assets-api request failed: {path}: {e}") from e
        return response.json()

    def hourly_returns(self, tickers: list[str], window_hours: int) -> np.ndarray:
        # window_hours returns need window_hours + 1 price points; service caps at 90d.
        hours = min(2160, window_hours + 1)
        body = self._get("/v1/history", {"tickers": ",".join(tickers), "hours": hours})
        bars: dict[str, list[dict]] = body.get("bars", {})

        missing = [t for t in tickers if not bars.get(t)]
        if missing:
            raise AssetsApiError(f"no price history for: {missing}")

        timestamps = sorted({bar["t"] for series in bars.values() for bar in series})
        row = {t: i for i, t in enumerate(timestamps)}
        prices = np.full((len(timestamps), len(tickers)), np.nan)
        for col, ticker in enumerate(tickers):
            for bar in bars[ticker]:
                prices[row[bar["t"]], col] = bar["c"]

        prices = _forward_fill(prices)
        returns = prices[1:] / prices[:-1] - 1.0
        return returns[-window_hours:]

    def health(self) -> dict:
        return self._get("/healthz", {})

    def spot_prices(self, tickers: list[str]) -> dict[str, float]:
        body = self._get("/v1/spot", {"tickers": ",".join(tickers)})
        quotes: dict[str, dict] = body.get("prices", {})
        missing = [t for t in tickers if t not in quotes]
        if missing:
            raise AssetsApiError(f"no spot price for: {missing}")
        return {t: float(quotes[t]["price"]) for t in tickers}


def _forward_fill(prices: np.ndarray) -> np.ndarray:
    """Fill NaN gaps with the last known value; backfill leading NaNs."""
    filled = prices.copy()
    for col in range(filled.shape[1]):
        series = filled[:, col]
        mask = np.isnan(series)
        if mask.all():
            continue
        first = int(np.argmin(mask))
        series[:first] = series[first]
        for i in range(first + 1, len(series)):
            if np.isnan(series[i]):
                series[i] = series[i - 1]
    return filled
