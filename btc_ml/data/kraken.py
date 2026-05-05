"""Kraken public OHLC API downloader.

Downloads 1-minute and daily BTC/USD candlestick data via pagination.
No authentication required.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
import requests

from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.kraken.com/0/public/OHLC"
_TICKER_URL = "https://api.kraken.com/0/public/Ticker"

# Kraken may return the pair under different key names
_PAIR_KEY_CANDIDATES = ["XXBTZUSD", "XBTUSD"]

# OHLC response column order from Kraken
_OHLC_COLUMNS = ["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"]


class KrakenDownloader:
    """Downloads OHLCV data from Kraken's public REST API.

    Attributes:
        pair: Trading pair identifier (e.g. 'XBTUSD').
        request_delay: Seconds to wait between paginated API calls.
    """

    def __init__(self, pair: str = "XBTUSD", request_delay: float = 0.5) -> None:
        self.pair = pair
        self.request_delay = request_delay

    # ── Public interface ──────────────────────────────────────────────────────

    def download_1min(self, days: int = 90) -> pd.DataFrame:
        """Download 1-minute OHLCV candles for the past N days.

        Kraken returns a maximum of 720 candles per call. This method
        paginates automatically using the `last` field until the full
        requested history is collected or the API returns no more data.

        Args:
            days: Number of calendar days of history to request.

        Returns:
            DataFrame with DatetimeIndex (UTC) and columns:
            open, high, low, close, vwap, volume, count.
        """
        since_ts = int(
            (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
        )
        logger.info(
            "Downloading 1-min candles: last %d days for %s (since %s)",
            days,
            self.pair,
            datetime.fromtimestamp(since_ts, tz=timezone.utc).strftime("%Y-%m-%d"),
        )

        candles = self._paginate(interval=1, since=since_ts)
        df = self._to_dataframe(candles)
        logger.info(
            "1-min download complete: %d candles (%.1f days)",
            len(df),
            len(df) / 1440,
        )
        return df

    def download_daily(self, days: int = 90, since_ts: int | None = None) -> pd.DataFrame:
        """Download daily OHLCV candles."""
        if since_ts:
            logger.info("Incremental download: Daily candles since %s", datetime.fromtimestamp(since_ts, tz=timezone.utc))
        else:
            logger.info("Downloading daily candles for %s (last %d days)", self.pair, days)
            since_ts = int(
                (datetime.now(timezone.utc) - timedelta(days=days + 5)).timestamp()
            )

        candles = self._paginate(interval=1440, since=since_ts)
        df = self._to_dataframe(candles)

        if not since_ts:
            # Trim to requested days only if it's a full download
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            df = df[df.index >= cutoff]

        logger.info("Daily download complete: %d candles", len(df))
        return df

    # ── Private helpers ───────────────────────────────────────────────────────

    def _paginate(self, interval: int, since: int) -> List[list]:
        """Paginate the OHLC endpoint until no new data is returned.

        Args:
            interval: Candle size in minutes.
            since: Unix timestamp to start from.

        Returns:
            List of raw candle arrays from Kraken.
        """
        all_candles: List[list] = []
        current_since = since
        seen_last: set[int] = set()

        while True:
            candles, last = self._fetch_ohlc(interval=interval, since=current_since)

            if not candles:
                logger.debug("No candles returned; stopping pagination.")
                break

            all_candles.extend(candles)
            logger.debug(
                "  Fetched %d candles (total so far: %d), last=%d",
                len(candles),
                len(all_candles),
                last,
            )

            if last in seen_last or last <= current_since:
                break  # No progress — stop

            seen_last.add(last)
            current_since = last
            time.sleep(self.request_delay)

        return all_candles

    def _fetch_ohlc(self, interval: int, since: int) -> tuple[list, int]:
        """Single OHLC API call.

        Args:
            interval: Candle size in minutes.
            since: Unix timestamp start.

        Returns:
            Tuple of (candle_list, last_timestamp).

        Raises:
            RuntimeError: On API error or network failure.
        """
        params = {"pair": self.pair, "interval": interval, "since": since}
        try:
            resp = requests.get(_BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Kraken API request failed: {exc}") from exc

        if data.get("error"):
            raise RuntimeError(f"Kraken API error: {data['error']}")

        result = data["result"]
        last = int(result.get("last", since))

        # Find pair data under possible key names
        candles = None
        for key in _PAIR_KEY_CANDIDATES:
            if key in result:
                candles = result[key]
                break

        if candles is None:
            # Fallback: take first non-'last' key
            for key, val in result.items():
                if key != "last" and isinstance(val, list):
                    candles = val
                    break

        return candles or [], last

    @staticmethod
    def _to_dataframe(candles: List[list]) -> pd.DataFrame:
        """Convert raw Kraken candle arrays to a clean DataFrame.

        Args:
            candles: List of [timestamp, open, high, low, close, vwap, volume, count].

        Returns:
            DataFrame with UTC DatetimeIndex and numeric columns.
        """
        if not candles:
            return pd.DataFrame(columns=_OHLC_COLUMNS[1:])

        df = pd.DataFrame(candles, columns=_OHLC_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="s", utc=True)
        df = df.set_index("timestamp").sort_index()

        numeric_cols = ["open", "high", "low", "close", "vwap", "volume", "count"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # Drop duplicate timestamps (Kraken can return overlapping pages)
        df = df[~df.index.duplicated(keep="last")]

        return df
