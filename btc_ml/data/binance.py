"""Binance public OHLCV API downloader.

Used for historical 1-minute BTC data. Binance stores years of 1-minute
candlestick data and makes it freely available without authentication.

Note: We use Binance only for TRAINING data. Live inference still uses Kraken.
Prices are nearly identical between exchanges for feature engineering purposes.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
import requests

from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)

_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Binance kline response column order
_KLINE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

# Columns we keep (drop Binance-specific extras)
_KEEP_COLUMNS = ["open", "high", "low", "close", "vwap", "volume", "count"]


class BinanceDownloader:
    """Downloads historical OHLCV data from Binance's public klines endpoint.

    Uses BTCUSDT as the symbol (Bitcoin priced in Tether). Prices track Kraken
    XBTUSD within <0.1% — negligible for feature engineering.

    Args:
        symbol: Trading pair (default 'BTCUSDT').
        request_delay: Seconds to wait between paginated calls.
    """

    def __init__(self, symbol: str = "BTCUSDT", request_delay: float = 0.3) -> None:
        self.symbol = symbol
        self.request_delay = request_delay

    def download_1min(self, days: int = 90, since_ts: int | None = None) -> pd.DataFrame:
        """Download 1-minute OHLCV candles.

        Args:
            days: Number of calendar days of history to ensure.
            since_ts: Optional unix timestamp (seconds) to start from. 
                      If provided, 'days' is ignored and we fetch from this point to now.

        Returns:
            DataFrame with UTC DatetimeIndex and columns:
            open, high, low, close, vwap, volume, count.
        """
        if since_ts:
            start_ms = since_ts * 1000
            logger.info("Incremental download: Fetching from %s", datetime.fromtimestamp(since_ts, tz=timezone.utc))
        else:
            start_ms = int(
                (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
            )
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        logger.info(
            "Downloading Binance 1-min candles: last %d days for %s "
            "(%.0f expected candles, ~%d API calls)",
            days,
            self.symbol,
            days * 1440,
            (days * 1440) / 1000 + 1,
        )

        all_candles: List[list] = []
        current_start_ms = start_ms
        call_count = 0

        while current_start_ms < now_ms:
            batch = self._fetch_klines(
                interval="1m",
                start_ms=current_start_ms,
                limit=1000,
            )

            if not batch:
                logger.debug("Empty batch at start_ms=%d — stopping.", current_start_ms)
                break

            all_candles.extend(batch)
            call_count += 1

            # Next page starts after the last candle's open time
            last_open_ms = int(batch[-1][0])
            next_start_ms = last_open_ms + 60_000  # +1 minute in ms

            if next_start_ms <= current_start_ms:
                break  # No progress guard

            current_start_ms = next_start_ms

            if call_count % 10 == 0:
                elapsed_days = (last_open_ms - start_ms) / (1000 * 86_400)
                logger.info(
                    "  Progress: %.1f / %d days downloaded (%d candles, %d API calls)",
                    elapsed_days,
                    days,
                    len(all_candles),
                    call_count,
                )

            time.sleep(self.request_delay)

        df = self._to_dataframe(all_candles)
        logger.info(
            "Binance 1-min download complete: %d candles (%.1f days) in %d API calls",
            len(df),
            len(df) / 1440,
            call_count,
        )
        return df

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch_klines(
        self,
        interval: str,
        start_ms: int,
        limit: int = 1000,
    ) -> list:
        """Single Binance klines API call.

        Args:
            interval: Candle interval string (e.g. '1m', '1h', '1d').
            start_ms: Start time in milliseconds epoch.
            limit: Max candles to return (max 1000).

        Returns:
            List of raw kline arrays.

        Raises:
            RuntimeError: On API error or network failure.
        """
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "startTime": start_ms,
            "limit": limit,
        }
        try:
            resp = requests.get(_KLINES_URL, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Binance API request failed: {exc}") from exc

    @staticmethod
    def _to_dataframe(candles: list) -> pd.DataFrame:
        """Convert raw Binance kline arrays to a clean DataFrame.

        Binance returns:
          [openTime, open, high, low, close, volume, closeTime,
           quoteVolume, numTrades, takerBuyBase, takerBuyQuote, ignore]

        We map this to the same schema as KrakenDownloader for pipeline
        compatibility: open, high, low, close, vwap, volume, count.

        Args:
            candles: Raw list of kline arrays from Binance.

        Returns:
            DataFrame with UTC DatetimeIndex and OHLCV columns.
        """
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "vwap", "volume", "count"])

        df = pd.DataFrame(candles, columns=_KLINE_COLUMNS)

        # Timestamp from milliseconds to UTC datetime
        df["timestamp"] = pd.to_datetime(
            df["timestamp"].astype(float), unit="ms", utc=True
        )
        df = df.set_index("timestamp").sort_index()

        # Cast numeric columns
        for col in ["open", "high", "low", "close", "volume", "quote_volume", "num_trades"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Map to Kraken-compatible schema:
        # vwap ≈ quote_volume / volume (volume-weighted average price)
        df["vwap"] = (df["quote_volume"] / df["volume"].replace(0, float("nan")))
        df["count"] = df["num_trades"].astype(int)

        # Keep only the columns the pipeline expects
        df = df[["open", "high", "low", "close", "vwap", "volume", "count"]]

        # Drop duplicates (shouldn't happen but safety measure)
        df = df[~df.index.duplicated(keep="last")]

        return df
