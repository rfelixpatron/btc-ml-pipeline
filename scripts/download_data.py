"""Step 1: Download BTC OHLCV data from Kraken and sentiment from APIs.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --days 60
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_ml.config import load_config
from btc_ml.data.binance import BinanceDownloader
from btc_ml.data.kraken import KrakenDownloader
from btc_ml.data.sentiment import SentimentDownloader
from btc_ml.utils.io import ensure_dir, load_parquet, save_parquet
from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BTC data from Kraken + sentiment")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Override history_days from config.yaml",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: project root)",
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    days = args.days or cfg.history_days

    logger.info("=== Step 1: Data Download ===")
    logger.info("Pair: %s | History: %d days", cfg.pair, days)

    ensure_dir(cfg.paths.raw_data_dir)

    # ── 1-minute candles (short-term model) ──────────────────────────────────
    logger.info("--- Downloading 1-minute OHLCV (Binance BTCUSDT) ---")
    binance = BinanceDownloader(symbol="BTCUSDT")
    
    try:
        df_1min_existing = load_parquet(cfg.paths.btc_1min_file)
        last_ts = int(df_1min_existing.index.max().timestamp())
        # Only download if existing data is older than 15 mins
        if (datetime.now(timezone.utc).timestamp() - last_ts) > 900:
            df_1min_new = binance.download_1min(since_ts=last_ts)
            df_1min = pd.concat([df_1min_existing, df_1min_new])
            df_1min = df_1min[~df_1min.index.duplicated(keep="last")].sort_index()
        else:
            logger.info("1-min data is already up to date. Skipping download.")
            df_1min = df_1min_existing
    except (FileNotFoundError, Exception):
        logger.info("No existing 1-min data found. Performing full download.")
        df_1min = binance.download_1min(days=days)

    if df_1min.empty:
        logger.error("No 1-min data returned. Check API connectivity.")
        sys.exit(1)

    save_parquet(df_1min, cfg.paths.btc_1min_file)

    # ── Daily candles (long-term model) ──────────────────────────────────────
    logger.info("--- Downloading daily OHLCV (Kraken) ---")
    kraken = KrakenDownloader(pair=cfg.pair)
    
    try:
        df_daily_existing = load_parquet(cfg.paths.btc_daily_file)
        last_ts = int(df_daily_existing.index.max().timestamp())
        # Only download if last candle is from yesterday or older
        if (datetime.now(timezone.utc).timestamp() - last_ts) > 86400:
            df_daily_new = kraken.download_daily(since_ts=last_ts)
            df_daily = pd.concat([df_daily_existing, df_daily_new])
            df_daily = df_daily[~df_daily.index.duplicated(keep="last")].sort_index()
        else:
            logger.info("Daily data is already up to date. Skipping download.")
            df_daily = df_daily_existing
    except (FileNotFoundError, Exception):
        logger.info("No existing daily data found. Performing full download.")
        df_daily = kraken.download_daily(days=days)

    if df_daily.empty:
        logger.error("No daily data returned.")
        sys.exit(1)

    save_parquet(df_daily, cfg.paths.btc_daily_file)

    # ── Sentiment ─────────────────────────────────────────────────────────────
    # Sentiment is small enough that a full download of 90 days is very fast.
    # We keep it simple for now but could also implement incremental here.
    logger.info("--- Downloading sentiment data (Fear & Greed + News) ---")
    sentiment_dl = SentimentDownloader(eodhd_api_key=cfg.eodhd_api_key)
    sentiment = sentiment_dl.download(days=days)

    if sentiment.empty:
        logger.error("No sentiment data retrieved.")
        sys.exit(1)

    save_parquet(sentiment, cfg.paths.sentiment_file)

    logger.info("=== Step 1 Complete ===")
    logger.info(
        "Files verified and updated:\n  %s\n  %s\n  %s",
        cfg.paths.btc_1min_file,
        cfg.paths.btc_daily_file,
        cfg.paths.sentiment_file,
    )


if __name__ == "__main__":
    main()
