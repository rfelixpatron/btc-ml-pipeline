"""Step 1: Download BTC OHLCV data from Kraken and sentiment from APIs.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --days 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_ml.config import load_config
from btc_ml.data.binance import BinanceDownloader
from btc_ml.data.kraken import KrakenDownloader
from btc_ml.data.sentiment import SentimentDownloader
from btc_ml.utils.io import ensure_dir, save_parquet
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

    kraken = KrakenDownloader(pair=cfg.pair)

    # ── 1-minute candles (short-term model) ──────────────────────────────────
    # Note: Kraken's public OHLC API only serves the most recent ~720 1-min
    # candles (~12 hours). For 90-day history we use Binance's public klines
    # API (free, no auth, years of 1-min data). Prices track Kraken within
    # <0.1% — negligible for feature engineering.
    logger.info("--- Downloading 1-minute OHLCV (Binance BTCUSDT) ---")
    binance = BinanceDownloader(symbol="BTCUSDT")
    df_1min = binance.download_1min(days=days)

    if df_1min.empty:
        logger.error("No 1-min data returned from Kraken. Check API connectivity.")
        sys.exit(1)

    save_parquet(df_1min, cfg.paths.btc_1min_file)
    logger.info(
        "1-min data: %d candles | %s → %s",
        len(df_1min),
        df_1min.index[0].strftime("%Y-%m-%d %H:%M"),
        df_1min.index[-1].strftime("%Y-%m-%d %H:%M"),
    )

    # ── Daily candles (long-term model) ──────────────────────────────────────
    logger.info("--- Downloading daily OHLCV ---")
    df_daily = kraken.download_daily(days=days)

    if df_daily.empty:
        logger.error("No daily data returned from Kraken.")
        sys.exit(1)

    save_parquet(df_daily, cfg.paths.btc_daily_file)
    logger.info(
        "Daily data: %d candles | %s → %s",
        len(df_daily),
        df_daily.index[0].strftime("%Y-%m-%d"),
        df_daily.index[-1].strftime("%Y-%m-%d"),
    )

    # ── Sentiment ─────────────────────────────────────────────────────────────
    logger.info("--- Downloading sentiment data ---")
    sentiment_dl = SentimentDownloader(eodhd_api_key=cfg.eodhd_api_key)
    sentiment = sentiment_dl.download(days=days)

    if sentiment.empty:
        logger.error("No sentiment data retrieved.")
        sys.exit(1)

    save_parquet(sentiment, cfg.paths.sentiment_file)

    logger.info("=== Step 1 Complete ===")
    logger.info(
        "Files saved:\n  %s\n  %s\n  %s",
        cfg.paths.btc_1min_file,
        cfg.paths.btc_daily_file,
        cfg.paths.sentiment_file,
    )


if __name__ == "__main__":
    main()
