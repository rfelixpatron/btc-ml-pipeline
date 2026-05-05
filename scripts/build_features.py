"""Step 2: Build feature matrices for both models from raw OHLCV + sentiment.

Usage:
    python scripts/build_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_ml.config import load_config
from btc_ml.features.pipeline import FeaturePipeline
from btc_ml.utils.io import ensure_dir, load_parquet, save_parquet
from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    logger.info("=== Step 2: Feature Engineering ===")

    ensure_dir(cfg.paths.processed_data_dir)
    pipeline = FeaturePipeline(zigzag_threshold=cfg.zigzag_threshold_pct / 100.0)

    # ── Short-term features (1-min) ───────────────────────────────────────────
    logger.info("--- Building short-term features ---")
    df_1min = load_parquet(cfg.paths.btc_1min_file)

    X_short, y_up_short, y_down_short = pipeline.build_short_term_features(
        df=df_1min,
        horizon=cfg.short_term.horizon_candles,
        up_threshold_pct=cfg.short_term.up_threshold_pct,
        down_threshold_pct=cfg.short_term.down_threshold_pct,
    )

    # Persist: features + both labels together
    short_combined = X_short.copy()
    short_combined["label_up"] = y_up_short
    short_combined["label_down"] = y_down_short
    save_parquet(short_combined, cfg.paths.features_short_file)

    logger.info(
        "Short-term feature matrix: %d rows × %d features",
        len(X_short),
        X_short.shape[1],
    )

    # ── Long-term features (daily) ────────────────────────────────────────────
    logger.info("--- Building long-term features ---")
    df_daily = load_parquet(cfg.paths.btc_daily_file)
    sentiment = load_parquet(cfg.paths.sentiment_file)

    X_long, y_up_long, y_down_long = pipeline.build_long_term_features(
        df=df_daily,
        sentiment=sentiment,
        horizon=cfg.long_term.horizon_days,
        up_threshold_pct=cfg.long_term.up_threshold_pct,
        down_threshold_pct=cfg.long_term.down_threshold_pct,
    )

    long_combined = X_long.copy()
    long_combined["label_up"] = y_up_long
    long_combined["label_down"] = y_down_long
    save_parquet(long_combined, cfg.paths.features_long_file)

    logger.info(
        "Long-term feature matrix: %d rows × %d features",
        len(X_long),
        X_long.shape[1],
    )

    logger.info("=== Step 2 Complete ===")


if __name__ == "__main__":
    main()
