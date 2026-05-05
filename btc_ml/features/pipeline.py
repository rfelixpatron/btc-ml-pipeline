"""Feature pipeline: applies all indicators and returns labeled feature matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btc_ml.features.technical import (
    compute_atr,
    compute_bollinger_pct,
    compute_candle_features,
    compute_ema_spread,
    compute_macd,
    compute_roc,
    compute_rolling_vol,
    compute_rsi,
    compute_time_features,
    compute_volume_ratio,
    compute_zigzag_distance,
)
from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    """Transforms raw OHLCV DataFrames into ML-ready feature matrices.

    Handles both short-term (1-min) and long-term (daily) data.
    The pipeline is stateless — fit() is a no-op for sklearn compatibility.

    Args:
        zigzag_threshold: Min % reversal to define a swing extreme (default 0.8%).
    """

    def __init__(self, zigzag_threshold: float = 0.008) -> None:
        self.zigzag_threshold = zigzag_threshold

    def build_short_term_features(
        self,
        df: pd.DataFrame,
        horizon: int = 15,
        up_threshold_pct: float = 0.70,
        down_threshold_pct: float = 0.70,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Build feature matrix and labels for the short-term model.

        Labels:
          - label_up   = 1 if close[T + horizon] >= close[T] * (1 + up_pct)
          - label_down = 1 if close[T + horizon] <= close[T] * (1 - down_pct)
          - Neutral rows (neither condition met) are flagged but kept in X.
            Callers should filter them out for training.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
            horizon: Prediction horizon in candles (default 15 = 15 min).
            up_threshold_pct: Min % gain for UP label.
            down_threshold_pct: Min % drop for DOWN label.

        Returns:
            Tuple of (feature_df, label_up_series, label_down_series).
            All share the same index; rows with NaN features are dropped.
        """
        logger.info(
            "Building short-term features: %d candles, horizon=%d, "
            "up=%.2f%%, down=%.2f%%",
            len(df),
            horizon,
            up_threshold_pct,
            down_threshold_pct,
        )

        features = self._compute_features(df, mode="short")
        label_up, label_down = self._create_labels(
            df["close"], horizon, up_threshold_pct / 100.0, down_threshold_pct / 100.0
        )

        # Align all on feature index (NaN rows already dropped)
        label_up = label_up.reindex(features.index)
        label_down = label_down.reindex(features.index)

        # Drop rows where we can't compute a future label (end of series)
        valid_mask = label_up.notna() & label_down.notna()
        features = features[valid_mask]
        label_up = label_up[valid_mask].astype(int)
        label_down = label_down[valid_mask].astype(int)

        self._log_label_balance(label_up, label_down, up_threshold_pct, down_threshold_pct)
        return features, label_up, label_down

    def build_long_term_features(
        self,
        df: pd.DataFrame,
        sentiment: pd.DataFrame,
        horizon: int = 1,
        up_threshold_pct: float = 0.75,
        down_threshold_pct: float = 0.75,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Build feature matrix and labels for the long-term (daily) model.

        Args:
            df: Daily OHLCV DataFrame with DatetimeIndex.
            sentiment: Daily sentiment DataFrame (fear_greed, eodhd_vader_score).
            horizon: Prediction horizon in days (default 1 = next day).
            up_threshold_pct: Min % gain for UP label.
            down_threshold_pct: Min % drop for DOWN label.

        Returns:
            Tuple of (feature_df, label_up_series, label_down_series).
        """
        logger.info(
            "Building long-term features: %d daily candles, horizon=%d day, "
            "up=%.2f%%, down=%.2f%%",
            len(df),
            horizon,
            up_threshold_pct,
            down_threshold_pct,
        )

        # Compute technical features on daily OHLCV
        features = self._compute_features(df, mode="long")

        # Additional daily-only features
        close_reindexed = df["close"].reindex(features.index)
        high_reindexed = df["high"].reindex(features.index)
        low_reindexed = df["low"].reindex(features.index)

        features["daily_range_pct"] = (high_reindexed - low_reindexed) / close_reindexed
        features["weekly_return"] = close_reindexed.pct_change(7)
        features["monthly_return"] = close_reindexed.pct_change(30)

        # Normalize sentiment index to UTC date-only, then join
        sent = sentiment.copy()
        sent.index = sent.index.normalize()
        features.index = features.index.normalize()
        features = features.join(sent, how="left")

        # Derived sentiment features
        if "fear_greed" in features.columns:
            features["fear_greed_7d_avg"] = features["fear_greed"].rolling(7).mean()
            features["fear_greed_delta"] = features["fear_greed"].diff()

        # Drop non-numeric columns (e.g., sentiment labels) for sklearn compatibility
        features = features.select_dtypes(include=[np.number])

        # Drop rows where all features are NaN (warm-up)
        features = features.dropna(how="all")

        # Create labels aligned to the original df's close (NOT the feature-filtered index)
        # Then reindex to the surviving feature rows
        close_full = df["close"].copy()
        close_full.index = close_full.index.normalize()
        label_up_full, label_down_full = self._create_labels(
            close_full, horizon, up_threshold_pct / 100.0, down_threshold_pct / 100.0
        )

        # Align labels to feature index
        label_up = label_up_full.reindex(features.index)
        label_down = label_down_full.reindex(features.index)

        # Drop rows where labels are NaN (end-of-series and any gaps)
        valid_mask = label_up.notna() & label_down.notna()

        # Also drop rows with too many NaN features
        feature_nan_mask = features.isna().mean(axis=1) < 0.5  # max 50% NaN features
        final_mask = valid_mask & feature_nan_mask

        features = features[final_mask]
        label_up = label_up[final_mask].astype(int)
        label_down = label_down[final_mask].astype(int)

        self._log_label_balance(label_up, label_down, up_threshold_pct, down_threshold_pct)
        return features, label_up, label_down

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_features(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """Compute all technical features from OHLCV data.

        Args:
            df: OHLCV DataFrame.
            mode: 'short' or 'long' (controls which time features are included).

        Returns:
            Feature DataFrame with NaN rows dropped.
        """
        o, h, l, c, v = (
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
        )

        parts = [
            # Momentum
            compute_rsi(c, 14),
            compute_rsi(c, 21),
            compute_macd(c)[2],  # histogram only
            # Trend
            compute_ema_spread(c, fast=9, slow=21),
            compute_ema_spread(c, fast=21, slow=55),
            # Rate of change
            compute_roc(c, 5),
            compute_roc(c, 15),
            compute_roc(c, 30),
            # Volatility
            compute_atr(h, l, c, 14),
            compute_bollinger_pct(c),
            compute_rolling_vol(c, 15),
            # Volume
            compute_volume_ratio(v, 20),
            # Candle anatomy
            compute_candle_features(o, h, l, c),
            # ZigZag
            compute_zigzag_distance(c, self.zigzag_threshold),
        ]

        # Time features (useful for short-term intraday patterns)
        if mode == "short":
            parts.append(compute_time_features(df.index))

        features = pd.concat(parts, axis=1)

        # Drop rows that are entirely NaN (warm-up period)
        before = len(features)
        features = features.dropna()
        dropped = before - len(features)
        if dropped:
            logger.debug("Dropped %d warm-up rows (NaN from indicators)", dropped)

        return features

    @staticmethod
    def _create_labels(
        close: pd.Series,
        horizon: int,
        up_threshold: float,
        down_threshold: float,
    ) -> tuple[pd.Series, pd.Series]:
        """Create binary UP and DOWN labels.

        Args:
            close: Closing price series.
            horizon: Number of periods to look ahead.
            up_threshold: Minimum return (decimal) for UP label.
            down_threshold: Maximum return (decimal, as positive number) for DOWN label.

        Returns:
            Tuple of (label_up, label_down) Series (0 or 1, NaN at tail).
        """
        future_return = close.shift(-horizon) / close - 1.0
        label_up = (future_return >= up_threshold).astype(float)
        label_down = (future_return <= -down_threshold).astype(float)

        # Mark last `horizon` rows as NaN (no future data)
        label_up.iloc[-horizon:] = float("nan")
        label_down.iloc[-horizon:] = float("nan")

        return label_up, label_down

    @staticmethod
    def _log_label_balance(
        label_up: pd.Series,
        label_down: pd.Series,
        up_pct: float,
        down_pct: float,
    ) -> None:
        n = len(label_up)
        n_up = label_up.sum()
        n_down = label_down.sum()
        n_neutral = n - (label_up | label_down).sum()
        logger.info(
            "Label balance (n=%d): UP>=%.2f%%: %d (%.1f%%) | "
            "DOWN>=%.2f%%: %d (%.1f%%) | NEUTRAL: %d (%.1f%%)",
            n,
            up_pct,
            n_up,
            100.0 * n_up / n,
            down_pct,
            n_down,
            100.0 * n_down / n,
            n_neutral,
            100.0 * n_neutral / n,
        )
