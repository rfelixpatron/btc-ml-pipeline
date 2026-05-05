"""Pure technical indicator functions.

All functions are stateless and operate on pandas Series/DataFrames.
They return Series aligned to the input index, with NaN for warm-up periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Momentum ──────────────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (EWM approximation).

    Args:
        close: Closing price series.
        period: Look-back period (default 14).

    Returns:
        RSI series (0–100), NaN for first `period` rows.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).rename(f"rsi_{period}")


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram.

    Args:
        close: Closing price series.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal EMA period (default 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram) Series.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow).rename("macd_line")
    signal_line = macd_line.ewm(span=signal, adjust=False).mean().rename("macd_signal")
    histogram = (macd_line - signal_line).rename("macd_hist")
    return macd_line, signal_line, histogram


def compute_roc(close: pd.Series, period: int) -> pd.Series:
    """Rate of Change: percentage price change over N periods.

    Args:
        close: Closing price series.
        period: Look-back period.

    Returns:
        ROC series as decimal (0.01 = 1% gain).
    """
    return (close / close.shift(period) - 1.0).rename(f"roc_{period}")


# ── Volatility ────────────────────────────────────────────────────────────────

def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range (Wilder's smoothing).

    Args:
        high: High price series.
        low: Low price series.
        close: Closing price series.
        period: Look-back period (default 14).

    Returns:
        ATR series, NaN for first `period` rows.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period, adjust=False).mean().rename(f"atr_{period}")


def compute_bollinger_pct(
    close: pd.Series, period: int = 20, n_std: float = 2.0
) -> pd.Series:
    """Bollinger Band %B: position of close within the band (0–1, can exceed bounds).

    Args:
        close: Closing price series.
        period: Rolling window for mean/std (default 20).
        n_std: Number of standard deviations for band width (default 2).

    Returns:
        %B series. 0 = lower band, 1 = upper band.
    """
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    band_width = upper - lower
    return ((close - lower) / band_width.replace(0, np.nan)).rename("bb_pct")


def compute_rolling_vol(close: pd.Series, period: int = 15) -> pd.Series:
    """Rolling standard deviation of log returns.

    Args:
        close: Closing price series.
        period: Rolling window (default 15).

    Returns:
        Volatility series.
    """
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(period).std().rename(f"vol_std_{period}")


# ── Trend ─────────────────────────────────────────────────────────────────────

def compute_ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average.

    Args:
        close: Closing price series.
        period: EMA span.

    Returns:
        EMA series.
    """
    return close.ewm(span=period, adjust=False).mean().rename(f"ema_{period}")


def compute_ema_spread(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Normalised spread between two EMAs: (EMA_fast - EMA_slow) / EMA_slow.

    Args:
        close: Closing price series.
        fast: Fast EMA period.
        slow: Slow EMA period.

    Returns:
        Spread series (positive = fast above slow = uptrend).
    """
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    return ((ema_fast - ema_slow) / ema_slow.replace(0, np.nan)).rename(
        f"ema{fast}_ema{slow}_spread"
    )


# ── Volume ────────────────────────────────────────────────────────────────────

def compute_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Relative volume: current volume vs. rolling mean.

    Args:
        volume: Volume series.
        period: Rolling window for mean (default 20).

    Returns:
        Ratio series (1.0 = average, 2.0 = twice average).
    """
    rolling_mean = volume.rolling(period).mean()
    return (volume / rolling_mean.replace(0, np.nan)).rename(f"vol_ratio_{period}")


# ── Candle anatomy ────────────────────────────────────────────────────────────

def compute_candle_features(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """Compute candle body, upper shadow, and lower shadow ratios.

    All values normalised by the candle range (high - low).
    A zero-range candle (doji) yields NaN.

    Args:
        open_: Open price series.
        high: High price series.
        low: Low price series.
        close: Closing price series.

    Returns:
        DataFrame with columns: candle_body, upper_shadow, lower_shadow.
    """
    candle_range = (high - low).replace(0, np.nan)
    body = (close - open_) / candle_range
    upper_shadow = (high - pd.concat([open_, close], axis=1).max(axis=1)) / candle_range
    lower_shadow = (pd.concat([open_, close], axis=1).min(axis=1) - low) / candle_range

    return pd.DataFrame(
        {
            "candle_body": body,
            "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
        }
    )


# ── ZigZag ────────────────────────────────────────────────────────────────────

def compute_zigzag_distance(close: pd.Series, threshold: float = 0.008) -> pd.Series:
    """Percentage distance from current price to the last ZigZag swing extreme.

    Positive = price is above the last extreme (we're in an upswing).
    Negative = price is below the last extreme (we're in a downswing).

    Args:
        close: Closing price series.
        threshold: Minimum percentage reversal to define a new extreme (default 0.8%).

    Returns:
        Distance series as decimal.
    """
    prices = close.values
    n = len(prices)
    distances = np.full(n, np.nan)

    direction: str | None = None
    extreme_price: float = prices[0]

    for i in range(n):
        p = prices[i]
        if np.isnan(p):
            continue

        if direction is None:
            chg = (p - extreme_price) / extreme_price
            if chg >= threshold:
                direction = "up"
                extreme_price = p
            elif chg <= -threshold:
                direction = "down"
                extreme_price = p
        elif direction == "up":
            if p > extreme_price:
                extreme_price = p
            elif (extreme_price - p) / extreme_price >= threshold:
                direction = "down"
                extreme_price = p
        else:  # down
            if p < extreme_price:
                extreme_price = p
            elif (p - extreme_price) / extreme_price >= threshold:
                direction = "up"
                extreme_price = p

        distances[i] = (p - extreme_price) / extreme_price

    return pd.Series(distances, index=close.index, name="zigzag_dist")


# ── Time features ─────────────────────────────────────────────────────────────

def compute_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Extract cyclical time features from a DatetimeIndex.

    Args:
        index: DatetimeIndex (should be timezone-aware UTC).

    Returns:
        DataFrame with columns: hour_of_day, day_of_week.
    """
    return pd.DataFrame(
        {
            "hour_of_day": index.hour,
            "day_of_week": index.dayofweek,
        },
        index=index,
    )
