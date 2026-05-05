"""Configuration loader: merges config.yaml with .env API keys."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class ShortTermConfig:
    interval_min: int
    lookback_candles: int
    horizon_candles: int
    up_threshold_pct: float
    down_threshold_pct: float
    eval_folds: int


@dataclass
class LongTermConfig:
    interval_min: int
    lookback_days: int
    horizon_days: int
    up_threshold_pct: float
    down_threshold_pct: float
    eval_folds: int


@dataclass
class ModelConfig:
    type: str
    n_estimators: int
    learning_rate: float
    max_depth: int
    num_leaves: int
    class_weight: str
    random_state: int


@dataclass
class FeesConfig:
    taker_pct: float
    spread_est_pct: float
    round_trip_pct: float


@dataclass
class EvaluationConfig:
    min_precision_threshold: float
    report_output_dir: str


@dataclass
class PathsConfig:
    raw_data_dir: str
    processed_data_dir: str
    btc_1min_file: str
    btc_daily_file: str
    sentiment_file: str
    features_short_file: str
    features_long_file: str


@dataclass
class Config:
    """Top-level configuration object for the BTC ML pipeline."""

    pair: str
    history_days: int
    short_term: ShortTermConfig
    long_term: LongTermConfig
    model: ModelConfig
    zigzag_threshold_pct: float
    fees: FeesConfig
    evaluation: EvaluationConfig
    paths: PathsConfig
    eodhd_api_key: str


def load_config(config_path: str | Path | None = None) -> Config:
    """Load and validate configuration from config.yaml + .env.

    Args:
        config_path: Path to config.yaml. Defaults to project root/config.yaml.

    Returns:
        Fully populated Config dataclass.

    Raises:
        FileNotFoundError: If config.yaml is not found.
        ValueError: If required environment variables are missing.
    """
    if config_path is None:
        config_path = _PROJECT_ROOT / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")

    with config_path.open() as f:
        raw = yaml.safe_load(f)

    eodhd_key = os.getenv("EODHD_API_KEY", "")
    if not eodhd_key:
        import warnings
        warnings.warn(
            "EODHD_API_KEY not set in .env — EODHD news sentiment will be skipped. "
            "Only Fear & Greed Index will be used.",
            UserWarning,
            stacklevel=2,
        )

    st = raw["short_term"]
    lt = raw["long_term"]
    m = raw["model"]
    ev = raw["evaluation"]
    p = raw["paths"]

    return Config(
        pair=raw["data"]["pair"],
        history_days=raw["data"]["history_days"],
        short_term=ShortTermConfig(
            interval_min=st["interval_min"],
            lookback_candles=st["lookback_candles"],
            horizon_candles=st["horizon_candles"],
            up_threshold_pct=st["up_threshold_pct"],
            down_threshold_pct=st["down_threshold_pct"],
            eval_folds=st["eval_folds"],
        ),
        long_term=LongTermConfig(
            interval_min=lt["interval_min"],
            lookback_days=lt["lookback_days"],
            horizon_days=lt["horizon_days"],
            up_threshold_pct=lt["up_threshold_pct"],
            down_threshold_pct=lt["down_threshold_pct"],
            eval_folds=lt["eval_folds"],
        ),
        model=ModelConfig(
            type=m["type"],
            n_estimators=m["n_estimators"],
            learning_rate=m["learning_rate"],
            max_depth=m["max_depth"],
            num_leaves=m["num_leaves"],
            class_weight=m["class_weight"],
            random_state=m["random_state"],
        ),
        zigzag_threshold_pct=raw["zigzag"]["threshold_pct"],
        fees=FeesConfig(
            taker_pct=raw["fees"]["taker_pct"],
            spread_est_pct=raw["fees"]["spread_est_pct"],
            round_trip_pct=raw["fees"]["round_trip_pct"],
        ),
        evaluation=EvaluationConfig(
            min_precision_threshold=ev["min_precision_threshold"],
            report_output_dir=ev["report_output_dir"],
        ),
        paths=PathsConfig(
            raw_data_dir=p["raw_data_dir"],
            processed_data_dir=p["processed_data_dir"],
            btc_1min_file=p["btc_1min_file"],
            btc_daily_file=p["btc_daily_file"],
            sentiment_file=p["sentiment_file"],
            features_short_file=p["features_short_file"],
            features_long_file=p["features_long_file"],
        ),
        eodhd_api_key=eodhd_key,
    )
