"""Rolling walk-forward evaluator for both short-term and long-term models."""

from __future__ import annotations

from typing import Type

import numpy as np
import pandas as pd

from btc_ml.config import Config
from btc_ml.evaluation.metrics import (
    aggregate_fold_results,
    compute_expected_value,
    compute_metrics,
    get_confusion_matrix,
)
from btc_ml.models.base import BaseBTCClassifier
from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)


class RollingEvaluator:
    """Walk-forward window-based evaluation engine.

    Supports:
    - Short-term: 1-hour test windows within the last 24h.
    - Long-term: 30-day training windows followed by 1-day test.

    Args:
        model_class: Concrete subclass of BaseBTCClassifier.
        config: Full pipeline configuration.
    """

    def __init__(
        self,
        model_class: Type[BaseBTCClassifier],
        config: Config,
    ) -> None:
        self.model_class = model_class
        self.config = config

    def evaluate_short_term_windows(
        self,
        features: pd.DataFrame,
        label_up: pd.Series,
        label_down: pd.Series,
        close_prices: pd.Series,
        n_windows: int = 10,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, list[pd.DataFrame]]:
        """Evaluate in 1-hour windows within the last 24 hours.

        Each fold trains on all data prior to the 1-hour window and
        evaluates all 60 minutes within that window.
        """
        last_ts = features.index.max()
        start_ts = last_ts - pd.Timedelta(hours=24)
        
        # Features restricted to last 24h for window selection
        pool = features[features.index >= start_ts]
        horizon = self.config.short_term.horizon_candles
        # Select N start times directly from available indices to avoid data gaps
        pool_indices = pool.index
        if len(pool_indices) < 60:
            window_starts = [pool_indices[0]] if len(pool_indices) > 0 else []
        else:
            # We need at least 60 mins of data after the start point
            max_idx = len(pool_indices) - 60 - horizon
            if max_idx <= 0:
                window_starts = [pool_indices[0]]
            else:
                sel_indices = np.linspace(0, max_idx, n_windows, dtype=int)
                window_starts = pool_indices[sel_indices]

        results_up, results_down = [], []
        detailed_folds = []
        cm_up_total = np.zeros((2, 2), dtype=int)
        cm_down_total = np.zeros((2, 2), dtype=int)

        for i, start in enumerate(window_starts):
            end = start + pd.Timedelta(hours=1)
            
            X_train = features[features.index < start]
            y_up_train = label_up[label_up.index < start]
            y_down_train = label_down[label_down.index < start]

            X_test = features[(features.index >= start) & (features.index < end)]
            y_up_test = label_up[(label_up.index >= start) & (label_up.index < end)]
            y_down_test = label_down[(label_down.index >= start) & (label_down.index < end)]

            if len(X_test) == 0:
                continue

            model = self.model_class(self.config.model)
            model.fit(X_train, y_up_train, y_down_train)

            prob_up = model.predict_proba_up(X_test)
            prob_down = model.predict_proba_down(X_test)

            # Metrics
            m_up = compute_metrics(y_up_test.values, prob_up)
            m_down = compute_metrics(y_down_test.values, prob_down)
            
            # Expected Value
            m_up["expected_value"] = compute_expected_value(
                m_up["precision"], self.config.short_term.up_threshold_pct, 
                self.config.fees.round_trip_pct, self.config.fees.round_trip_pct
            )
            m_down["expected_value"] = compute_expected_value(
                m_down["precision"], self.config.short_term.down_threshold_pct, 
                self.config.fees.round_trip_pct, self.config.fees.round_trip_pct
            )

            m_up["fold"], m_down["fold"] = i + 1, i + 1
            results_up.append(m_up)
            results_down.append(m_down)

            cm_up_total += get_confusion_matrix(y_up_test.values, prob_up)
            cm_down_total += get_confusion_matrix(y_down_test.values, prob_down)

            # Detailed results for this fold
            df_fold = pd.DataFrame({
                "timestamp": X_test.index,
                "price_now": close_prices.reindex(X_test.index),
                "price_future": close_prices.shift(-horizon).reindex(X_test.index),
                "label_up": y_up_test.values,
                "label_down": y_down_test.values,
                "prob_up": prob_up,
                "prob_down": prob_down,
            })
            df_fold["return_pct"] = (df_fold["price_future"] / df_fold["price_now"] - 1) * 100
            df_fold["pred_up"] = (df_fold["prob_up"] >= 0.5).astype(int)
            df_fold["pred_down"] = (df_fold["prob_down"] >= 0.5).astype(int)
            detailed_folds.append(df_fold)

            logger.info("Fold %d: %s to %s | %d samples", i+1, start, end, len(X_test))

        per_fold_up = pd.DataFrame(results_up).set_index("fold")
        per_fold_down = pd.DataFrame(results_down).set_index("fold")
        per_fold_up.attrs["confusion_matrix"] = cm_up_total
        per_fold_down.attrs["confusion_matrix"] = cm_down_total

        return per_fold_up, per_fold_down, aggregate_fold_results(results_up), aggregate_fold_results(results_down), detailed_folds

    def evaluate_long_term_windows(
        self,
        features: pd.DataFrame,
        label_up: pd.Series,
        label_down: pd.Series,
        close_prices: pd.Series,
        n_windows: int = 15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, list[pd.DataFrame]]:
        """Evaluate with sliding 30-day training windows and 1-day test."""
        # Find valid start dates that have 30 days of history + 1 day for test
        available_dates = features.index.normalize().unique()
        
        # We need at least 32 days (30 train, 1 test, 1 for buffer)
        if len(available_dates) < 32:
            n_windows = 1
            start_indices = [0]
        else:
            # Spread start dates across the available history
            # Leave room at the end for 30d train + 1d test
            max_start_idx = len(available_dates) - 32
            start_indices = np.linspace(0, max_start_idx, n_windows, dtype=int)

        results_up, results_down = [], []
        detailed_folds = []
        cm_up_total = np.zeros((2, 2), dtype=int)
        cm_down_total = np.zeros((2, 2), dtype=int)

        horizon = self.config.long_term.horizon_days

        for i, idx in enumerate(start_indices):
            d_start = available_dates[idx]
            d_test = available_dates[idx + 30]
            d_end = available_dates[idx + 31]

            X_train = features[(features.index >= d_start) & (features.index < d_test)]
            y_up_train = label_up[(label_up.index >= d_start) & (label_up.index < d_test)]
            y_down_train = label_down[(label_down.index >= d_start) & (label_down.index < d_test)]

            X_test = features[(features.index >= d_test) & (features.index < d_end)]
            y_up_test = label_up[(label_up.index >= d_test) & (label_up.index < d_end)]
            y_down_test = label_down[(label_down.index >= d_test) & (label_down.index < d_end)]

            if len(X_test) == 0:
                continue

            model = self.model_class(self.config.model)
            model.fit(X_train, y_up_train, y_down_train)

            prob_up = model.predict_proba_up(X_test)
            prob_down = model.predict_proba_down(X_test)

            # Metrics (usually just 1 sample here)
            m_up = compute_metrics(y_up_test.values, prob_up)
            m_down = compute_metrics(y_down_test.values, prob_down)
            
            m_up["expected_value"] = compute_expected_value(
                m_up["precision"], self.config.long_term.up_threshold_pct, 
                self.config.fees.round_trip_pct, self.config.fees.round_trip_pct
            )
            m_down["expected_value"] = compute_expected_value(
                m_down["precision"], self.config.long_term.down_threshold_pct, 
                self.config.fees.round_trip_pct, self.config.fees.round_trip_pct
            )

            m_up["fold"], m_down["fold"] = i + 1, i + 1
            results_up.append(m_up)
            results_down.append(m_down)

            cm_up_total += get_confusion_matrix(y_up_test.values, prob_up)
            cm_down_total += get_confusion_matrix(y_down_test.values, prob_down)

            df_fold = pd.DataFrame({
                "timestamp": X_test.index,
                "price_now": close_prices.reindex(X_test.index),
                "price_future": close_prices.shift(-horizon).reindex(X_test.index),
                "label_up": y_up_test.values,
                "label_down": y_down_test.values,
                "prob_up": prob_up,
                "prob_down": prob_down,
            })
            df_fold["return_pct"] = (df_fold["price_future"] / df_fold["price_now"] - 1) * 100
            df_fold["pred_up"] = (df_fold["prob_up"] >= 0.5).astype(int)
            df_fold["pred_down"] = (df_fold["prob_down"] >= 0.5).astype(int)
            detailed_folds.append(df_fold)

            logger.info("Fold %d: Train %s to %s | Test %s", i+1, d_start.date(), d_test.date(), d_test.date())

        per_fold_up = pd.DataFrame(results_up).set_index("fold")
        per_fold_down = pd.DataFrame(results_down).set_index("fold")
        per_fold_up.attrs["confusion_matrix"] = cm_up_total
        per_fold_down.attrs["confusion_matrix"] = cm_down_total

        return per_fold_up, per_fold_down, aggregate_fold_results(results_up), aggregate_fold_results(results_down), detailed_folds

