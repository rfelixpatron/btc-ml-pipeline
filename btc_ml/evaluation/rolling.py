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
    """Walk-forward rolling evaluation engine.

    Splits data chronologically into N folds. For each fold:
    - Trains the model on all data before the test window
    - Evaluates on the test window
    - Reports per-fold metrics for UP and DOWN classifiers

    This ensures no future data leaks into the training set.

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

    def evaluate(
        self,
        features: pd.DataFrame,
        label_up: pd.Series,
        label_down: pd.Series,
        n_folds: int,
        min_train_size: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
        """Run rolling evaluation.

        Args:
            features: Feature matrix (full dataset, chronological order).
            label_up: Binary UP labels aligned to features.
            label_down: Binary DOWN labels aligned to features.
            n_folds: Number of rolling test windows.
            min_train_size: Minimum number of rows for training. If None,
                defaults to 60% of the total dataset.

        Returns:
            Tuple of:
              - per_fold_up (DataFrame): Per-fold metrics for UP classifier.
              - per_fold_down (DataFrame): Per-fold metrics for DOWN classifier.
              - summary_up (dict): Aggregate mean/std across folds (UP).
              - summary_down (dict): Aggregate mean/std across folds (DOWN).
        """
        n = len(features)
        if min_train_size is None:
            min_train_size = max(int(n * 0.60), 1)

        test_pool_size = n - min_train_size
        if test_pool_size < n_folds:
            logger.warning(
                "Only %d samples available for testing (%d requested folds). "
                "Reducing to %d folds.",
                test_pool_size,
                n_folds,
                test_pool_size,
            )
            n_folds = max(1, test_pool_size)

        fold_size = test_pool_size // n_folds

        logger.info(
            "Rolling evaluation: %d folds | total=%d | train_min=%d | "
            "test_per_fold=%d",
            n_folds,
            n,
            min_train_size,
            fold_size,
        )

        results_up: list[dict] = []
        results_down: list[dict] = []

        # Accumulate confusion matrices across folds
        cm_up_total = np.zeros((2, 2), dtype=int)
        cm_down_total = np.zeros((2, 2), dtype=int)

        for fold_idx in range(n_folds):
            test_start = min_train_size + fold_idx * fold_size
            test_end = test_start + fold_size

            X_train = features.iloc[:test_start]
            y_up_train = label_up.iloc[:test_start]
            y_down_train = label_down.iloc[:test_start]

            X_test = features.iloc[test_start:test_end]
            y_up_test = label_up.iloc[test_start:test_end]
            y_down_test = label_down.iloc[test_start:test_end]

            if len(X_test) == 0:
                break

            # Train model
            model = self.model_class(self.config.model)
            model.fit(X_train, y_up_train, y_down_train)

            # Evaluate UP
            prob_up = model.predict_proba_up(X_test)
            metrics_up = compute_metrics(y_up_test.values, prob_up)
            ev_up = compute_expected_value(
                precision=metrics_up["precision"],
                avg_gain_pct=self.config.short_term.up_threshold_pct
                if hasattr(self.config, "short_term")
                else self.config.long_term.up_threshold_pct,
                avg_loss_pct=self.config.fees.round_trip_pct,
                round_trip_fee_pct=self.config.fees.round_trip_pct,
            )
            metrics_up["expected_value"] = ev_up
            metrics_up["fold"] = fold_idx + 1
            results_up.append(metrics_up)
            cm_up_total += get_confusion_matrix(y_up_test.values, prob_up)

            # Evaluate DOWN
            prob_down = model.predict_proba_down(X_test)
            metrics_down = compute_metrics(y_down_test.values, prob_down)
            ev_down = compute_expected_value(
                precision=metrics_down["precision"],
                avg_gain_pct=self.config.short_term.down_threshold_pct
                if hasattr(self.config, "short_term")
                else self.config.long_term.down_threshold_pct,
                avg_loss_pct=self.config.fees.round_trip_pct,
                round_trip_fee_pct=self.config.fees.round_trip_pct,
            )
            metrics_down["expected_value"] = ev_down
            metrics_down["fold"] = fold_idx + 1
            results_down.append(metrics_down)
            cm_down_total += get_confusion_matrix(y_down_test.values, prob_down)

            logger.info(
                "Fold %2d/%d | UP  prec=%.3f rec=%.3f auc=%.3f ev=%.3f%% | "
                "DOWN prec=%.3f rec=%.3f auc=%.3f ev=%.3f%%",
                fold_idx + 1,
                n_folds,
                metrics_up["precision"],
                metrics_up["recall"],
                metrics_up["auc"],
                ev_up,
                metrics_down["precision"],
                metrics_down["recall"],
                metrics_down["auc"],
                ev_down,
            )

        per_fold_up = pd.DataFrame(results_up).set_index("fold")
        per_fold_down = pd.DataFrame(results_down).set_index("fold")
        summary_up = aggregate_fold_results(results_up)
        summary_down = aggregate_fold_results(results_down)

        # Attach confusion matrices for report generation
        per_fold_up.attrs["confusion_matrix"] = cm_up_total
        per_fold_down.attrs["confusion_matrix"] = cm_down_total

        return per_fold_up, per_fold_down, summary_up, summary_down
