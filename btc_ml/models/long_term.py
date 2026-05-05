"""Long-term BTC direction classifier (next-day horizon)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btc_ml.config import ModelConfig
from btc_ml.models.base import BaseBTCClassifier
from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)


class LongTermClassifier(BaseBTCClassifier):
    """Predicts next-day BTC price direction using daily OHLCV + sentiment features.

    Trains two independent binary classifiers:
    - UP:   P(next-day price >= close * (1 + threshold))
    - DOWN: P(next-day price <= close * (1 - threshold))

    Args:
        model_cfg: Hyperparameter configuration.
    """

    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__(model_cfg)

    def fit(
        self,
        X: pd.DataFrame,
        label_up: pd.Series,
        label_down: pd.Series,
    ) -> "LongTermClassifier":
        """Train UP and DOWN classifiers on daily feature data.

        Args:
            X: Daily feature matrix (unscaled), includes sentiment columns.
            label_up: Binary label — 1 if next day gained >= threshold.
            label_down: Binary label — 1 if next day dropped >= threshold.

        Returns:
            self (for method chaining).
        """
        self._feature_names = list(X.columns)
        logger.info(
            "Training LongTermClassifier: %d samples, %d features",
            len(X),
            len(self._feature_names),
        )

        self._scaler_up, self._clf_up = self._fit_direction(X, label_up, "up")
        self._scaler_down, self._clf_down = self._fit_direction(X, label_down, "down")
        return self

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        """P(next-day UP) for each row.

        Args:
            X: Daily feature matrix (unscaled).

        Returns:
            1-D probability array.
        """
        return self._predict_proba_direction(X, self._scaler_up, self._clf_up)

    def predict_proba_down(self, X: pd.DataFrame) -> np.ndarray:
        """P(next-day DOWN) for each row.

        Args:
            X: Daily feature matrix (unscaled).

        Returns:
            1-D probability array.
        """
        return self._predict_proba_direction(X, self._scaler_down, self._clf_down)
