"""Short-term BTC direction classifier (15-minute horizon)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btc_ml.config import ModelConfig
from btc_ml.models.base import BaseBTCClassifier
from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)


class ShortTermClassifier(BaseBTCClassifier):
    """Predicts 15-minute BTC price direction using 1-min OHLCV features.

    Trains two independent binary classifiers:
    - UP:   P(price >= close * (1 + threshold) in 15 min)
    - DOWN: P(price <= close * (1 - threshold) in 15 min)

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
    ) -> "ShortTermClassifier":
        """Train UP and DOWN classifiers.

        Args:
            X: Feature matrix (unscaled). Neutral rows will be filtered per
               direction (rows where neither label is 1 are still trained against
               the negative class, which is standard).
            label_up: Binary label — 1 if price rose >= threshold.
            label_down: Binary label — 1 if price fell >= threshold.

        Returns:
            self (for method chaining).
        """
        self._feature_names = list(X.columns)
        logger.info(
            "Training ShortTermClassifier: %d samples, %d features",
            len(X),
            len(self._feature_names),
        )

        self._scaler_up, self._clf_up = self._fit_direction(X, label_up, "up")
        self._scaler_down, self._clf_down = self._fit_direction(X, label_down, "down")
        return self

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        """P(UP signal) for each row.

        Args:
            X: Feature matrix (unscaled, same columns as training).

        Returns:
            1-D probability array, shape (n_samples,).
        """
        return self._predict_proba_direction(X, self._scaler_up, self._clf_up)

    def predict_proba_down(self, X: pd.DataFrame) -> np.ndarray:
        """P(DOWN signal) for each row.

        Args:
            X: Feature matrix (unscaled, same columns as training).

        Returns:
            1-D probability array, shape (n_samples,).
        """
        return self._predict_proba_direction(X, self._scaler_down, self._clf_down)
