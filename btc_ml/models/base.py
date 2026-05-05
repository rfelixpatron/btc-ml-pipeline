"""Abstract base class for all BTC direction classifiers."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from btc_ml.config import ModelConfig
from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)


def _build_estimator(cfg: ModelConfig):
    """Factory: instantiate the configured classifier.

    Tries LightGBM first. If libomp is missing on macOS (OSError), falls back
    to sklearn's HistGradientBoostingClassifier (histogram-based, same family,
    no extra system dependencies required).

    Args:
        cfg: ModelConfig with type and hyperparameters.

    Returns:
        Unfitted scikit-learn-compatible classifier.

    Raises:
        ValueError: If model type is unknown.
    """
    if cfg.type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier  # type: ignore

            return LGBMClassifier(
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                num_leaves=cfg.num_leaves,
                random_state=cfg.random_state,
                class_weight=cfg.class_weight,
                n_jobs=-1,
                verbose=-1,
            )
        except OSError:
            logger.warning(
                "LightGBM failed to load (libomp missing on macOS). "
                "Falling back to HistGradientBoostingClassifier. "
                "Install libomp to use LightGBM: brew install libomp"
            )
            return _hist_gradient_boosting(cfg)

    if cfg.type == "xgboost":
        try:
            from xgboost import XGBClassifier  # type: ignore

            return XGBClassifier(
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                random_state=cfg.random_state,
                eval_metric="logloss",
                n_jobs=-1,
            )
        except (OSError, Exception):
            logger.warning(
                "XGBoost failed to load (libomp missing on macOS). "
                "Falling back to HistGradientBoostingClassifier."
            )
            return _hist_gradient_boosting(cfg)

    if cfg.type == "gradient_boosting":
        return _hist_gradient_boosting(cfg)

    raise ValueError(
        f"Unknown model type '{cfg.type}'. "
        "Choose from: lightgbm | xgboost | gradient_boosting"
    )


def _hist_gradient_boosting(cfg: ModelConfig):
    """Return an sklearn HistGradientBoostingClassifier.

    This is sklearn's native LightGBM-equivalent — histogram-based gradient
    boosting with comparable speed and accuracy, zero extra system dependencies.

    Args:
        cfg: ModelConfig with hyperparameters.

    Returns:
        Configured HistGradientBoostingClassifier instance.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    logger.info("Using HistGradientBoostingClassifier (sklearn built-in)")
    return HistGradientBoostingClassifier(
        max_iter=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        class_weight=cfg.class_weight,
    )


class BaseBTCClassifier(ABC):
    """Abstract base for UP and DOWN direction classifiers.

    Each concrete subclass trains two independent binary classifiers:
    one for UP signals and one for DOWN signals.

    Args:
        model_cfg: Hyperparameter configuration.
    """

    def __init__(self, model_cfg: ModelConfig) -> None:
        self.model_cfg = model_cfg
        self._clf_up = None
        self._clf_down = None
        self._scaler_up = StandardScaler()
        self._scaler_down = StandardScaler()
        self._feature_names: list[str] = []

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        label_up: pd.Series,
        label_down: pd.Series,
    ) -> "BaseBTCClassifier":
        """Train both UP and DOWN classifiers."""

    @abstractmethod
    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of an UP move for each row in X."""

    @abstractmethod
    def predict_proba_down(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of a DOWN move for each row in X."""

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _fit_direction(
        self,
        X: pd.DataFrame,
        label: pd.Series,
        direction: str,
    ):
        """Train a single direction classifier.

        Args:
            X: Feature matrix (unscaled).
            label: Binary label series (0/1).
            direction: 'up' or 'down' (used for logging + attribute selection).

        Returns:
            Fitted (scaler, classifier) tuple.
        """
        # Filter to non-neutral rows for this direction
        mask = label.notna()
        X_train = X[mask]
        y_train = label[mask]

        if y_train.nunique() < 2:
            logger.warning(
                "%s classifier: only one class present in training data "
                "(%d rows). Skipping.", direction.upper(), len(y_train)
            )
            return None, None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        clf = _build_estimator(self.model_cfg)
        clf.fit(X_scaled, y_train)

        n_pos = int(y_train.sum())
        logger.debug(
            "%s classifier trained: %d samples (%d positive, %.1f%%)",
            direction.upper(),
            len(y_train),
            n_pos,
            100.0 * n_pos / len(y_train),
        )
        return scaler, clf

    def _predict_proba_direction(
        self,
        X: pd.DataFrame,
        scaler: StandardScaler,
        clf,
    ) -> np.ndarray:
        """Score X with a fitted direction classifier.

        Args:
            X: Feature matrix (unscaled).
            scaler: Fitted StandardScaler.
            clf: Fitted classifier.

        Returns:
            1-D array of probabilities for the positive class.
        """
        if clf is None:
            return np.full(len(X), 0.5)
        X_scaled = scaler.transform(X)
        return clf.predict_proba(X_scaled)[:, 1]

    def feature_importances(self) -> pd.DataFrame:
        """Return average feature importances from UP and DOWN classifiers.

        Returns:
            DataFrame with columns ['feature', 'importance_up', 'importance_down',
            'importance_avg'], sorted descending by importance_avg.
        """
        rows = []
        for name, clf in [("up", self._clf_up), ("down", self._clf_down)]:
            if clf is None:
                continue
            imp = getattr(clf, "feature_importances_", None)
            if imp is None:
                continue
            rows.append(pd.Series(imp, index=self._feature_names, name=f"importance_{name}"))

        if not rows:
            return pd.DataFrame()

        df = pd.concat(rows, axis=1)
        df.index.name = "feature"
        df = df.reset_index()
        df["importance_avg"] = df[[c for c in df.columns if c.startswith("importance")]].mean(axis=1)
        return df.sort_values("importance_avg", ascending=False).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        """Serialize model to disk.

        Args:
            path: Destination file path (.pkl).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        logger.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseBTCClassifier":
        """Deserialize model from disk.

        Args:
            path: Source file path (.pkl).

        Returns:
            Loaded model instance.
        """
        with Path(path).open("rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded ← %s", path)
        return model
