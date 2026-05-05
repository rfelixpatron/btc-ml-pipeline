"""Evaluation metric computation functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute a comprehensive set of binary classification metrics.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_prob: Predicted probability of the positive class.
        threshold: Decision boundary (default 0.5).

    Returns:
        Dict with keys: accuracy, precision, recall, f1, auc, mcc, n_pos, n_neg.
    """
    y_true = np.asarray(y_true)
    y_pred = (y_prob >= threshold).astype(int)

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    if n_pos == 0 or n_neg == 0:
        logger.warning(
            "Single class in y_true (%d pos, %d neg). Metrics will be degenerate.",
            n_pos,
            n_neg,
        )
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
            "mcc": float("nan"),
            "n_samples": len(y_true),
            "n_pos": n_pos,
            "n_neg": n_neg,
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "n_samples": len(y_true),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def compute_expected_value(
    precision: float,
    avg_gain_pct: float,
    avg_loss_pct: float,
    round_trip_fee_pct: float,
) -> float:
    """Compute the expected value per trade after fees.

    EV = P(win) * avg_gain - P(loss) * avg_loss - fees

    Args:
        precision: Fraction of trades that are correct (0–1).
        avg_gain_pct: Average gain on winning trades (percentage, e.g. 0.9).
        avg_loss_pct: Average loss on losing trades (percentage, e.g. 0.5).
        round_trip_fee_pct: Total fee for one buy + sell cycle (e.g. 0.62).

    Returns:
        Expected value in percentage points per trade (positive = profitable).
    """
    win_rate = precision
    loss_rate = 1.0 - precision
    ev = win_rate * avg_gain_pct - loss_rate * avg_loss_pct - round_trip_fee_pct
    return round(ev, 4)


def aggregate_fold_results(fold_results: list[dict]) -> dict[str, float]:
    """Compute mean and std across rolling fold metrics.

    Args:
        fold_results: List of metric dicts (one per fold).

    Returns:
        Dict with keys like 'precision_mean', 'precision_std', etc.
    """
    df = pd.DataFrame(fold_results)
    numeric = df.select_dtypes(include="number")
    summary = {}
    for col in numeric.columns:
        summary[f"{col}_mean"] = float(numeric[col].mean())
        summary[f"{col}_std"] = float(numeric[col].std())
    return summary


def get_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return 2x2 confusion matrix.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities.
        threshold: Decision boundary.

    Returns:
        2x2 numpy array [[TN, FP], [FN, TP]].
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return confusion_matrix(np.asarray(y_true), y_pred)
