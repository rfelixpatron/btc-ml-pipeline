"""Step 3a: Evaluate the short-term (15-minute) classifier with rolling folds.

Usage:
    python scripts/evaluate_short.py
    python scripts/evaluate_short.py --folds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_ml.config import load_config
from btc_ml.evaluation.report import ReportGenerator
from btc_ml.evaluation.rolling import RollingEvaluator
from btc_ml.models.short_term import ShortTermClassifier
from btc_ml.utils.io import load_parquet
from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate short-term BTC classifier")
    parser.add_argument("--folds", type=int, default=None, help="Override eval_folds")
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    n_folds = args.folds or cfg.short_term.eval_folds

    logger.info("=== Step 3a: Short-Term Model Evaluation ===")
    logger.info(
        "Horizon: %d min | UP>=%.2f%% | DOWN>=%.2f%% | Folds: %d",
        cfg.short_term.horizon_candles,
        cfg.short_term.up_threshold_pct,
        cfg.short_term.down_threshold_pct,
        n_folds,
    )

    # Load pre-built feature matrix
    data = load_parquet(cfg.paths.features_short_file)
    label_up = data.pop("label_up")
    label_down = data.pop("label_down")
    features = data

    logger.info("Feature matrix: %d rows × %d cols", *features.shape)

    # Load raw data for price verification
    try:
        raw_df = load_parquet(cfg.paths.btc_1min_file)
        close_prices = raw_df["close"]
    except FileNotFoundError:
        logger.warning("Raw 1-min data not found. Price verification will be limited.")
        close_prices = pd.Series(dtype=float)

    # Window-based evaluation
    evaluator = RollingEvaluator(model_class=ShortTermClassifier, config=cfg)
    per_fold_up, per_fold_down, summary_up, summary_down, detailed_folds = evaluator.evaluate_short_term_windows(
        features=features,
        label_up=label_up,
        label_down=label_down,
        close_prices=close_prices,
        n_windows=n_folds,
    )

    # Reports
    report = ReportGenerator(
        output_dir=cfg.evaluation.report_output_dir,
        min_precision=cfg.evaluation.min_precision_threshold,
    )

    report.print_column_explanations()

    report.print_fold_results(
        per_fold_up, per_fold_down, summary_up, summary_down,
        model_label="Short-Term 15-Minute"
    )

    if detailed_folds:
        report.print_detailed_window_analysis(detailed_folds[0], "Short-Term 15-Minute", fold_idx=1)
        report.plot_short_term_predictions(detailed_folds[0], "short_term_fold_1_signals.png")

    report.save_fold_csv(per_fold_up, per_fold_down, filename_prefix="short_term")

    if "confusion_matrix" in per_fold_up.attrs:
        report.save_confusion_matrix(
            per_fold_up.attrs["confusion_matrix"],
            title="Short-Term UP — Aggregate Confusion Matrix",
            filename="short_term_up_cm.png",
        )
        report.save_confusion_matrix(
            per_fold_down.attrs["confusion_matrix"],
            title="Short-Term DOWN — Aggregate Confusion Matrix",
            filename="short_term_down_cm.png",
        )

    report.save_roc_curve(
        per_fold_up, per_fold_down,
        filename="short_term_auc_per_fold.png",
        title="Short-Term: AUC per Fold (UP vs DOWN)",
    )

    # Train one final model on all data for feature importance
    logger.info("Training full-data model for feature importance analysis ...")
    final_model = ShortTermClassifier(cfg.model)
    final_model.fit(features, label_up, label_down)
    importance = final_model.feature_importances()
    if not importance.empty:
        report.save_feature_importance(
            importance,
            filename="short_term_feature_importance.png",
            title="Short-Term: Feature Importance",
        )
        logger.info("\nTop 10 features (short-term):")
        print(importance.head(10).to_string(index=False))

    # Return summary for master table
    logger.info("=== Short-Term Evaluation Complete ===")
    return {
        "summary_up": summary_up,
        "summary_down": summary_down,
        "per_fold_up": per_fold_up,
        "per_fold_down": per_fold_down,
    }


if __name__ == "__main__":
    main()
