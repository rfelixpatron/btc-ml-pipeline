"""Step 3b: Evaluate the long-term (next-day) classifier with rolling folds.

Usage:
    python scripts/evaluate_long.py
    python scripts/evaluate_long.py --folds 15
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
from btc_ml.models.long_term import LongTermClassifier
from btc_ml.utils.io import load_parquet
from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate long-term BTC classifier")
    parser.add_argument("--folds", type=int, default=None, help="Override eval_folds")
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    n_folds = args.folds or cfg.long_term.eval_folds

    logger.info("=== Step 3b: Long-Term Model Evaluation ===")
    logger.info(
        "Horizon: %d day | UP>=%.2f%% | DOWN>=%.2f%% | Folds: %d",
        cfg.long_term.horizon_days,
        cfg.long_term.up_threshold_pct,
        cfg.long_term.down_threshold_pct,
        n_folds,
    )

    data = load_parquet(cfg.paths.features_long_file)
    label_up = data.pop("label_up")
    label_down = data.pop("label_down")
    features = data

    logger.info("Feature matrix: %d rows × %d cols", *features.shape)

    # Load raw data for price verification
    try:
        raw_df = load_parquet(cfg.paths.btc_daily_file)
        close_prices = raw_df["close"]
    except FileNotFoundError:
        logger.warning("Raw daily data not found. Price verification will be limited.")
        close_prices = pd.Series(dtype=float)

    # Window-based evaluation
    evaluator = RollingEvaluator(model_class=LongTermClassifier, config=cfg)
    per_fold_up, per_fold_down, summary_up, summary_down, detailed_folds = evaluator.evaluate_long_term_windows(
        features=features,
        label_up=label_up,
        label_down=label_down,
        close_prices=close_prices,
        n_windows=n_folds,
    )

    report = ReportGenerator(
        output_dir=cfg.evaluation.report_output_dir,
        min_precision=cfg.evaluation.min_precision_threshold,
    )

    if detailed_folds:
        # For long term, each fold is 1 day. Concatenate to show a summary table and compute aggregate metrics.
        all_detailed = pd.concat(detailed_folds)
        
        # Compute aggregate metrics across all windows
        from btc_ml.evaluation.metrics import compute_metrics
        metrics_up = compute_metrics(all_detailed["label_up"], all_detailed["prob_up"])
        metrics_down = compute_metrics(all_detailed["label_down"], all_detailed["prob_down"])
        
        # Update summary for the master comparison table
        summary_up = {f"{k}_mean": v for k, v in metrics_up.items()}
        summary_down = {f"{k}_mean": v for k, v in metrics_down.items()}

        report.print_column_explanations()
        
        # Print aggregate results instead of per-fold NaNs
        print("\n" + "=" * 70)
        print("  LONG-TERM NEXT-DAY — AGGREGATE RESULTS (Across all windows)")
        print("=" * 70)
        report._print_summary(summary_up, "UP")
        report._print_summary(summary_down, "DOWN")

        report.print_detailed_window_analysis(all_detailed, "Long-Term Next-Day", fold_idx="Combined")

    report.save_fold_csv(per_fold_up, per_fold_down, filename_prefix="long_term")

    if "confusion_matrix" in per_fold_up.attrs:
        report.save_confusion_matrix(
            per_fold_up.attrs["confusion_matrix"],
            title="Long-Term UP — Aggregate Confusion Matrix",
            filename="long_term_up_cm.png",
        )
        report.save_confusion_matrix(
            per_fold_down.attrs["confusion_matrix"],
            title="Long-Term DOWN — Aggregate Confusion Matrix",
            filename="long_term_down_cm.png",
        )

    report.save_roc_curve(
        per_fold_up, per_fold_down,
        filename="long_term_auc_per_fold.png",
        title="Long-Term: AUC per Fold (UP vs DOWN)",
    )

    # Feature importance
    logger.info("Training full-data model for feature importance analysis ...")
    final_model = LongTermClassifier(cfg.model)
    final_model.fit(features, label_up, label_down)
    importance = final_model.feature_importances()
    if not importance.empty:
        report.save_feature_importance(
            importance,
            filename="long_term_feature_importance.png",
            title="Long-Term: Feature Importance (includes Sentiment)",
        )
        logger.info("\nTop 10 features (long-term):")
        print(importance.head(10).to_string(index=False))

    logger.info("=== Long-Term Evaluation Complete ===")
    return {
        "summary_up": summary_up,
        "summary_down": summary_down,
        "per_fold_up": per_fold_up,
        "per_fold_down": per_fold_down,
    }


if __name__ == "__main__":
    main()
