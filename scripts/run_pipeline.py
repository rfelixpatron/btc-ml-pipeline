"""Master pipeline: runs all steps sequentially and prints the final comparison table.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-download   # if data already downloaded
    python scripts/run_pipeline.py --skip-features   # if features already built
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.download_data as step1
import scripts.build_features as step2
import scripts.evaluate_short as step3a
import scripts.evaluate_long as step3b

from btc_ml.config import load_config
from btc_ml.evaluation.report import ReportGenerator
from btc_ml.utils.io import load_parquet
from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full BTC ML pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download step")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature build step")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║     BTC ML Pipeline — Full Run           ║")
    logger.info("╚══════════════════════════════════════════╝")

    # Step 1: Download data
    if not args.skip_download:
        step1.main()
    else:
        logger.info("Skipping Step 1 (--skip-download)")

    # Step 2: Feature engineering
    if not args.skip_features:
        step2.main()
    else:
        logger.info("Skipping Step 2 (--skip-features)")

    # Step 3a: Short-term evaluation
    short_results = step3a.main()

    # Step 3b: Long-term evaluation
    long_results = step3b.main()

    # Print sentiment table
    report = ReportGenerator(
        output_dir=cfg.evaluation.report_output_dir,
        min_precision=cfg.evaluation.min_precision_threshold,
    )
    try:
        sentiment = load_parquet(cfg.paths.sentiment_file)
        report.print_sentiment_table(sentiment)
    except FileNotFoundError:
        logger.warning("Sentiment file not found — skipping sentiment table.")

    # Master comparison table
    if short_results and long_results:
        def _row(model, direction, summary):
            key = f"{direction.lower()}"
            return {
                "model": model,
                "direction": direction,
                "precision": summary.get(f"precision_{key}_mean", float("nan")),
                "recall": summary.get(f"recall_{key}_mean", float("nan")),
                "f1": summary.get(f"f1_{key}_mean", float("nan")),
                "auc": summary.get(f"auc_{key}_mean", float("nan")),
                "ev_after_fees": summary.get(f"expected_value_{key}_mean", float("nan")),
            }

        rows = [
            {
                "model": "Short-Term 15-min",
                "direction": "UP",
                **{k: v for k, v in short_results["summary_up"].items()
                   if k.endswith("_mean") and not k.startswith("n_")}
            },
            {
                "model": "Short-Term 15-min",
                "direction": "DOWN",
                **{k: v for k, v in short_results["summary_down"].items()
                   if k.endswith("_mean") and not k.startswith("n_")}
            },
            {
                "model": "Long-Term Next-Day",
                "direction": "UP",
                **{k: v for k, v in long_results["summary_up"].items()
                   if k.endswith("_mean") and not k.startswith("n_")}
            },
            {
                "model": "Long-Term Next-Day",
                "direction": "DOWN",
                **{k: v for k, v in long_results["summary_down"].items()
                   if k.endswith("_mean") and not k.startswith("n_")}
            },
        ]

        # Flatten for display
        display_rows = []
        for r in rows:
            display_rows.append({
                "Model": r["model"],
                "Direction": r["direction"],
                "Precision": r.get("precision_mean", float("nan")),
                "Recall": r.get("recall_mean", float("nan")),
                "F1": r.get("f1_mean", float("nan")),
                "AUC": r.get("auc_mean", float("nan")),
                "EV (%)": r.get("expected_value_mean", float("nan")),
            })

        report.print_master_table(display_rows)

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║     Pipeline Complete ✅                  ║")
    logger.info("║     Reports saved to: %-18s ║", cfg.evaluation.report_output_dir)
    logger.info("╚══════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
