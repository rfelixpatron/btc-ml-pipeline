"""Report generator: prints formatted tables and saves charts to reports/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)

# Consistent plot style
plt.rcParams.update(
    {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#0d1117",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "text.color": "#c9d1d9",
        "grid.color": "#21262d",
        "figure.dpi": 120,
    }
)


class ReportGenerator:
    """Generates and saves evaluation reports.

    Args:
        output_dir: Directory to save charts and CSV files.
        min_precision: Threshold below which model is flagged as unreliable.
    """

    def __init__(self, output_dir: str | Path, min_precision: float = 0.55) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_precision = min_precision

    # ── Main report entry points ──────────────────────────────────────────────

    def print_sentiment_table(self, sentiment: pd.DataFrame) -> None:
        """Print the full daily sentiment table to stdout.

        Args:
            sentiment: DataFrame with fear_greed, fear_greed_label, eodhd_vader_score.
        """
        print("\n" + "=" * 70)
        print("  DAILY SENTIMENT SCORES (90-day history)")
        print("=" * 70)
        display = sentiment.copy()
        display.index = display.index.strftime("%Y-%m-%d")
        display.index.name = "Date"
        print(tabulate(display, headers="keys", tablefmt="rounded_outline", floatfmt=".3f"))
        print()

    def print_fold_results(
        self,
        per_fold_up: pd.DataFrame,
        per_fold_down: pd.DataFrame,
        summary_up: dict,
        summary_down: dict,
        model_label: str,
    ) -> None:
        """Print per-fold results and aggregate summary.

        Args:
            per_fold_up: Per-fold metrics for UP classifier.
            per_fold_down: Per-fold metrics for DOWN classifier.
            summary_up: Aggregate stats for UP classifier.
            summary_down: Aggregate stats for DOWN classifier.
            model_label: Display name (e.g. 'Short-Term 15-Min').
        """
        print(f"\n{'=' * 70}")
        print(f"  {model_label.upper()} — UP CLASSIFIER (per fold)")
        print("=" * 70)
        self._print_fold_table(per_fold_up)
        self._print_summary(summary_up, "UP")

        print(f"\n{'=' * 70}")
        print(f"  {model_label.upper()} — DOWN CLASSIFIER (per fold)")
        print("=" * 70)
        self._print_fold_table(per_fold_down)
        self._print_summary(summary_down, "DOWN")

    def save_fold_csv(
        self,
        per_fold_up: pd.DataFrame,
        per_fold_down: pd.DataFrame,
        filename_prefix: str,
    ) -> None:
        """Save per-fold results to CSV.

        Args:
            per_fold_up: Per-fold metrics DataFrame for UP.
            per_fold_down: Per-fold metrics DataFrame for DOWN.
            filename_prefix: Prefix for output files (e.g. 'short_term').
        """
        up_path = self.output_dir / f"{filename_prefix}_up_folds.csv"
        down_path = self.output_dir / f"{filename_prefix}_down_folds.csv"
        per_fold_up.to_csv(up_path)
        per_fold_down.to_csv(down_path)
        logger.info("Saved fold CSVs: %s, %s", up_path, down_path)

    def save_confusion_matrix(
        self,
        cm: np.ndarray,
        title: str,
        filename: str,
    ) -> None:
        """Save a confusion matrix heatmap.

        Args:
            cm: 2x2 confusion matrix array.
            title: Chart title.
            filename: Output filename (without path).
        """
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            ax=ax,
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            linewidths=0.5,
        )
        ax.set_title(title, color="#f0b429", fontsize=12, pad=12)
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved confusion matrix → %s", path)

    def save_roc_curve(
        self,
        per_fold_up: pd.DataFrame,
        per_fold_down: pd.DataFrame,
        filename: str,
        title: str,
    ) -> None:
        """Save a bar chart of AUC per fold for UP and DOWN classifiers.

        Args:
            per_fold_up: Per-fold metrics for UP.
            per_fold_down: Per-fold metrics for DOWN.
            filename: Output filename.
            title: Chart title.
        """
        folds = per_fold_up.index.tolist()
        x = np.arange(len(folds))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(folds) * 0.4), 4))
        ax.bar(x - width / 2, per_fold_up["auc"], width, label="UP", color="#22c55e", alpha=0.8)
        ax.bar(x + width / 2, per_fold_down["auc"], width, label="DOWN", color="#ef4444", alpha=0.8)
        ax.axhline(0.5, linestyle="--", color="#f0b429", linewidth=1, label="Random (0.50)")
        ax.set_xlabel("Fold")
        ax.set_ylabel("AUC")
        ax.set_title(title, color="#f0b429")
        ax.set_xticks(x)
        ax.set_xticklabels(folds)
        ax.legend()
        ax.set_ylim(0, 1)
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved AUC chart → %s", path)

    def save_feature_importance(
        self,
        importance_df: pd.DataFrame,
        filename: str,
        title: str,
        top_n: int = 15,
    ) -> None:
        """Save a horizontal bar chart of feature importances.

        Args:
            importance_df: DataFrame from BaseBTCClassifier.feature_importances().
            filename: Output filename.
            title: Chart title.
            top_n: Show top N features (default 15).
        """
        if importance_df.empty:
            logger.warning("No feature importances available to plot.")
            return

        top = importance_df.head(top_n).copy()
        fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.35)))
        ax.barh(top["feature"], top["importance_avg"], color="#3b82f6", alpha=0.85)
        ax.invert_yaxis()
        ax.set_xlabel("Importance (avg UP + DOWN)")
        ax.set_title(title, color="#f0b429")
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved feature importance → %s", path)

    def print_master_table(self, rows: list[dict]) -> None:
        """Print the master comparison table across all models.

        Args:
            rows: List of dicts with keys matching the display columns.
        """
        print("\n" + "=" * 90)
        print("  MASTER COMPARISON TABLE")
        print("=" * 90)
        df = pd.DataFrame(rows)
        print(tabulate(df, headers="keys", tablefmt="rounded_outline", floatfmt=".4f", showindex=False))

        # Reliability flag (checks Precision or precision)
        print("\n  Reliability threshold: precision >= {:.2f}".format(self.min_precision))
        for row in rows:
            prec = row.get("Precision") or row.get("precision") or 0.0
            name = row.get("Model") or row.get("model") or "Unknown"
            dir_ = row.get("Direction") or row.get("direction") or ""
            flag = "✅" if prec >= self.min_precision else "⚠️ "
            print(f"  {flag}  {name} {dir_}: precision={prec:.3f}")
        print()

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _print_fold_table(per_fold: pd.DataFrame) -> None:
        cols = ["accuracy", "precision", "recall", "f1", "auc", "mcc", "expected_value"]
        available = [c for c in cols if c in per_fold.columns]
        print(
            tabulate(
                per_fold[available],
                headers=available,
                tablefmt="rounded_outline",
                floatfmt=".4f",
            )
        )

    @staticmethod
    def _print_summary(summary: dict, direction: str) -> None:
        metrics = ["accuracy", "precision", "recall", "f1", "auc", "mcc", "expected_value"]
        print(f"\n  {direction} — Aggregate (mean ± std across folds):")
        for m in metrics:
            mean_key = f"{m}_mean"
            std_key = f"{m}_std"
            if mean_key in summary:
                print(
                    f"    {m:<20} {summary[mean_key]:.4f} ± {summary.get(std_key, 0):.4f}"
                )
        print()
