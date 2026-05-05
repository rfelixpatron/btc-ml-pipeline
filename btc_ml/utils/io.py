"""I/O helpers: parquet read/write with consistent path handling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from btc_ml.utils.logging import get_logger

logger = get_logger(__name__)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Save DataFrame to parquet, creating parent directories if needed.

    Args:
        df: DataFrame to save.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    logger.info("Saved %d rows → %s", len(df), path)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a parquet file into a DataFrame.

    Args:
        path: Source file path.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run `python scripts/download_data.py` first."
        )
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows ← %s", len(df), path)
    return df


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
