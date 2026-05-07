"""BATADAL Training Dataset 2 loading utilities."""

from __future__ import annotations

import pandas as pd

from src.config import BatadalConfig


def load_batadal_training_dataset2(cfg: BatadalConfig) -> pd.DataFrame:
    """Load BATADAL Training Dataset 2 from disk and validate required columns."""

    df = pd.read_csv(cfg.csv_path)
    if cfg.strip_column_whitespace:
        df.columns = [str(c).strip() for c in df.columns]

    if cfg.datetime_col not in df.columns:
        raise ValueError(
            f"Missing datetime column '{cfg.datetime_col}'. Available: {list(df.columns)}"
        )
    if cfg.target_col not in df.columns:
        raise ValueError(
            f"Missing target column '{cfg.target_col}'. Available: {list(df.columns)}"
        )

    # The BATADAL ATT_FLAG column uses values in {-999, 0, 1}. Treat anything
    # other than 1 as the normal class so the downstream pipeline sees a clean
    # binary label.
    df[cfg.target_col] = (df[cfg.target_col] == 1).astype(int)

    # Preserve temporal order: sort by datetime to be safe.
    df = df.sort_values(by=cfg.datetime_col).reset_index(drop=True)
    return df


def build_batadal_features_target(
    df: pd.DataFrame,
    datetime_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the BATADAL frame into model features and the binary target."""

    feature_cols = [c for c in df.columns if c not in {datetime_col, target_col}]
    return df[feature_cols], df[target_col].astype(int)
