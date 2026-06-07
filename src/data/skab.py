from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import SkabConfig


def _iter_csv_files(root_dir: Path, folder_names: Iterable[str]) -> Iterable[Path]:
    for folder in folder_names:
        folder_path = root_dir / folder
        if not folder_path.exists():
            continue
        for csv_path in sorted(folder_path.glob("*.csv")):
            yield csv_path


def load_skab_valves(cfg: SkabConfig) -> pd.DataFrame:
    """Concatenate every valve1 / valve2 csv into a single dataframe.

    Adds ``source_group`` (folder name) and ``source_file`` (csv filename)
    bookkeeping columns. These columns must NOT be used as model inputs;
    they are reserved for split definition and result analysis.
    """

    frames: list[pd.DataFrame] = []
    for csv_path in _iter_csv_files(cfg.root_dir, cfg.include_folders):
        source_group = csv_path.parent.name
        df = pd.read_csv(csv_path, sep=cfg.csv_separator)
        df.columns = [str(c).strip() for c in df.columns]
        df[cfg.source_group_col] = source_group
        df[cfg.source_file_col] = csv_path.name
        frames.append(df)

    if not frames:
        raise ValueError(
            f"No SKAB csv files found under {cfg.root_dir} for folders {cfg.include_folders}."
        )

    combined = pd.concat(frames, ignore_index=True)
    if cfg.target_col not in combined.columns:
        raise ValueError(
            f"Missing target column '{cfg.target_col}' in SKAB. Columns: {list(combined.columns)}"
        )
    combined[cfg.target_col] = combined[cfg.target_col].astype(int)
    return combined


def build_skab_features_target(
    df: pd.DataFrame,
    datetime_col: str,
    target_col: str,
    source_group_col: str,
    source_file_col: str,
    changepoint_col: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop the bookkeeping columns and return (features, target)."""

    drop_cols = {datetime_col, target_col, source_group_col, source_file_col}
    if changepoint_col and changepoint_col in df.columns:
        drop_cols.add(changepoint_col)
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols], df[target_col].astype(int)
