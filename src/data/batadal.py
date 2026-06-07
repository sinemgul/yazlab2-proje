from __future__ import annotations

import pandas as pd

from src.config import BatadalConfig


def load_batadal_training_dataset2(cfg: BatadalConfig) -> pd.DataFrame:

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

                                                                             
                                                                              
                   
    df[cfg.target_col] = (df[cfg.target_col] == 1).astype(int)

                                                           
    df = df.sort_values(by=cfg.datetime_col).reset_index(drop=True)
    return df


def build_batadal_features_target(
    df: pd.DataFrame,
    datetime_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:

    feature_cols = [c for c in df.columns if c not in {datetime_col, target_col}]
    return df[feature_cols], df[target_col].astype(int)
