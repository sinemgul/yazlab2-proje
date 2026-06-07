from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import PreprocessingConfig


def handle_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Apply a missing-value strategy in place-safe fashion."""

    if strategy == "ffill_then_bfill":
        return df.ffill().bfill()
    if strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    if strategy == "drop":
        return df.dropna()
    raise ValueError(f"Unknown missing strategy: {strategy}")


def _make_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler: {name}")


@dataclass
class FittedPreprocessor:
    """Container for the train-fit transformers."""

    scaler: object
    pca: Optional[PCA]
    feature_columns: list[str]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_columns].to_numpy()
        x_scaled = self.scaler.transform(x)
        if self.pca is not None:
            x_scaled = self.pca.transform(x_scaled)
        return x_scaled


def fit_preprocessor(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    cfg: PreprocessingConfig,
    use_pca: bool,
) -> FittedPreprocessor:
    """Fit the scaler and (optionally) PCA on the training set only."""

    train_clean = handle_missing_values(train_df[feature_columns], cfg.missing_strategy)
    scaler = _make_scaler(cfg.scaler)
    scaler.fit(train_clean.to_numpy())

    pca: Optional[PCA] = None
    if use_pca and cfg.apply_pca_for_automata and len(feature_columns) > cfg.pca_n_components:
        pca = PCA(n_components=cfg.pca_n_components, random_state=0)
        pca.fit(scaler.transform(train_clean.to_numpy()))

    return FittedPreprocessor(scaler=scaler, pca=pca, feature_columns=feature_columns)


def add_gaussian_noise(x: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Inject zero-mean Gaussian noise with the given standard deviation."""

    if std <= 0:
        return x
    return x + rng.normal(loc=0.0, scale=std, size=x.shape)
