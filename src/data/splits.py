from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


@dataclass
class SplitSet:
    """Holds train / validation / test slices."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series

    def sizes(self) -> Tuple[int, int, int]:
        return len(self.x_train), len(self.x_val), len(self.x_test)


def time_ordered_split(
    x: pd.DataFrame,
    y: pd.Series,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> SplitSet:
    """Slice (x, y) along the row axis preserving temporal ordering."""

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.")

    total = len(x)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return SplitSet(
        x_train=x.iloc[:train_end].copy(),
        y_train=y.iloc[:train_end].copy(),
        x_val=x.iloc[train_end:val_end].copy(),
        y_val=y.iloc[train_end:val_end].copy(),
        x_test=x.iloc[val_end:].copy(),
        y_test=y.iloc[val_end:].copy(),
    )


def group_kfold_indices(
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
    use_stratified: bool,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` group-aware folds.

    ``StratifiedGroupKFold`` is preferred when the label distribution allows
    it; fall back to ``GroupKFold`` otherwise.
    """

    indices = np.arange(len(y))
    if use_stratified:
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
            yield from splitter.split(indices.reshape(-1, 1), y, groups=groups)
            return
        except ValueError:
            # Falls back to plain GroupKFold below.
            pass
    splitter = GroupKFold(n_splits=n_splits)
    yield from splitter.split(indices.reshape(-1, 1), y, groups=groups)


def carve_validation_from_train(
    train_idx: np.ndarray,
    groups: pd.Series,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pull a contiguous group-aware validation slice from a training fold."""

    train_groups = groups.iloc[train_idx]
    unique_groups = list(dict.fromkeys(train_groups.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    n_val = max(1, int(len(unique_groups) * val_ratio))
    val_group_set = set(unique_groups[:n_val])
    is_val = train_groups.isin(val_group_set).to_numpy()
    return train_idx[~is_val], train_idx[is_val]
