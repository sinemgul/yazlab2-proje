import numpy as np
import pandas as pd

from src.data.splits import (
    carve_validation_from_train,
    group_kfold_indices,
    time_ordered_split,
)


def test_time_ordered_split_preserves_order() -> None:
    x = pd.DataFrame({"feature": np.arange(10)})
    y = pd.Series(np.zeros(10, dtype=int))
    split = time_ordered_split(x, y, 0.6, 0.2, 0.2)
    assert list(split.x_train["feature"]) == [0, 1, 2, 3, 4, 5]
    assert list(split.x_val["feature"]) == [6, 7]
    assert list(split.x_test["feature"]) == [8, 9]


def test_group_kfold_no_group_overlap() -> None:
    n = 20
    y = pd.Series([0] * 10 + [1] * 10)
    groups = pd.Series([f"g{i // 4}" for i in range(n)])
    fold_iter = list(group_kfold_indices(y=y, groups=groups, n_splits=5, use_stratified=False))
    assert fold_iter, "expected at least one fold"
    for train_idx, test_idx in fold_iter:
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        assert not (train_groups & test_groups), "groups must not leak between train and test"


def test_carve_validation_from_train_groups_only() -> None:
    n = 16
    y = pd.Series([0] * 16)
    groups = pd.Series([f"g{i // 4}" for i in range(n)])
    train_idx = np.arange(n)
    inner_train, val = carve_validation_from_train(train_idx, groups, val_ratio=0.25, seed=0)
    train_groups = set(groups.iloc[inner_train])
    val_groups = set(groups.iloc[val])
    assert not (train_groups & val_groups)
    assert len(inner_train) + len(val) == n
