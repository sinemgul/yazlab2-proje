from __future__ import annotations

from typing import Tuple

import numpy as np


def build_sliding_sequences(
    x: np.ndarray, y: np.ndarray, sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n_rows = x.shape[0]
    if n_rows < sequence_length:
        return (
            np.empty((0, sequence_length, x.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    sequences = np.lib.stride_tricks.sliding_window_view(
        x, window_shape=sequence_length, axis=0
    )
    sequences = sequences.transpose(0, 2, 1).astype(np.float32, copy=False)
    labels = y[sequence_length - 1 :]
    return sequences, labels
