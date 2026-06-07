from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.stats import norm


def piecewise_aggregate_approximation(series: Sequence[float], segments: int) -> np.ndarray:

    if segments <= 0:
        raise ValueError("segments must be positive")
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        raise ValueError("PAA expects a 1-D series")
    n = arr.shape[0]
    if n == 0:
        raise ValueError("empty series")
    if segments > n:
        raise ValueError("segments cannot exceed series length")

                                                                             
    if n % segments == 0:
        return arr.reshape(segments, n // segments).mean(axis=1)

    indices = np.linspace(0, n, segments + 1)
    out = np.empty(segments, dtype=float)
    for i in range(segments):
        start = indices[i]
        end = indices[i + 1]
                                                                          
                                             
        lo = int(np.floor(start))
        hi = int(np.ceil(end))
        weights = np.minimum(np.arange(lo + 1, hi + 1), end) - np.maximum(
            np.arange(lo, hi), start
        )
        weights = np.clip(weights, 0.0, None)
        out[i] = float(np.dot(arr[lo:hi], weights) / weights.sum())
    return out


def sax_breakpoints(alphabet_size: int) -> np.ndarray:

    if alphabet_size < 2:
        raise ValueError("alphabet_size must be >= 2")
    return norm.ppf(np.linspace(0, 1, alphabet_size + 1))[1:-1]


def sax_letters(alphabet_size: int) -> List[str]:

    if alphabet_size > 26:
        raise ValueError("alphabet_size must be <= 26 for the default letter set")
    return [chr(ord("a") + i) for i in range(alphabet_size)]


def _digitize(values: np.ndarray, breakpoints: np.ndarray) -> np.ndarray:
    return np.digitize(values, breakpoints)


def sax_transform(series: Sequence[float], segments: int, alphabet_size: int) -> str:

    paa = piecewise_aggregate_approximation(series, segments)
    breakpoints = sax_breakpoints(alphabet_size)
    letters = sax_letters(alphabet_size)
    indices = _digitize(paa, breakpoints)
    return "".join(letters[i] for i in indices)


@dataclass
class SaxEncoder:

    paa_segments: int
    alphabet_size: int

    @property
    def breakpoints(self) -> np.ndarray:
        return sax_breakpoints(self.alphabet_size)

    @property
    def letters(self) -> List[str]:
        return sax_letters(self.alphabet_size)

    def encode_window(self, window: Sequence[float]) -> str:
        return sax_transform(window, self.paa_segments, self.alphabet_size)


def sliding_windows(
    series: Sequence[float], window_size: int, stride: int = 1
) -> List[np.ndarray]:

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    arr = np.asarray(series, dtype=float)
    n = arr.shape[0]
    if window_size > n:
        return []
    return [arr[i : i + window_size] for i in range(0, n - window_size + 1, stride)]
