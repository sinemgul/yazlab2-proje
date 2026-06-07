import math

import numpy as np
import pytest

from src.automata.sax import (
    SaxEncoder,
    piecewise_aggregate_approximation,
    sax_breakpoints,
    sax_letters,
    sax_transform,
    sliding_windows,
)


def test_paa_divisible_length() -> None:
    series = np.array([1.0, 2.0, 3.0, 4.0])
    paa = piecewise_aggregate_approximation(series, segments=2)
    assert paa.tolist() == [1.5, 3.5]


def test_paa_non_divisible_length() -> None:
    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    paa = piecewise_aggregate_approximation(series, segments=2)
    assert pytest.approx(paa[0] + paa[1], rel=1e-6) == 6.0


def test_sax_breakpoints_three_letters() -> None:
    bps = sax_breakpoints(3)
    assert len(bps) == 2
    # Breakpoints should be symmetric around 0 for an odd alphabet split.
    assert math.isclose(bps[0], -bps[1], rel_tol=1e-9)


def test_sax_letters() -> None:
    assert sax_letters(4) == ["a", "b", "c", "d"]


def test_sax_transform_length_matches_segments() -> None:
    series = np.linspace(-1, 1, 8)
    encoded = sax_transform(series, segments=4, alphabet_size=3)
    assert len(encoded) == 4
    assert set(encoded).issubset(set("abc"))


def test_sliding_windows_count() -> None:
    series = np.arange(10.0)
    windows = sliding_windows(series, window_size=4, stride=1)
    assert len(windows) == 7
    assert windows[0].tolist() == [0, 1, 2, 3]
    assert windows[-1].tolist() == [6, 7, 8, 9]


def test_sax_encoder_uses_window_settings() -> None:
    encoder = SaxEncoder(paa_segments=4, alphabet_size=3)
    encoded = encoder.encode_window(np.array([-3.0, -1.0, 1.0, 3.0]))
    # Increasing series should map to a non-decreasing letter pattern.
    assert encoded[0] <= encoded[-1]
