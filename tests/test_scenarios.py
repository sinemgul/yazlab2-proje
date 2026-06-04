"""Unit tests for experiment scenarios (PDF VI.A unseen definition)."""

import numpy as np

from src.experiments.scenarios import (
    dictionary_unseen_scenario,
    gaussian_noise_scenario,
    original_scenario,
)


def test_dictionary_unseen_does_not_modify_features() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    out = dictionary_unseen_scenario(x, y)
    assert out.name == "unseen"
    assert np.array_equal(out.x, x)
    assert np.array_equal(out.y, y)


def test_noise_scenario_changes_values() -> None:
    x = np.ones((10, 3))
    y = np.zeros(10, dtype=int)
    out = gaussian_noise_scenario(x, y, std=0.5, seed=42)
    assert not np.allclose(out.x, x)


def test_original_is_copy() -> None:
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([0, 0, 1])
    out = original_scenario(x, y)
    out.x[0] = 99.0
    assert x[0] == 1.0
