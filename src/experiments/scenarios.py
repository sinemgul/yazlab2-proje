"""Scenario generators (original / Gaussian noise / unseen patterns)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScenarioOutput:
    name: str
    x: np.ndarray
    y: np.ndarray


def original_scenario(x: np.ndarray, y: np.ndarray) -> ScenarioOutput:
    return ScenarioOutput(name="original", x=x.copy(), y=y.copy())


def gaussian_noise_scenario(
    x: np.ndarray, y: np.ndarray, std: float, seed: int
) -> ScenarioOutput:
    rng = np.random.default_rng(seed)
    noisy = x + rng.normal(loc=0.0, scale=std, size=x.shape)
    return ScenarioOutput(name="noise", x=noisy, y=y.copy())


def unseen_scenario(
    x: np.ndarray, y: np.ndarray, inject_ratio: float, seed: int
) -> ScenarioOutput:
    """Replace a fraction of samples with extreme values to force unseen
    SAX patterns at test time."""

    rng = np.random.default_rng(seed)
    x_modified = x.copy()
    n = x_modified.shape[0]
    n_to_inject = max(1, int(n * inject_ratio))
    if n_to_inject == 0 or n == 0:
        return ScenarioOutput(name="unseen", x=x_modified, y=y.copy())

    indices = rng.choice(n, size=n_to_inject, replace=False)
    # Push the injected values well outside the typical range so the SAX
    # encoder produces patterns that the training-time dictionary does not
    # contain.
    extreme_value = float(np.nanmax(np.abs(x_modified)) + 5.0) if n > 0 else 5.0
    x_modified[indices] = extreme_value * rng.choice([-1.0, 1.0], size=n_to_inject)[
        :, None
    ] if x_modified.ndim == 2 else extreme_value * rng.choice([-1.0, 1.0], size=n_to_inject)
    return ScenarioOutput(name="unseen", x=x_modified, y=y.copy())
