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


def dictionary_unseen_scenario(x: np.ndarray, y: np.ndarray) -> ScenarioOutput:

    return ScenarioOutput(name="unseen", x=x.copy(), y=y.copy())
