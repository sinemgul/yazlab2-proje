"""Scenario generators (original / Gaussian noise / unseen patterns).

The **unseen** scenario follows PDF Bölüm VI.A: test data is not artificially
corrupted.  Patterns absent from the training SAX dictionary are labelled
``unseen`` at inference time by :class:`~src.automata.automaton.ProbabilisticAutomaton`.
"""

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
    """Unseen evaluation uses unmodified test features (PDF VI.A).

    Unseen patterns are those whose SAX representation was never observed
    during training; detection is handled inside the automaton pipeline.
    """

    return ScenarioOutput(name="unseen", x=x.copy(), y=y.copy())
