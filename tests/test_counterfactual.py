"""Unit tests for the counterfactual analysis helper."""

import numpy as np

from src.automata.automaton import ProbabilisticAutomaton
from src.automata.counterfactual import counterfactual_for_pattern
from src.config import AutomataConfig


def _automaton() -> ProbabilisticAutomaton:
    cfg = AutomataConfig(paa_segments=4, window_size=4, alphabet_size=3)
    automaton = ProbabilisticAutomaton.from_config(cfg)
    series = np.tile([-2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0], 8)
    automaton.fit([series])
    return automaton


def test_counterfactual_emits_one_alternative_per_edit() -> None:
    automaton = _automaton()
    pattern = automaton.states[0]
    report = counterfactual_for_pattern(automaton, previous_state=pattern, pattern=pattern)
    expected_count = len(pattern) * (automaton.encoder.alphabet_size - 1)
    assert len(report.alternatives) == expected_count


def test_counterfactual_includes_base_decision() -> None:
    automaton = _automaton()
    pattern = automaton.states[0]
    report = counterfactual_for_pattern(automaton, previous_state=None, pattern=pattern)
    assert report.base_decision in {"normal", "anomaly"}
    assert report.base_state == pattern


def test_counterfactual_handles_unseen_pattern() -> None:
    automaton = _automaton()
    report = counterfactual_for_pattern(
        automaton, previous_state=automaton.states[0], pattern="zzzz"
    )
    # The base pattern is unseen so the base decision must be anomaly.
    assert report.base_decision == "anomaly"
