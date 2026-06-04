"""Unit tests for the probabilistic automaton."""

import numpy as np

from src.automata.automaton import ProbabilisticAutomaton
from src.automata.sax import SaxEncoder
from src.config import AutomataConfig


def _series_with_clear_transitions() -> np.ndarray:
    return np.tile([-2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0], 8)


def _build_automaton(window_size: int = 4, alphabet_size: int = 3) -> ProbabilisticAutomaton:
    cfg = AutomataConfig(
        paa_segments=window_size,
        window_size=window_size,
        alphabet_size=alphabet_size,
    )
    return ProbabilisticAutomaton.from_config(cfg)


def test_fit_collects_states_and_transitions() -> None:
    automaton = _build_automaton()
    automaton.fit([_series_with_clear_transitions()])
    assert automaton.states, "automaton should learn at least one state"
    for state, row in automaton.transition_probs.items():
        total = sum(row.values())
        assert total > 0, f"state {state} has zero outgoing probability mass"


def test_sax_dictionary_populated_on_fit() -> None:
    automaton = _build_automaton()
    series = _series_with_clear_transitions()
    automaton.fit([series])
    encoded = set(automaton.encode_sequence(series))
    assert encoded.issubset(automaton.sax_dictionary)
    assert len(automaton.sax_dictionary) >= len(automaton.states)


def test_unseen_pattern_maps_via_levenshtein() -> None:
    automaton = _build_automaton()
    automaton.fit([_series_with_clear_transitions()])
    known_pattern = next(iter(automaton.sax_dictionary))
    unseen = "z" * len(known_pattern)
    assert unseen not in automaton.sax_dictionary
    state, status, mapped_to, distance = automaton.resolve_state(unseen)
    assert status == "unseen"
    assert mapped_to in automaton.states
    assert state == mapped_to
    assert distance is not None and distance >= 0


def test_seen_pattern_uses_dictionary() -> None:
    automaton = _build_automaton()
    automaton.fit([_series_with_clear_transitions()])
    known = next(iter(automaton.sax_dictionary))
    state, status, mapped_to, distance = automaton.resolve_state(known)
    assert status == "seen"
    assert state == known
    assert mapped_to is None
    assert distance == 0


def test_transition_structure_metrics() -> None:
    automaton = _build_automaton()
    automaton.fit([_series_with_clear_transitions()])
    metrics = automaton.transition_structure_metrics()
    assert metrics["n_states"] > 0
    assert metrics["transition_density"] >= 0.0
    assert metrics["sax_dictionary_size"] >= metrics["n_states"]


def test_explain_sequence_emits_one_step_per_window() -> None:
    automaton = _build_automaton()
    series = _series_with_clear_transitions()
    automaton.fit([series])
    explanations = automaton.explain_sequence(series)
    assert len(explanations) == len(series) - automaton.window_size + 1
    final_step = explanations[-1]
    assert final_step.path_probability >= 0.0


def test_explain_sequence_uses_log_probability_for_long_sequences() -> None:
    automaton = _build_automaton()
    automaton.fit([_series_with_clear_transitions()])
    long_series = np.tile(_series_with_clear_transitions(), 20)
    explanations = automaton.explain_sequence(long_series)
    assert explanations[-1].log_path_probability <= 0.0


def test_anomaly_threshold_triggers_decision() -> None:
    # Setting the per-step transition threshold above 1.0 forces every
    # non-initial decision to be flagged as anomalous.
    cfg = AutomataConfig(
        paa_segments=4,
        window_size=4,
        alphabet_size=3,
        transition_probability_threshold=2.0,
    )
    automaton = ProbabilisticAutomaton.from_config(cfg)
    series = _series_with_clear_transitions()
    automaton.fit([series])
    explanations = automaton.explain_sequence(series)
    decisions = {step.decision for step in explanations[1:]}
    assert "anomaly" in decisions


def test_encoder_used_consistently() -> None:
    automaton = _build_automaton()
    encoder = SaxEncoder(paa_segments=4, alphabet_size=3)
    series = _series_with_clear_transitions()
    automaton.fit([series])
    encoded = automaton.encode_sequence(series[:4])
    assert encoded == [encoder.encode_window(series[:4])]
