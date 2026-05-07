"""Counterfactual analysis for the probabilistic automaton (optional bonus).

The module asks "what would the decision look like if a single SAX symbol of
the observed pattern were replaced by another letter?" The output is a JSON
record per pattern showing the perturbed alternatives, their nearest known
state, the resulting transition probability and the decision change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.automata.automaton import ProbabilisticAutomaton


@dataclass
class CounterfactualOutcome:
    perturbed_pattern: str
    nearest_state: str
    status: str
    nearest_distance: int | None
    transition_probability: float
    decision: str


@dataclass
class CounterfactualReport:
    original_pattern: str
    base_state: str | None
    base_transition_probability: float
    base_decision: str
    alternatives: List[CounterfactualOutcome] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "original_pattern": self.original_pattern,
            "base_state": self.base_state,
            "base_transition_probability": self.base_transition_probability,
            "base_decision": self.base_decision,
            "alternatives": [
                {
                    "perturbed_pattern": alt.perturbed_pattern,
                    "nearest_state": alt.nearest_state,
                    "status": alt.status,
                    "nearest_distance": alt.nearest_distance,
                    "transition_probability": alt.transition_probability,
                    "decision": alt.decision,
                }
                for alt in self.alternatives
            ],
        }


def counterfactual_for_pattern(
    automaton: ProbabilisticAutomaton,
    previous_state: str | None,
    pattern: str,
) -> CounterfactualReport:
    """Generate the counterfactual outcomes obtained by single-symbol edits."""

    base_state, base_status, _, _ = automaton.resolve_state(pattern)
    if previous_state is None:
        base_p = 1.0
    else:
        base_p = automaton.transition_probability(previous_state, base_state)
    base_decision = (
        "anomaly"
        if base_status == "unseen"
        or (
            previous_state is not None
            and base_p < automaton.transition_probability_threshold
        )
        else "normal"
    )

    alternatives: list[CounterfactualOutcome] = []
    letters = automaton.encoder.letters
    for index in range(len(pattern)):
        for letter in letters:
            if letter == pattern[index]:
                continue
            perturbed = pattern[:index] + letter + pattern[index + 1 :]
            resolved, status, _, distance = automaton.resolve_state(perturbed)
            transition_p = (
                automaton.transition_probability(previous_state, resolved)
                if previous_state is not None
                else 1.0
            )
            decision = (
                "anomaly"
                if status == "unseen"
                or (
                    previous_state is not None
                    and transition_p < automaton.transition_probability_threshold
                )
                else "normal"
            )
            alternatives.append(
                CounterfactualOutcome(
                    perturbed_pattern=perturbed,
                    nearest_state=resolved,
                    status=status,
                    nearest_distance=distance,
                    transition_probability=transition_p,
                    decision=decision,
                )
            )
    return CounterfactualReport(
        original_pattern=pattern,
        base_state=base_state,
        base_transition_probability=base_p,
        base_decision=base_decision,
        alternatives=alternatives,
    )
