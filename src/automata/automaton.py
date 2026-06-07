from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.config import AutomataConfig
from src.automata.levenshtein import find_nearest_pattern
from src.automata.sax import SaxEncoder, sliding_windows


@dataclass
class TransitionExplanation:

    from_state: str
    to_state: str
    probability: float


@dataclass
class StepExplanation:

    time_step: int
    state: str
    pattern: str
    status: str                     
    mapped_to: Optional[str]
    nearest_distance: Optional[int]
    transitions: List[TransitionExplanation] = field(default_factory=list)
    path_probability: float = 0.0
    log_path_probability: float = 0.0
    decision: str = "normal"
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "time_step": self.time_step,
            "state": self.state,
            "pattern": self.pattern,
            "status": self.status,
            "mapped_to": self.mapped_to,
            "nearest_distance": self.nearest_distance,
            "transitions": [
                {
                    "from": t.from_state,
                    "to": t.to_state,
                    "probability": t.probability,
                }
                for t in self.transitions
            ],
            "probability": self.path_probability,
            "log_probability": self.log_path_probability,
            "decision": self.decision,
            "confidence": self.confidence,
        }


@dataclass
class ProbabilisticAutomaton:

    encoder: SaxEncoder
    window_size: int
    stride: int = 1
    laplace_smoothing: float = 1e-6
    transition_probability_threshold: float = 0.05
    path_probability_threshold: float = 0.0
    enable_levenshtein_fallback: bool = True

    states: List[str] = field(default_factory=list)
    sax_dictionary: set[str] = field(default_factory=set)
    transition_counts: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    transition_probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    state_outgoing_totals: Dict[str, int] = field(default_factory=dict)

                                                                        
                     
                                                                        
    @classmethod
    def from_config(cls, cfg: AutomataConfig) -> "ProbabilisticAutomaton":
        encoder = SaxEncoder(paa_segments=cfg.paa_segments, alphabet_size=cfg.alphabet_size)
        return cls(
            encoder=encoder,
            window_size=cfg.window_size,
            stride=cfg.stride,
            laplace_smoothing=cfg.laplace_smoothing,
            transition_probability_threshold=cfg.transition_probability_threshold,
            path_probability_threshold=cfg.path_probability_threshold,
            enable_levenshtein_fallback=cfg.enable_levenshtein_fallback,
        )

    def encode_sequence(self, series: Sequence[float]) -> List[str]:

        windows = sliding_windows(series, self.window_size, self.stride)
        return [self.encoder.encode_window(w) for w in windows]

                                                                        
              
                                                                        
    def fit(self, training_series: Iterable[Sequence[float]]) -> "ProbabilisticAutomaton":

        states_seen: set[str] = set()
        dictionary: set[str] = set()
        for series in training_series:
            patterns = self.encode_sequence(series)
            dictionary.update(patterns)
            for prev, curr in zip(patterns[:-1], patterns[1:]):
                self.transition_counts[prev][curr] += 1
                states_seen.add(prev)
                states_seen.add(curr)

        self.sax_dictionary = dictionary
        self.states = sorted(states_seen)
        self.transition_probs = self._compute_transition_probs(self.transition_counts)
        self.state_outgoing_totals = {
            s: int(sum(c.values())) for s, c in self.transition_counts.items()
        }
        return self

    def _compute_transition_probs(
        self, counts: Dict[str, Counter]
    ) -> Dict[str, Dict[str, float]]:
        probs: Dict[str, Dict[str, float]] = {}
        n_states = len(self.states) if self.states else 1
        for state, neighbours in counts.items():
            total = sum(neighbours.values())
            denom = total + self.laplace_smoothing * n_states
            row: Dict[str, float] = {}
            for target in self.states:
                count = neighbours.get(target, 0)
                row[target] = (count + self.laplace_smoothing) / denom if denom > 0 else 0.0
            probs[state] = row
        return probs

                                                                        
                             
                                                                        
    def transition_probability(self, from_state: str, to_state: str) -> float:
        row = self.transition_probs.get(from_state)
        if row is None:
            return self.laplace_smoothing / max(len(self.states), 1)
        return float(row.get(to_state, self.laplace_smoothing / max(len(self.states), 1)))

    def transition_structure_metrics(self) -> dict[str, float | int]:

        n_states = len(self.states)
        n_edges = sum(len(counter) for counter in self.transition_counts.values())
        n_events = sum(sum(counter.values()) for counter in self.transition_counts.values())
        density = float(n_edges / (n_states * n_states)) if n_states > 0 else 0.0
        return {
            "n_states": n_states,
            "sax_dictionary_size": len(self.sax_dictionary),
            "n_transition_edges": n_edges,
            "n_transition_events": n_events,
            "transition_density": density,
            "avg_outgoing_edges": float(n_edges / n_states) if n_states > 0 else 0.0,
        }

    def resolve_state(self, pattern: str) -> tuple[str, str, Optional[str], Optional[int]]:

        if pattern in self.sax_dictionary:
            return pattern, "seen", None, 0
        if self.enable_levenshtein_fallback and self.states:
            nearest, distance = find_nearest_pattern(pattern, self.states)
            if nearest is not None:
                return nearest, "unseen", nearest, distance
        return pattern, "unseen", None, None

    def explain_sequence(self, series: Sequence[float]) -> List[StepExplanation]:

        patterns = self.encode_sequence(series)
        explanations: List[StepExplanation] = []
        if not patterns:
            return explanations

        log_path_prob = 0.0
        path_prob = 1.0
        previous_resolved: Optional[str] = None

        for index, pattern in enumerate(patterns):
            resolved_state, status, mapped_to, distance = self.resolve_state(pattern)
            transitions: List[TransitionExplanation] = []
            last_transition_prob = 1.0
            if previous_resolved is not None:
                p = self.transition_probability(previous_resolved, resolved_state)
                last_transition_prob = p
                transitions.append(
                    TransitionExplanation(previous_resolved, resolved_state, p)
                )
                                                                         
                                                                        
                log_path_prob += math.log(max(p, 1e-300))
                path_prob *= p

            decision = "normal"
            if status == "unseen":
                decision = "anomaly"
            elif (
                previous_resolved is not None
                and last_transition_prob < self.transition_probability_threshold
            ):
                decision = "anomaly"
            elif (
                self.path_probability_threshold > 0.0
                and path_prob < self.path_probability_threshold
            ):
                decision = "anomaly"

                                                                               
                                                                             
            confidence = (
                last_transition_prob if previous_resolved is not None else 1.0
            )
            explanations.append(
                StepExplanation(
                    time_step=index,
                    state=resolved_state,
                    pattern=pattern,
                    status=status,
                    mapped_to=mapped_to,
                    nearest_distance=distance,
                    transitions=transitions,
                    path_probability=path_prob,
                    log_path_probability=log_path_prob,
                    decision=decision,
                    confidence=confidence,
                )
            )
            previous_resolved = resolved_state

        return explanations

                                                                        
                        
                                                                        
    def predict_window_anomaly(self, series: Sequence[float]) -> List[int]:

        return [
            1 if step.decision == "anomaly" else 0
            for step in self.explain_sequence(series)
        ]

    def predict_sample_anomaly(self, series: Sequence[float]) -> int:

        labels = self.predict_window_anomaly(series)
        if not labels:
            return 0
                                                                         
        return int(max(labels))


def sequence_anomaly_probability(
    automaton: ProbabilisticAutomaton, series: Sequence[float]
) -> float:

    explanations = automaton.explain_sequence(series)
    if not explanations:
        return 1.0
    return float(np.exp(explanations[-1].log_path_probability))
