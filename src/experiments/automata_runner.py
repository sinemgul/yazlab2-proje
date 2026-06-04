"""Train / evaluate the probabilistic automaton over different scenarios.

This module wires together preprocessing, the SAX encoder, the automaton and
the evaluation utilities so the pipeline runner can call a single function
per dataset / scenario / parameter combination.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.automata.automaton import ProbabilisticAutomaton
from src.automata.explainability import (
    confidence_score_summary,
    explanations_to_dataframe,
)
from src.automata.sax import SaxEncoder
from src.config import AutomataConfig
from src.evaluation.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    confusion,
)


@dataclass
class AutomataRunResult:
    metrics: ClassificationMetrics
    confusion_matrix: np.ndarray
    n_states: int
    n_unique_test_patterns: int
    n_unseen_test_patterns: int
    explanations_summary: dict
    detection_rate: float = 0.0
    mapping_accuracy: float = 0.0
    train_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    sax_dictionary_size: int = 0
    n_transition_edges: int = 0
    transition_density: float = 0.0
    avg_nearest_distance_unseen: float = 0.0
    anomaly_score: Optional[np.ndarray] = None
    y_true_aligned: Optional[np.ndarray] = None
    y_pred_aligned: Optional[np.ndarray] = None


def _to_1d_series(x: np.ndarray) -> np.ndarray:
    """Flatten an ``(n, d)`` array down to ``(n,)`` (assumes PCA d == 1)."""

    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.ravel()
    raise ValueError(
        "Automaton expects a 1-D series; apply PCA(n_components=1) first."
    )


def _window_labels(y: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Aggregate row labels to per-window labels (any anomaly within the
    window flags the window as anomalous)."""

    arr = np.asarray(y).astype(int)
    n = arr.shape[0]
    if n < window_size:
        return np.empty((0,), dtype=int)
    labels: list[int] = []
    for start in range(0, n - window_size + 1, stride):
        labels.append(int(arr[start : start + window_size].max()))
    return np.array(labels, dtype=int)


def fit_automaton(
    train_series_groups: Iterable[np.ndarray], cfg: AutomataConfig
) -> Tuple[ProbabilisticAutomaton, float]:
    """Build and fit a :class:`ProbabilisticAutomaton` from training groups.

    Returns the fitted automaton and the wall-clock training time in seconds.
    """

    automaton = ProbabilisticAutomaton.from_config(cfg)
    start = time.perf_counter()
    automaton.fit(_to_1d_series(s) for s in train_series_groups)
    return automaton, time.perf_counter() - start


def evaluate_automaton(
    automaton: ProbabilisticAutomaton,
    test_series: np.ndarray,
    test_labels: np.ndarray,
    mapping_distance_threshold: int = 1,
) -> AutomataRunResult:
    """Evaluate the automaton on a single test series.

    ``mapping_distance_threshold`` controls when a Levenshtein-based mapping
    is considered "successful" for the unseen mapping accuracy metric.
    """

    series_1d = _to_1d_series(test_series)
    inference_start = time.perf_counter()
    explanations = automaton.explain_sequence(series_1d)
    inference_time = time.perf_counter() - inference_start

    if not explanations:
        empty_metrics = ClassificationMetrics(0.0, 0.0, 0.0, 0.0, 0)
        structure = automaton.transition_structure_metrics()
        return AutomataRunResult(
            metrics=empty_metrics,
            confusion_matrix=np.zeros((2, 2), dtype=int),
            n_states=int(structure["n_states"]),
            n_unique_test_patterns=0,
            n_unseen_test_patterns=0,
            explanations_summary=confidence_score_summary([]),
            detection_rate=0.0,
            mapping_accuracy=0.0,
            inference_time_sec=inference_time,
            sax_dictionary_size=int(structure["sax_dictionary_size"]),
            n_transition_edges=int(structure["n_transition_edges"]),
            transition_density=float(structure["transition_density"]),
        )

    y_pred = np.array([1 if step.decision == "anomaly" else 0 for step in explanations])
    y_true = _window_labels(test_labels, automaton.window_size, automaton.stride)

    aligned_len = min(len(y_pred), len(y_true))
    y_pred = y_pred[:aligned_len]
    y_true = y_true[:aligned_len]

    metrics = compute_classification_metrics(y_true, y_pred)
    cm = confusion(y_true, y_pred)
    test_patterns = {step.pattern for step in explanations}
    unseen_patterns = {
        step.pattern for step in explanations if step.status == "unseen"
    }

    # ----- Detection rate / mapping accuracy for the unseen scenario -----
    unseen_steps = [s for s in explanations if s.status == "unseen"]
    detection_rate = (
        float(sum(1 for s in unseen_steps if s.decision == "anomaly") / len(unseen_steps))
        if unseen_steps
        else 0.0
    )
    mapping_accuracy = (
        float(
            sum(
                1
                for s in unseen_steps
                if s.nearest_distance is not None
                and s.nearest_distance <= mapping_distance_threshold
            )
            / len(unseen_steps)
        )
        if unseen_steps
        else 0.0
    )
    distances = [
        float(s.nearest_distance)
        for s in unseen_steps
        if s.nearest_distance is not None
    ]
    avg_nearest_distance_unseen = float(np.mean(distances)) if distances else 0.0

    structure = automaton.transition_structure_metrics()

    # Anomaly score for ROC/PR: higher = more anomalous => use 1 - confidence.
    anomaly_score = np.array([1.0 - float(s.confidence) for s in explanations[:aligned_len]])

    return AutomataRunResult(
        metrics=metrics,
        confusion_matrix=cm,
        n_states=int(structure["n_states"]),
        n_unique_test_patterns=len(test_patterns),
        n_unseen_test_patterns=len(unseen_patterns),
        explanations_summary=confidence_score_summary(explanations),
        detection_rate=detection_rate,
        mapping_accuracy=mapping_accuracy,
        inference_time_sec=inference_time,
        sax_dictionary_size=int(structure["sax_dictionary_size"]),
        n_transition_edges=int(structure["n_transition_edges"]),
        transition_density=float(structure["transition_density"]),
        avg_nearest_distance_unseen=avg_nearest_distance_unseen,
        anomaly_score=anomaly_score,
        y_true_aligned=y_true,
        y_pred_aligned=y_pred,
    )


def encoder_from_config(cfg: AutomataConfig) -> SaxEncoder:
    return SaxEncoder(paa_segments=cfg.paa_segments, alphabet_size=cfg.alphabet_size)


def collect_explanation_dataframe(
    automaton: ProbabilisticAutomaton, series: np.ndarray
):
    return explanations_to_dataframe(automaton.explain_sequence(_to_1d_series(series)))
