"""Explainability artefacts for the probabilistic automaton."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from src.automata.automaton import ProbabilisticAutomaton, StepExplanation


def explanations_to_dataframe(explanations: Iterable[StepExplanation]) -> pd.DataFrame:
    """Convert a list of StepExplanation entries into a tidy DataFrame."""

    rows: List[dict] = []
    for step in explanations:
        row = {
            "time_step": step.time_step,
            "state": step.state,
            "pattern": step.pattern,
            "status": step.status,
            "mapped_to": step.mapped_to,
            "nearest_distance": step.nearest_distance,
            "probability": step.path_probability,
            "log_probability": step.log_path_probability,
            "decision": step.decision,
            "confidence": step.confidence,
            "transitions": [
                {"from": t.from_state, "to": t.to_state, "probability": t.probability}
                for t in step.transitions
            ],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_explanations_jsonl(
    explanations: Iterable[StepExplanation], path: Path
) -> None:
    """Persist explanations to a JSON lines file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for step in explanations:
            f.write(json.dumps(step.to_dict()) + "\n")


def transition_matrix(automaton: ProbabilisticAutomaton) -> pd.DataFrame:
    """Return the dense transition probability matrix as a DataFrame."""

    states = automaton.states
    matrix = np.zeros((len(states), len(states)), dtype=float)
    for i, src in enumerate(states):
        for j, dst in enumerate(states):
            matrix[i, j] = automaton.transition_probability(src, dst)
    return pd.DataFrame(matrix, index=states, columns=states)


def confidence_score_summary(explanations: Iterable[StepExplanation]) -> dict:
    """Aggregate confidence scores into a small summary dict."""

    explanations = list(explanations)
    if not explanations:
        return {"count": 0}
    confidences = np.array([s.confidence for s in explanations])
    return {
        "count": int(len(confidences)),
        "mean_confidence": float(confidences.mean()),
        "min_confidence": float(confidences.min()),
        "max_confidence": float(confidences.max()),
        "anomaly_rate": float(
            sum(1 for s in explanations if s.decision == "anomaly") / len(explanations)
        ),
    }
