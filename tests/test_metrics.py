"""Unit tests for the metric helpers."""

import numpy as np

from src.evaluation.metrics import (
    aggregate_metrics,
    compute_classification_metrics,
    mcnemar_test,
    wilcoxon_signed_rank,
)


def test_compute_classification_metrics_perfect() -> None:
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 0, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.support == 6


def test_compute_classification_metrics_empty() -> None:
    metrics = compute_classification_metrics([], [])
    assert metrics.support == 0


def test_aggregate_metrics_returns_mean_and_std() -> None:
    metrics_list = [
        compute_classification_metrics([0, 1], [0, 1]),
        compute_classification_metrics([0, 1], [1, 0]),
    ]
    summary = aggregate_metrics(metrics_list)
    assert summary["n_runs"] == 2
    assert summary["accuracy_mean"] == 0.5
    assert summary["accuracy_std"] >= 0.0


def test_mcnemar_returns_no_difference_when_predictions_identical() -> None:
    y_true = [0, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 1, 0]
    result = mcnemar_test(y_true, y_pred, y_pred)
    assert result["n01"] == 0
    assert result["n10"] == 0


def test_wilcoxon_returns_dict_when_paired_lengths_equal() -> None:
    out = wilcoxon_signed_rank([0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8])
    assert "p_value" in out or "reason" in out
