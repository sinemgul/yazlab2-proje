from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from scipy import stats

    _SCIPY_AVAILABLE = True
except Exception:                    
    stats = None                            
    _SCIPY_AVAILABLE = False


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


def compute_classification_metrics(
    y_true: Iterable[int], y_pred: Iterable[int], positive_label: int = 1
) -> ClassificationMetrics:

    y_true_arr = np.asarray(list(y_true)).astype(int)
    y_pred_arr = np.asarray(list(y_pred)).astype(int)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true_arr.size == 0:
        return ClassificationMetrics(0.0, 0.0, 0.0, 0.0, 0)

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true_arr, y_pred_arr)),
        precision=float(
            precision_score(y_true_arr, y_pred_arr, pos_label=positive_label, zero_division=0)
        ),
        recall=float(
            recall_score(y_true_arr, y_pred_arr, pos_label=positive_label, zero_division=0)
        ),
        f1=float(f1_score(y_true_arr, y_pred_arr, pos_label=positive_label, zero_division=0)),
        support=int(y_true_arr.size),
    )


def confusion(y_true: Iterable[int], y_pred: Iterable[int]) -> np.ndarray:

    return confusion_matrix(list(y_true), list(y_pred), labels=[0, 1])


def aggregate_metrics(metrics: Sequence[ClassificationMetrics]) -> dict:

    if not metrics:
        return {}
    arr = np.array([[m.accuracy, m.precision, m.recall, m.f1] for m in metrics])
    means = arr.mean(axis=0)
    stds = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
    return {
        "accuracy_mean": float(means[0]),
        "accuracy_std": float(stds[0]),
        "precision_mean": float(means[1]),
        "precision_std": float(stds[1]),
        "recall_mean": float(means[2]),
        "recall_std": float(stds[2]),
        "f1_mean": float(means[3]),
        "f1_std": float(stds[3]),
        "n_runs": int(len(metrics)),
    }


                                                                             
                                        
                                                                             


def roc_pr_summary(y_true: Sequence[int], scores: Sequence[float]) -> dict:

    y_arr = np.asarray(list(y_true)).astype(int)
    s_arr = np.asarray(list(scores)).astype(float)
    if y_arr.size == 0 or len(np.unique(y_arr)) < 2:
        return {
            "available": False,
            "reason": "need both classes to compute ROC/PR",
        }
    fpr, tpr, _ = roc_curve(y_arr, s_arr)
    precision, recall, _ = precision_recall_curve(y_arr, s_arr)
    return {
        "available": True,
        "roc_auc": float(roc_auc_score(y_arr, s_arr)),
        "pr_auc": float(average_precision_score(y_arr, s_arr)),
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "pr_precision": precision.tolist(),
        "pr_recall": recall.tolist(),
    }


                                                                             
                   
                                                                             


def wilcoxon_signed_rank(scores_a: Sequence[float], scores_b: Sequence[float]) -> dict:

    if not _SCIPY_AVAILABLE:
        return {"available": False, "reason": "scipy not installed"}
    if len(scores_a) != len(scores_b):
        raise ValueError("paired score arrays must have the same length")
    if len(scores_a) < 2:
        return {"available": True, "reason": "need at least 2 paired runs"}

    stat, p_value = stats.wilcoxon(scores_a, scores_b, zero_method="zsplit")
    return {
        "available": True,
        "statistic": float(stat),
        "p_value": float(p_value),
        "n": len(scores_a),
    }


def mcnemar_test(y_true: Sequence[int], y_pred_a: Sequence[int], y_pred_b: Sequence[int]) -> dict:

    y_true_arr = np.asarray(list(y_true)).astype(int)
    a = np.asarray(list(y_pred_a)).astype(int) == y_true_arr
    b = np.asarray(list(y_pred_b)).astype(int) == y_true_arr
    n01 = int(np.sum(~a & b))                      
    n10 = int(np.sum(a & ~b))                      

    if n01 + n10 == 0:
        return {
            "available": True,
            "statistic": 0.0,
            "p_value": 1.0,
            "n01": n01,
            "n10": n10,
        }

    if not _SCIPY_AVAILABLE:
        return {
            "available": False,
            "reason": "scipy not installed",
            "n01": n01,
            "n10": n10,
        }

                                                                               
    if n01 + n10 < 25:
        p_value = float(stats.binom.cdf(min(n01, n10), n01 + n10, 0.5) * 2.0)
        statistic = float(min(n01, n10))
    else:
        statistic = float(((abs(n01 - n10) - 1) ** 2) / (n01 + n10))
        p_value = float(1.0 - stats.chi2.cdf(statistic, df=1))

    return {
        "available": True,
        "statistic": statistic,
        "p_value": p_value,
        "n01": n01,
        "n10": n10,
    }
