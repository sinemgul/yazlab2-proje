from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    _MPL_AVAILABLE = False


def _ensure_mpl() -> bool:
    return _MPL_AVAILABLE


def plot_confusion_matrix(cm: np.ndarray, title: str, output_path: Path) -> None:
    """Render a 2x2 confusion matrix to disk."""

    if not _ensure_mpl():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["normal", "anomaly"])
    ax.set_yticklabels(["normal", "anomaly"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_transition_heatmap(matrix: pd.DataFrame, title: str, output_path: Path) -> None:
    """Render a transition probability heatmap."""

    if not _ensure_mpl():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    if matrix.shape[0] <= 30:
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_xticklabels(matrix.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(matrix.index, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_state_diagram(
    matrix: pd.DataFrame, title: str, output_path: Path, top_k_edges: int = 30
) -> None:
    """Render a (best-effort) state-diagram visualisation using a circular layout."""

    if not _ensure_mpl():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    states = list(matrix.index)
    n = len(states)
    if n == 0:
        return
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.column_stack([np.cos(theta), np.sin(theta)])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(positions[:, 0], positions[:, 1], s=80, color="#1f77b4", zorder=3)
    if n <= 30:
        for (x, y), label in zip(positions, states):
            ax.text(x * 1.08, y * 1.08, label, ha="center", va="center", fontsize=7)

    flat = matrix.to_numpy().flatten()
    if flat.size == 0:
        plt.close(fig)
        return
    threshold = np.sort(flat)[-min(top_k_edges, len(flat))]
    for i, src in enumerate(states):
        for j, dst in enumerate(states):
            p = float(matrix.iat[i, j])
            if p < threshold or p <= 0.0:
                continue
            ax.annotate(
                "",
                xy=positions[j],
                xytext=positions[i],
                arrowprops=dict(arrowstyle="->", alpha=min(0.8, p * 4), color="black"),
            )
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_parameter_sensitivity(
    summary_df: pd.DataFrame,
    metric_col: str,
    output_path: Path,
    title: str = "Parameter sensitivity",
) -> None:
    """Plot mean metric across the (window_size, alphabet_size) grid."""

    if not _ensure_mpl():
        return
    if not {"window_size", "alphabet_size", metric_col}.issubset(summary_df.columns):
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = summary_df.pivot_table(
        index="window_size",
        columns="alphabet_size",
        values=metric_col,
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pivot.to_numpy(), cmap="magma", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("alphabet_size")
    ax.set_ylabel("window_size")
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(
                j,
                i,
                f"{pivot.iat[i, j]:.2f}" if not pd.isna(pivot.iat[i, j]) else "-",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_all_figures(
    transitions_csv_dir: Path,
    figures_dir: Path,
    automata_summary_paths: Iterable[Path] = (),
) -> None:
    """Walk through transitions CSVs and parameter summaries to render figures."""

    if not _ensure_mpl():
        return
    figures_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(Path(transitions_csv_dir).glob("*_transitions.csv")):
        matrix = pd.read_csv(csv_path, index_col=0)
        plot_transition_heatmap(
            matrix=matrix,
            title=f"Transition heatmap — {csv_path.stem}",
            output_path=figures_dir / f"heatmap_{csv_path.stem}.png",
        )
        plot_state_diagram(
            matrix=matrix,
            title=f"State diagram — {csv_path.stem}",
            output_path=figures_dir / f"diagram_{csv_path.stem}.png",
        )

    for summary_path in automata_summary_paths:
        if not Path(summary_path).exists():
            continue
        df = pd.read_csv(summary_path)
        for metric in (
            "f1_mean",
            "accuracy_mean",
            "n_states_mean",
            "transition_density_mean",
        ):
            if metric in df.columns:
                plot_parameter_sensitivity(
                    summary_df=df,
                    metric_col=metric,
                    output_path=figures_dir
                    / f"sensitivity_{Path(summary_path).stem}_{metric}.png",
                    title=f"{Path(summary_path).stem} — {metric}",
                )
