"""Quick smoke test for the experiment pipeline using a tiny config.

Used during development to verify that loading, preprocessing, the automaton
and the result aggregation work end-to-end without launching the full
parameter sweep.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    AutomataConfig,
    DeepLearningConfig,
    ExperimentConfig,
    ParameterSweepConfig,
    PathsConfig,
    PreprocessingConfig,
    ProjectConfig,
    TrainingConfig,
)
from src.pipeline.runner import run_experiments


def make_smoke_config() -> ProjectConfig:
    cfg = ProjectConfig()
    cfg.training = TrainingConfig(
        random_seeds=(42,),
        batch_size=32,
        max_epochs=2,
        early_stopping_patience=1,
    )
    cfg.experiment = ExperimentConfig(
        scenarios=("original", "noise", "unseen"),
        gaussian_noise_std=0.1,
    )
    cfg.deep_learning = DeepLearningConfig(
        models=("lstm", "gru", "cnn1d"),
        sequence_length=8,
        hidden_size=8,
        num_layers=1,
        cnn_channels=8,
        cnn_kernel_size=3,
    )
    cfg.sweep = ParameterSweepConfig(window_sizes=(4,), alphabet_sizes=(3,))
    cfg.automata = AutomataConfig()
    cfg.preprocessing = PreprocessingConfig()
    cfg.paths = PathsConfig(
        artifacts_dir=Path("artifacts/smoke"),
        logs_dir=Path("artifacts/smoke/logs"),
        results_dir=Path("artifacts/smoke/results"),
        explanations_dir=Path("artifacts/smoke/explanations"),
        figures_dir=Path("artifacts/smoke/figures"),
    )
    return cfg


def main() -> None:
    cfg = make_smoke_config()
    run_experiments(cfg)


if __name__ == "__main__":
    main()
