"""Centralized configuration for the Yazlab2 project.

All experiment, model and dataset parameters live here. Hard-coded values
elsewhere in the code base are not allowed; every component must read its
parameters from a `ProjectConfig` instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------


@dataclass
class BatadalConfig:
    """Configuration for the BATADAL Training Dataset 2."""

    csv_path: Path = Path(r"c:\Users\sinem\Downloads\BATADAL_dataset04.csv")
    datetime_col: str = "DATETIME"
    target_col: str = "ATT_FLAG"
    split_train: float = 0.60
    split_val: float = 0.20
    split_test: float = 0.20
    # Strip whitespace from column headers (BATADAL ships with leading spaces).
    strip_column_whitespace: bool = True


@dataclass
class SkabConfig:
    """Configuration for the SKAB valve1 / valve2 subset."""

    root_dir: Path = Path(r"c:\Users\sinem\Downloads\archive (3)\SKAB")
    include_folders: List[str] = field(default_factory=lambda: ["valve1", "valve2"])
    datetime_col: str = "datetime"
    target_col: str = "anomaly"
    changepoint_col: str = "changepoint"
    source_group_col: str = "source_group"
    source_file_col: str = "source_file"
    csv_separator: str = ";"
    n_splits: int = 5
    use_stratified_group_kfold: bool = True


# ---------------------------------------------------------------------------
# Preprocessing configuration
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    """Shared preprocessing parameters."""

    missing_strategy: str = "ffill_then_bfill"  # ffill_then_bfill | mean | drop
    scaler: str = "standard"  # standard | minmax
    pca_n_components: int = 1
    apply_pca_for_automata: bool = True


# ---------------------------------------------------------------------------
# Symbolic / automata configuration
# ---------------------------------------------------------------------------


@dataclass
class AutomataConfig:
    """Parameters for the PAA + SAX + probabilistic automaton pipeline."""

    paa_segments: int = 4  # PAA segment count per window (defaults to window size).
    window_size: int = 4
    alphabet_size: int = 3
    stride: int = 1
    laplace_smoothing: float = 1e-6
    # Per-step transition probability below which the automaton flags the
    # current window as anomalous. Path probability is reported separately
    # for the explainability output.
    transition_probability_threshold: float = 0.05
    # Optional cumulative path probability threshold (kept for compatibility;
    # falls back to disabled when set to 0).
    path_probability_threshold: float = 0.0
    enable_levenshtein_fallback: bool = True


@dataclass
class ParameterSweepConfig:
    """Parameter sweep ranges for sensitivity analysis."""

    window_sizes: Tuple[int, ...] = (3, 4, 5, 6)
    alphabet_sizes: Tuple[int, ...] = (3, 4, 5, 6)


# ---------------------------------------------------------------------------
# Deep learning configuration
# ---------------------------------------------------------------------------


@dataclass
class DeepLearningConfig:
    """Hyper-parameters shared by all sequence DL models."""

    models: Tuple[str, ...] = ("lstm", "gru", "cnn1d")  # PDF: en az ikisi; üçünü de koşuyoruz
    sequence_length: int = 16
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    cnn_channels: int = 32
    cnn_kernel_size: int = 3


# ---------------------------------------------------------------------------
# Training / experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Training loop hyper-parameters (fixed by the project brief)."""

    random_seeds: Tuple[int, ...] = (42, 123, 2026, 7, 999)
    batch_size: int = 32
    max_epochs: int = 50
    early_stopping_patience: int = 5


@dataclass
class ExperimentConfig:
    """Experiment / scenario parameters."""

    scenarios: Tuple[str, ...] = ("original", "noise", "unseen")
    gaussian_noise_std: float = 0.1
    # Unseen senaryosu PDF VI.A: test pattern'ları train SAX sözlüğüne göre işaretlenir
    # (aşırı değer enjeksiyonu kullanılmaz).


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------


@dataclass
class PathsConfig:
    """Filesystem layout for artifacts."""

    artifacts_dir: Path = Path("artifacts")
    logs_dir: Path = Path("artifacts/logs")
    results_dir: Path = Path("artifacts/results")
    explanations_dir: Path = Path("artifacts/explanations")
    figures_dir: Path = Path("artifacts/figures")

    def ensure(self) -> None:
        for directory in [
            self.artifacts_dir,
            self.logs_dir,
            self.results_dir,
            self.explanations_dir,
            self.figures_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Aggregated configuration
# ---------------------------------------------------------------------------


@dataclass
class ProjectConfig:
    """Aggregated configuration consumed by every pipeline component."""

    batadal: BatadalConfig = field(default_factory=BatadalConfig)
    skab: SkabConfig = field(default_factory=SkabConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    automata: AutomataConfig = field(default_factory=AutomataConfig)
    sweep: ParameterSweepConfig = field(default_factory=ParameterSweepConfig)
    deep_learning: DeepLearningConfig = field(default_factory=DeepLearningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


CONFIG = ProjectConfig()
