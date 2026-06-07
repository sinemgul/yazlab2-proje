from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


                                                                             
                        
                                                                             


@dataclass
class BatadalConfig:

    csv_path: Path = Path(r"c:\Users\sinem\Downloads\BATADAL_dataset04.csv")
    datetime_col: str = "DATETIME"
    target_col: str = "ATT_FLAG"
    split_train: float = 0.60
    split_val: float = 0.20
    split_test: float = 0.20
                                                                               
    strip_column_whitespace: bool = True


@dataclass
class SkabConfig:

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


                                                                             
                             
                                                                             


@dataclass
class PreprocessingConfig:

    missing_strategy: str = "ffill_then_bfill"                                  
    scaler: str = "standard"                     
    pca_n_components: int = 1
    apply_pca_for_automata: bool = True


                                                                             
                                   
                                                                             


@dataclass
class AutomataConfig:

    paa_segments: int = 4                                                           
    window_size: int = 4
    alphabet_size: int = 3
    stride: int = 1
    laplace_smoothing: float = 1e-6
                                                                         
                                                                          
                                    
    transition_probability_threshold: float = 0.05
                                                                             
                                            
    path_probability_threshold: float = 0.0
    enable_levenshtein_fallback: bool = True


@dataclass
class ParameterSweepConfig:

    window_sizes: Tuple[int, ...] = (3, 4, 5, 6)
    alphabet_sizes: Tuple[int, ...] = (3, 4, 5, 6)


                                                                             
                             
                                                                             


@dataclass
class DeepLearningConfig:

    models: Tuple[str, ...] = ("lstm", "gru", "cnn1d")                                        
    sequence_length: int = 16
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    cnn_channels: int = 32
    cnn_kernel_size: int = 3


                                                                             
                                     
                                                                             


@dataclass
class TrainingConfig:

    random_seeds: Tuple[int, ...] = (42, 123, 2026, 7, 999)
    batch_size: int = 32
    max_epochs: int = 50
    early_stopping_patience: int = 5


@dataclass
class ExperimentConfig:

    scenarios: Tuple[str, ...] = ("original", "noise", "unseen")
    gaussian_noise_std: float = 0.1
                                                                                       
                                            


                                                                             
              
                                                                             


@dataclass
class PathsConfig:

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


                                                                             
                          
                                                                             


@dataclass
class ProjectConfig:

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
