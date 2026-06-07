from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from src.automata.counterfactual import counterfactual_for_pattern
from src.automata.explainability import (
    save_explanations_jsonl,
    transition_matrix,
)
from src.config import (
    AutomataConfig,
    BatadalConfig,
    DeepLearningConfig,
    ExperimentConfig,
    PathsConfig,
    PreprocessingConfig,
    ProjectConfig,
    SkabConfig,
    TrainingConfig,
)
from src.data.batadal import build_batadal_features_target, load_batadal_training_dataset2
from src.data.preprocessing import (
    FittedPreprocessor,
    add_gaussian_noise,
    fit_preprocessor,
    handle_missing_values,
)
from src.data.skab import build_skab_features_target, load_skab_valves
from src.data.splits import (
    SplitSet,
    carve_validation_from_train,
    group_kfold_indices,
    time_ordered_split,
)
from src.evaluation.metrics import (
    ClassificationMetrics,
    aggregate_metrics,
    compute_classification_metrics,
    confusion,
    mcnemar_test,
    roc_pr_summary,
    wilcoxon_signed_rank,
)
from src.evaluation.visualization import plot_confusion_matrix
from src.experiments.automata_runner import (
    AutomataRunResult,
    evaluate_automaton,
    fit_automaton,
)
from src.experiments.scenarios import (
    ScenarioOutput,
    dictionary_unseen_scenario,
    gaussian_noise_scenario,
    original_scenario,
)
from src.models.deep_learning import fit_dl_model, torch_available
from src.models.sequence_dataset import build_sliding_sequences
from src.utils.logging import append_jsonl, get_logger, write_json
from src.utils.seeding import set_global_seed


                                                                             
         
                                                                             


def _scenario_for(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    exp_cfg: ExperimentConfig,
    seed: int,
) -> ScenarioOutput:
    if name == "original":
        return original_scenario(x, y)
    if name == "noise":
        return gaussian_noise_scenario(x, y, exp_cfg.gaussian_noise_std, seed)
    if name == "unseen":
        return dictionary_unseen_scenario(x, y)
    raise ValueError(f"Unknown scenario: {name}")


def _automata_param_grid(
    cfg: AutomataConfig, sweep_window_sizes: Sequence[int], sweep_alphabet_sizes: Sequence[int]
) -> List[AutomataConfig]:

    grid: list[AutomataConfig] = []
    for window in sweep_window_sizes:
        for alphabet in sweep_alphabet_sizes:
            grid.append(
                AutomataConfig(
                    paa_segments=window,
                    window_size=window,
                    alphabet_size=alphabet,
                    stride=cfg.stride,
                    laplace_smoothing=cfg.laplace_smoothing,
                    transition_probability_threshold=cfg.transition_probability_threshold,
                    path_probability_threshold=cfg.path_probability_threshold,
                    enable_levenshtein_fallback=cfg.enable_levenshtein_fallback,
                )
            )
    return grid


def _record(payload: dict, path: Path) -> None:
    append_jsonl(path, payload)


def _save_roc_pr_curve(
    roc_pr: dict,
    title: str,
    output_path_roc: Path,
    output_path_pr: Path,
) -> None:

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not roc_pr.get("available"):
        return

    output_path_roc.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(roc_pr["roc_fpr"], roc_pr["roc_tpr"], label=f"AUC={roc_pr['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path_roc)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(roc_pr["pr_recall"], roc_pr["pr_precision"], label=f"AP={roc_pr['pr_auc']:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR — {title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path_pr)
    plt.close(fig)


def _write_counterfactual_sample(
    automaton,
    series: np.ndarray,
    output_path: Path,
    max_samples: int = 5,
) -> None:

    series_1d = np.asarray(series).ravel()
    patterns = automaton.encode_sequence(series_1d)
    if not patterns:
        return
    snapshots = []
    indices = np.linspace(0, len(patterns) - 1, num=min(max_samples, len(patterns)), dtype=int)
    for idx in indices:
        previous = patterns[idx - 1] if idx > 0 else None
        report = counterfactual_for_pattern(automaton, previous_state=previous, pattern=patterns[idx])
        snapshots.append({"time_step": int(idx), **report.to_dict()})
    write_json(output_path, {"counterfactual": snapshots})


def _cache_predictions(
    results_dir: Path,
    *,
    dataset: str,
    scenario: str,
    seed: int,
    model: str,
    y_true,
    y_pred,
    fold: int | None = None,
) -> None:

    cache_dir = results_dir / "predictions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"fold{fold}_" if fold is not None else ""
    np.savez(
        cache_dir / f"{dataset}_{model}_{scenario}_{suffix}seed{seed}.npz",
        y_true=np.asarray(y_true).astype(int),
        y_pred=np.asarray(y_pred).astype(int),
    )


                                                                             
         
                                                                             


def run_batadal(cfg: ProjectConfig, logger) -> None:
    df = load_batadal_training_dataset2(cfg.batadal)
    x_df, y = build_batadal_features_target(df, cfg.batadal.datetime_col, cfg.batadal.target_col)

    feature_columns = list(x_df.columns)
    split = time_ordered_split(
        x_df,
        y,
        train_ratio=cfg.batadal.split_train,
        val_ratio=cfg.batadal.split_val,
        test_ratio=cfg.batadal.split_test,
    )

    logger.info(
        "BATADAL split sizes (train/val/test): %s",
        split.sizes(),
    )

    automata_results_path = cfg.paths.results_dir / "batadal_automata_runs.jsonl"
    dl_results_path = cfg.paths.results_dir / "batadal_dl_runs.jsonl"

    for seed in cfg.training.random_seeds:
        set_global_seed(seed)
        for scenario_name in cfg.experiment.scenarios:
            _run_batadal_seed_scenario(
                cfg=cfg,
                seed=seed,
                scenario_name=scenario_name,
                split=split,
                feature_columns=feature_columns,
                automata_results_path=automata_results_path,
                dl_results_path=dl_results_path,
                logger=logger,
            )


def _run_batadal_seed_scenario(
    *,
    cfg: ProjectConfig,
    seed: int,
    scenario_name: str,
    split: SplitSet,
    feature_columns: List[str],
    automata_results_path: Path,
    dl_results_path: Path,
    logger,
) -> None:
                                                  
    preprocessor_for_dl = fit_preprocessor(
        train_df=split.x_train,
        feature_columns=feature_columns,
        cfg=cfg.preprocessing,
        use_pca=False,
    )
    preprocessor_for_automata = fit_preprocessor(
        train_df=split.x_train,
        feature_columns=feature_columns,
        cfg=cfg.preprocessing,
        use_pca=True,
    )

    x_train_dl = preprocessor_for_dl.transform(split.x_train)
    x_val_dl = preprocessor_for_dl.transform(split.x_val)
    x_test_dl = preprocessor_for_dl.transform(split.x_test)

    x_train_auto = preprocessor_for_automata.transform(split.x_train).ravel()
    x_test_auto = preprocessor_for_automata.transform(split.x_test).ravel()

                                                                          
                               
    x_test_dl_scenario = _apply_scenario_to_array(
        x_test_dl, split.y_test.to_numpy(), scenario_name, cfg.experiment, seed
    )
    x_test_auto_scenario = _apply_scenario_to_array(
        x_test_auto.reshape(-1, 1), split.y_test.to_numpy(), scenario_name, cfg.experiment, seed
    ).ravel()

                          
    grid = _automata_param_grid(
        cfg.automata, cfg.sweep.window_sizes, cfg.sweep.alphabet_sizes
    )
    for auto_cfg in grid:
        automaton, train_time = fit_automaton([x_train_auto], auto_cfg)
        result = evaluate_automaton(
            automaton=automaton,
            test_series=x_test_auto_scenario,
            test_labels=split.y_test.to_numpy(),
        )
        result.train_time_sec = train_time
        roc_pr = (
            roc_pr_summary(result.y_true_aligned, result.anomaly_score)
            if result.anomaly_score is not None and result.y_true_aligned is not None
            else {"available": False}
        )
        payload = {
            "dataset": "batadal",
            "scenario": scenario_name,
            "seed": seed,
            "model": "automata",
            "window_size": auto_cfg.window_size,
            "alphabet_size": auto_cfg.alphabet_size,
            "metrics": result.metrics.to_dict(),
            "confusion_matrix": result.confusion_matrix.tolist(),
            "train_time_sec": result.train_time_sec,
            "inference_time_sec": result.inference_time_sec,
            "roc_auc": roc_pr.get("roc_auc") if roc_pr.get("available") else None,
            "pr_auc": roc_pr.get("pr_auc") if roc_pr.get("available") else None,
            **_automata_metrics_payload(result),
        }
        _record(payload, automata_results_path)

                                                                             
                                                           
        if (
            auto_cfg.window_size == cfg.automata.window_size
            and auto_cfg.alphabet_size == cfg.automata.alphabet_size
        ):
            save_explanations_jsonl(
                automaton.explain_sequence(x_test_auto_scenario),
                cfg.paths.explanations_dir
                / f"batadal_{scenario_name}_seed{seed}.jsonl",
            )
            transition_matrix(automaton).to_csv(
                cfg.paths.results_dir
                / f"batadal_{scenario_name}_seed{seed}_transitions.csv"
            )
            plot_confusion_matrix(
                cm=result.confusion_matrix,
                title=f"BATADAL automata — {scenario_name} (seed {seed})",
                output_path=cfg.paths.figures_dir
                / f"cm_batadal_automata_{scenario_name}_seed{seed}.png",
            )
            if roc_pr.get("available"):
                _save_roc_pr_curve(
                    roc_pr=roc_pr,
                    title=f"BATADAL automata — {scenario_name} seed {seed}",
                    output_path_roc=cfg.paths.figures_dir
                    / f"roc_batadal_automata_{scenario_name}_seed{seed}.png",
                    output_path_pr=cfg.paths.figures_dir
                    / f"pr_batadal_automata_{scenario_name}_seed{seed}.png",
                )
            if result.y_true_aligned is not None and result.y_pred_aligned is not None:
                _cache_predictions(
                    cfg.paths.results_dir,
                    dataset="batadal",
                    scenario=scenario_name,
                    seed=seed,
                    model="automata",
                    y_true=result.y_true_aligned,
                    y_pred=result.y_pred_aligned,
                )
            _write_counterfactual_sample(
                automaton=automaton,
                series=x_test_auto_scenario,
                output_path=cfg.paths.explanations_dir
                / f"counterfactual_batadal_{scenario_name}_seed{seed}.json",
            )

                               
    if not torch_available():
        logger.warning("PyTorch not available; skipping DL models for BATADAL.")
        return

    seq_x_train, seq_y_train = build_sliding_sequences(
        x_train_dl, split.y_train.to_numpy(), cfg.deep_learning.sequence_length
    )
    seq_x_val, seq_y_val = build_sliding_sequences(
        x_val_dl, split.y_val.to_numpy(), cfg.deep_learning.sequence_length
    )
    seq_x_test, seq_y_test = build_sliding_sequences(
        x_test_dl_scenario, split.y_test.to_numpy(), cfg.deep_learning.sequence_length
    )

    if seq_x_train.size == 0 or seq_x_val.size == 0 or seq_x_test.size == 0:
        logger.warning("BATADAL sequences too short; skipping DL run for this seed/scenario.")
        return

    for model_name in cfg.deep_learning.models:
        classifier, history = fit_dl_model(
            model_name=model_name,
            seed=seed,
            x_train=seq_x_train,
            y_train=seq_y_train,
            x_val=seq_x_val,
            y_val=seq_y_val,
            dl_cfg=cfg.deep_learning,
            train_cfg=cfg.training,
        )
        y_pred = classifier.predict(seq_x_test)
        proba = classifier.predict_proba(seq_x_test)
        anomaly_score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else None
        metrics = compute_classification_metrics(seq_y_test, y_pred)
        cm = confusion(seq_y_test, y_pred)
        roc_pr = (
            roc_pr_summary(seq_y_test, anomaly_score)
            if anomaly_score is not None
            else {"available": False}
        )
        payload = {
            "dataset": "batadal",
            "scenario": scenario_name,
            "seed": seed,
            "model": model_name,
            "metrics": metrics.to_dict(),
            "confusion_matrix": cm.tolist(),
            "best_epoch": history.best_epoch,
            "best_val_loss": history.best_val_loss,
            "train_time_sec": history.train_time_sec,
            "inference_time_sec": history.inference_time_sec,
            "roc_auc": roc_pr.get("roc_auc") if roc_pr.get("available") else None,
            "pr_auc": roc_pr.get("pr_auc") if roc_pr.get("available") else None,
        }
        _record(payload, dl_results_path)
        plot_confusion_matrix(
            cm=cm,
            title=f"BATADAL {model_name} — {scenario_name} (seed {seed})",
            output_path=cfg.paths.figures_dir
            / f"cm_batadal_{model_name}_{scenario_name}_seed{seed}.png",
        )
        if roc_pr.get("available"):
            _save_roc_pr_curve(
                roc_pr=roc_pr,
                title=f"BATADAL {model_name} — {scenario_name} seed {seed}",
                output_path_roc=cfg.paths.figures_dir
                / f"roc_batadal_{model_name}_{scenario_name}_seed{seed}.png",
                output_path_pr=cfg.paths.figures_dir
                / f"pr_batadal_{model_name}_{scenario_name}_seed{seed}.png",
            )
                                                                     
        _cache_predictions(
            cfg.paths.results_dir,
            dataset="batadal",
            scenario=scenario_name,
            seed=seed,
            model=model_name,
            y_true=seq_y_test,
            y_pred=y_pred,
        )


                                                                             
      
                                                                             


def run_skab(cfg: ProjectConfig, logger) -> None:
    df = load_skab_valves(cfg.skab)
    x_df, y = build_skab_features_target(
        df,
        datetime_col=cfg.skab.datetime_col,
        target_col=cfg.skab.target_col,
        source_group_col=cfg.skab.source_group_col,
        source_file_col=cfg.skab.source_file_col,
        changepoint_col=cfg.skab.changepoint_col,
    )
    feature_columns = list(x_df.columns)
    groups = df[cfg.skab.source_file_col]

    automata_results_path = cfg.paths.results_dir / "skab_automata_runs.jsonl"
    dl_results_path = cfg.paths.results_dir / "skab_dl_runs.jsonl"

    fold_iter = list(
        group_kfold_indices(
            y=y,
            groups=groups,
            n_splits=cfg.skab.n_splits,
            use_stratified=cfg.skab.use_stratified_group_kfold,
        )
    )
    if not fold_iter:
        logger.warning("SKAB GroupKFold produced no folds.")
        return

    logger.info("SKAB: %d folds discovered.", len(fold_iter))

    for seed in cfg.training.random_seeds:
        set_global_seed(seed)
        for fold_idx, (train_idx, test_idx) in enumerate(fold_iter):
            train_inner_idx, val_idx = carve_validation_from_train(
                train_idx=train_idx,
                groups=groups,
                val_ratio=0.2,
                seed=seed + fold_idx,
            )
            split = SplitSet(
                x_train=x_df.iloc[train_inner_idx].reset_index(drop=True),
                y_train=y.iloc[train_inner_idx].reset_index(drop=True),
                x_val=x_df.iloc[val_idx].reset_index(drop=True),
                y_val=y.iloc[val_idx].reset_index(drop=True),
                x_test=x_df.iloc[test_idx].reset_index(drop=True),
                y_test=y.iloc[test_idx].reset_index(drop=True),
            )

            for scenario_name in cfg.experiment.scenarios:
                _run_skab_fold_scenario(
                    cfg=cfg,
                    seed=seed,
                    fold_idx=fold_idx,
                    scenario_name=scenario_name,
                    split=split,
                    train_groups=groups.iloc[train_inner_idx].reset_index(drop=True),
                    feature_columns=feature_columns,
                    automata_results_path=automata_results_path,
                    dl_results_path=dl_results_path,
                    logger=logger,
                )


def _run_skab_fold_scenario(
    *,
    cfg: ProjectConfig,
    seed: int,
    fold_idx: int,
    scenario_name: str,
    split: SplitSet,
    train_groups: pd.Series,
    feature_columns: List[str],
    automata_results_path: Path,
    dl_results_path: Path,
    logger,
) -> None:
    preprocessor_for_dl = fit_preprocessor(
        train_df=split.x_train,
        feature_columns=feature_columns,
        cfg=cfg.preprocessing,
        use_pca=False,
    )
    preprocessor_for_automata = fit_preprocessor(
        train_df=split.x_train,
        feature_columns=feature_columns,
        cfg=cfg.preprocessing,
        use_pca=True,
    )

    x_train_dl = preprocessor_for_dl.transform(split.x_train)
    x_val_dl = preprocessor_for_dl.transform(split.x_val)
    x_test_dl = preprocessor_for_dl.transform(split.x_test)

    x_train_auto = preprocessor_for_automata.transform(split.x_train).ravel()
    x_test_auto = preprocessor_for_automata.transform(split.x_test).ravel()

    x_test_dl_scenario = _apply_scenario_to_array(
        x_test_dl, split.y_test.to_numpy(), scenario_name, cfg.experiment, seed
    )
    x_test_auto_scenario = _apply_scenario_to_array(
        x_test_auto.reshape(-1, 1), split.y_test.to_numpy(), scenario_name, cfg.experiment, seed
    ).ravel()

                                                                        
                                                        
    train_groups_index: dict[str, list[int]] = {}
    for i, g in enumerate(train_groups):
        train_groups_index.setdefault(g, []).append(i)
    grouped_train_series = [x_train_auto[idxs] for idxs in train_groups_index.values()]

    grid = _automata_param_grid(
        cfg.automata, cfg.sweep.window_sizes, cfg.sweep.alphabet_sizes
    )
    for auto_cfg in grid:
        automaton, train_time = fit_automaton(grouped_train_series, auto_cfg)
        result = evaluate_automaton(
            automaton=automaton,
            test_series=x_test_auto_scenario,
            test_labels=split.y_test.to_numpy(),
        )
        result.train_time_sec = train_time
        roc_pr = (
            roc_pr_summary(result.y_true_aligned, result.anomaly_score)
            if result.anomaly_score is not None and result.y_true_aligned is not None
            else {"available": False}
        )
        payload = {
            "dataset": "skab",
            "scenario": scenario_name,
            "seed": seed,
            "fold": fold_idx,
            "model": "automata",
            "window_size": auto_cfg.window_size,
            "alphabet_size": auto_cfg.alphabet_size,
            "metrics": result.metrics.to_dict(),
            "confusion_matrix": result.confusion_matrix.tolist(),
            "train_time_sec": result.train_time_sec,
            "inference_time_sec": result.inference_time_sec,
            "roc_auc": roc_pr.get("roc_auc") if roc_pr.get("available") else None,
            "pr_auc": roc_pr.get("pr_auc") if roc_pr.get("available") else None,
            **_automata_metrics_payload(result),
        }
        _record(payload, automata_results_path)

        if (
            auto_cfg.window_size == cfg.automata.window_size
            and auto_cfg.alphabet_size == cfg.automata.alphabet_size
        ):
            save_explanations_jsonl(
                automaton.explain_sequence(x_test_auto_scenario),
                cfg.paths.explanations_dir
                / f"skab_fold{fold_idx}_{scenario_name}_seed{seed}.jsonl",
            )
            transition_matrix(automaton).to_csv(
                cfg.paths.results_dir
                / f"skab_fold{fold_idx}_{scenario_name}_seed{seed}_transitions.csv"
            )
            if fold_idx == 0:
                plot_confusion_matrix(
                    cm=result.confusion_matrix,
                    title=f"SKAB automata — {scenario_name} fold0 seed{seed}",
                    output_path=cfg.paths.figures_dir
                    / f"cm_skab_automata_{scenario_name}_seed{seed}.png",
                )
                if roc_pr.get("available"):
                    _save_roc_pr_curve(
                        roc_pr=roc_pr,
                        title=f"SKAB automata — {scenario_name} fold0 seed{seed}",
                        output_path_roc=cfg.paths.figures_dir
                        / f"roc_skab_automata_{scenario_name}_seed{seed}.png",
                        output_path_pr=cfg.paths.figures_dir
                        / f"pr_skab_automata_{scenario_name}_seed{seed}.png",
                    )
            if result.y_true_aligned is not None and result.y_pred_aligned is not None:
                _cache_predictions(
                    cfg.paths.results_dir,
                    dataset="skab",
                    scenario=scenario_name,
                    seed=seed,
                    fold=fold_idx,
                    model="automata",
                    y_true=result.y_true_aligned,
                    y_pred=result.y_pred_aligned,
                )
            if fold_idx == 0:
                _write_counterfactual_sample(
                    automaton=automaton,
                    series=x_test_auto_scenario,
                    output_path=cfg.paths.explanations_dir
                    / f"counterfactual_skab_{scenario_name}_seed{seed}.json",
                )

    if not torch_available():
        return

    seq_x_train, seq_y_train = build_sliding_sequences(
        x_train_dl, split.y_train.to_numpy(), cfg.deep_learning.sequence_length
    )
    seq_x_val, seq_y_val = build_sliding_sequences(
        x_val_dl, split.y_val.to_numpy(), cfg.deep_learning.sequence_length
    )
    seq_x_test, seq_y_test = build_sliding_sequences(
        x_test_dl_scenario, split.y_test.to_numpy(), cfg.deep_learning.sequence_length
    )

    if seq_x_train.size == 0 or seq_x_val.size == 0 or seq_x_test.size == 0:
        logger.warning(
            "SKAB sequences too short for fold=%d seed=%d scenario=%s; skipping DL.",
            fold_idx,
            seed,
            scenario_name,
        )
        return

    for model_name in cfg.deep_learning.models:
        classifier, history = fit_dl_model(
            model_name=model_name,
            seed=seed,
            x_train=seq_x_train,
            y_train=seq_y_train,
            x_val=seq_x_val,
            y_val=seq_y_val,
            dl_cfg=cfg.deep_learning,
            train_cfg=cfg.training,
        )
        y_pred = classifier.predict(seq_x_test)
        proba = classifier.predict_proba(seq_x_test)
        anomaly_score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else None
        metrics = compute_classification_metrics(seq_y_test, y_pred)
        cm = confusion(seq_y_test, y_pred)
        roc_pr = (
            roc_pr_summary(seq_y_test, anomaly_score)
            if anomaly_score is not None
            else {"available": False}
        )
        payload = {
            "dataset": "skab",
            "scenario": scenario_name,
            "seed": seed,
            "fold": fold_idx,
            "model": model_name,
            "metrics": metrics.to_dict(),
            "confusion_matrix": cm.tolist(),
            "best_epoch": history.best_epoch,
            "best_val_loss": history.best_val_loss,
            "train_time_sec": history.train_time_sec,
            "inference_time_sec": history.inference_time_sec,
            "roc_auc": roc_pr.get("roc_auc") if roc_pr.get("available") else None,
            "pr_auc": roc_pr.get("pr_auc") if roc_pr.get("available") else None,
        }
        _record(payload, dl_results_path)
        if fold_idx == 0:
            plot_confusion_matrix(
                cm=cm,
                title=f"SKAB {model_name} — {scenario_name} fold0 seed{seed}",
                output_path=cfg.paths.figures_dir
                / f"cm_skab_{model_name}_{scenario_name}_seed{seed}.png",
            )
            if roc_pr.get("available"):
                _save_roc_pr_curve(
                    roc_pr=roc_pr,
                    title=f"SKAB {model_name} — {scenario_name} fold0 seed{seed}",
                    output_path_roc=cfg.paths.figures_dir
                    / f"roc_skab_{model_name}_{scenario_name}_seed{seed}.png",
                    output_path_pr=cfg.paths.figures_dir
                    / f"pr_skab_{model_name}_{scenario_name}_seed{seed}.png",
                )
        _cache_predictions(
            cfg.paths.results_dir,
            dataset="skab",
            scenario=scenario_name,
            seed=seed,
            fold=fold_idx,
            model=model_name,
            y_true=seq_y_test,
            y_pred=y_pred,
        )


def _apply_scenario_to_array(
    x: np.ndarray,
    y: np.ndarray,
    scenario_name: str,
    exp_cfg: ExperimentConfig,
    seed: int,
) -> np.ndarray:

    return _scenario_for(scenario_name, x, y, exp_cfg, seed).x


def _automata_metrics_payload(result: AutomataRunResult) -> dict:

    return {
        "n_states": result.n_states,
        "sax_dictionary_size": result.sax_dictionary_size,
        "n_transition_edges": result.n_transition_edges,
        "transition_density": result.transition_density,
        "n_unique_test_patterns": result.n_unique_test_patterns,
        "n_unseen_test_patterns": result.n_unseen_test_patterns,
        "detection_rate": result.detection_rate,
        "mapping_accuracy": result.mapping_accuracy,
        "avg_nearest_distance_unseen": result.avg_nearest_distance_unseen,
        "explanations_summary": result.explanations_summary,
    }


                                                                             
                     
                                                                             


def run_experiments(cfg: ProjectConfig) -> None:

    cfg.paths.ensure()
    logger = get_logger(name="yazlab2.experiments", logs_dir=cfg.paths.logs_dir)
    write_json(cfg.paths.logs_dir / "config_snapshot.json", _serialize_config(cfg))

    try:
        run_batadal(cfg, logger)
    except FileNotFoundError as exc:
        logger.warning("BATADAL data not available, skipping: %s", exc)
    except Exception:                                                
        logger.exception("BATADAL pipeline failed")
        raise

    try:
        run_skab(cfg, logger)
    except FileNotFoundError as exc:
        logger.warning("SKAB data not available, skipping: %s", exc)
    except Exception:                                                
        logger.exception("SKAB pipeline failed")
        raise

    try:
        run_cross_dataset(cfg, logger)
    except Exception:
        logger.warning("Cross-dataset pipeline failed", exc_info=True)

    logger.info("Aggregating per-scenario summaries.")
    aggregate_summaries(cfg.paths)
    run_statistical_tests(cfg.paths, logger)

    try:
        from src.evaluation.gallery import build_figure_gallery
        from src.evaluation.visualization import render_all_figures

        render_all_figures(
            transitions_csv_dir=cfg.paths.results_dir,
            figures_dir=cfg.paths.figures_dir,
            automata_summary_paths=[
                cfg.paths.results_dir / "batadal_automata_summary.csv",
                cfg.paths.results_dir / "skab_automata_summary.csv",
            ],
        )
        gallery = build_figure_gallery(cfg.paths.figures_dir)
        if gallery is not None:
            logger.info("Figure gallery: %s", gallery)
    except Exception:
        logger.warning("Figure rendering failed; continuing without plots.", exc_info=True)

    try:
        from src.experiments.report import build_markdown_report

        build_markdown_report(cfg.paths)
    except Exception:
        logger.warning("Markdown report build failed", exc_info=True)

    try:
        from src.evaluation.dashboard import build_results_dashboard

        dashboard = build_results_dashboard(
            report_md=cfg.paths.results_dir / "experiment_report.md",
            figures_dir=cfg.paths.figures_dir,
            output_path=cfg.paths.artifacts_dir / "dashboard.html",
            gallery_path=cfg.paths.figures_dir / "gallery.html",
        )
        logger.info("Full results HTML: %s", dashboard)
    except Exception:
        logger.warning("Results dashboard build failed", exc_info=True)


def aggregate_summaries(paths: PathsConfig) -> None:

    summary_payload = {}
    for source_name, jsonl in [
        ("batadal_automata", paths.results_dir / "batadal_automata_runs.jsonl"),
        ("batadal_dl", paths.results_dir / "batadal_dl_runs.jsonl"),
        ("skab_automata", paths.results_dir / "skab_automata_runs.jsonl"),
        ("skab_dl", paths.results_dir / "skab_dl_runs.jsonl"),
    ]:
        if not jsonl.exists():
            continue
        df = pd.read_json(jsonl, lines=True)
        if df.empty:
            continue
        metric_df = pd.json_normalize(df["metrics"])
        df = pd.concat([df.drop(columns=["metrics"]), metric_df], axis=1)
        group_cols = [c for c in ("scenario", "model", "window_size", "alphabet_size") if c in df.columns]
        if group_cols:
            agg_dict = {
                "accuracy_mean": ("accuracy", "mean"),
                "accuracy_std": ("accuracy", "std"),
                "precision_mean": ("precision", "mean"),
                "precision_std": ("precision", "std"),
                "recall_mean": ("recall", "mean"),
                "recall_std": ("recall", "std"),
                "f1_mean": ("f1", "mean"),
                "f1_std": ("f1", "std"),
                "runs": ("f1", "count"),
            }
            for opt_col in (
                "detection_rate",
                "mapping_accuracy",
                "avg_nearest_distance_unseen",
                "train_time_sec",
                "inference_time_sec",
                "roc_auc",
                "pr_auc",
                "n_states",
                "sax_dictionary_size",
                "n_transition_edges",
                "transition_density",
                "n_unseen_test_patterns",
            ):
                if opt_col in df.columns:
                    agg_dict[f"{opt_col}_mean"] = (opt_col, "mean")
                    agg_dict[f"{opt_col}_std"] = (opt_col, "std")
            grouped = df.groupby(group_cols).agg(**agg_dict)
            summary_path = paths.results_dir / f"{source_name}_summary.csv"
            grouped.to_csv(summary_path)
            summary_payload[source_name] = grouped.reset_index().to_dict(orient="records")
    if summary_payload:
        write_json(paths.results_dir / "experiment_summary.json", summary_payload)


                                                                             
                          
                                                                             


def run_cross_dataset(cfg: ProjectConfig, logger) -> None:

    cross_path = cfg.paths.results_dir / "cross_dataset_runs.jsonl"

    try:
        batadal_df = load_batadal_training_dataset2(cfg.batadal)
    except FileNotFoundError:
        logger.warning("Cross-dataset: BATADAL not available; skipping.")
        return
    try:
        skab_df = load_skab_valves(cfg.skab)
    except (ValueError, FileNotFoundError):
        logger.warning("Cross-dataset: SKAB not available; skipping.")
        return

    bx, by = build_batadal_features_target(
        batadal_df, cfg.batadal.datetime_col, cfg.batadal.target_col
    )
    sx, sy = build_skab_features_target(
        skab_df,
        datetime_col=cfg.skab.datetime_col,
        target_col=cfg.skab.target_col,
        source_group_col=cfg.skab.source_group_col,
        source_file_col=cfg.skab.source_file_col,
        changepoint_col=cfg.skab.changepoint_col,
    )

    pairs = (
        ("batadal", bx, by, "skab", sx, sy),
        ("skab", sx, sy, "batadal", bx, by),
    )

    for src_name, x_src, y_src, tgt_name, x_tgt, y_tgt in pairs:
                                                                      
                                                                           
                                                                        
                                             
        prep_src = fit_preprocessor(
            train_df=x_src,
            feature_columns=list(x_src.columns),
            cfg=cfg.preprocessing,
            use_pca=True,
        )
        prep_tgt = fit_preprocessor(
            train_df=x_tgt,
            feature_columns=list(x_tgt.columns),
            cfg=cfg.preprocessing,
            use_pca=True,
        )
        train_series = prep_src.transform(x_src).ravel()
        test_series = prep_tgt.transform(x_tgt).ravel()

        for scenario in cfg.experiment.scenarios:
            test_perturbed = _apply_scenario_to_array(
                test_series.reshape(-1, 1),
                y_tgt.to_numpy(),
                scenario,
                cfg.experiment,
                seed=cfg.training.random_seeds[0],
            ).ravel()

            automaton, train_time = fit_automaton([train_series], cfg.automata)
            result = evaluate_automaton(
                automaton=automaton,
                test_series=test_perturbed,
                test_labels=y_tgt.to_numpy(),
            )
            result.train_time_sec = train_time
            roc_pr = (
                roc_pr_summary(result.y_true_aligned, result.anomaly_score)
                if result.anomaly_score is not None and result.y_true_aligned is not None
                else {"available": False}
            )
            payload = {
                "train_dataset": src_name,
                "test_dataset": tgt_name,
                "scenario": scenario,
                "model": "automata",
                "metrics": result.metrics.to_dict(),
                "detection_rate": result.detection_rate,
                "mapping_accuracy": result.mapping_accuracy,
                "train_time_sec": result.train_time_sec,
                "inference_time_sec": result.inference_time_sec,
                "roc_auc": roc_pr.get("roc_auc") if roc_pr.get("available") else None,
                "pr_auc": roc_pr.get("pr_auc") if roc_pr.get("available") else None,
            }
            _record(payload, cross_path)
            logger.info(
                "Cross-dataset %s -> %s [%s]: F1=%.3f acc=%.3f",
                src_name,
                tgt_name,
                scenario,
                result.metrics.f1,
                result.metrics.accuracy,
            )


                                                                             
                   
                                                                             


def run_statistical_tests(paths: PathsConfig, logger) -> None:

    cache_dir = paths.results_dir / "predictions"
    if not cache_dir.exists():
        return

    files = sorted(cache_dir.glob("*.npz"))
    if not files:
        return

                                                                            
                 
    records: list[dict] = []
    for p in files:
        stem = p.stem.split("_")
                                                                         
        dataset = stem[0]
        model = stem[1]
        rest = stem[2:]
        fold_idx = None
        for token in rest:
            if token.startswith("fold"):
                fold_idx = int(token.replace("fold", ""))
        scenario = rest[0]
        seed = int([t for t in rest if t.startswith("seed")][0].replace("seed", ""))
        data = np.load(p)
        records.append(
            {
                "dataset": dataset,
                "model": model,
                "scenario": scenario,
                "seed": seed,
                "fold": fold_idx,
                "y_true": data["y_true"],
                "y_pred": data["y_pred"],
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return

    test_rows: list[dict] = []
    grouping = ["dataset", "scenario"]
    for keys, group in df.groupby(grouping):
        models = sorted(group["model"].unique())
        if len(models) < 2:
            continue
        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                a_runs = group[group["model"] == model_a]
                b_runs = group[group["model"] == model_b]
                                                  
                join_cols = [c for c in ("seed", "fold") if c in a_runs.columns]
                merged = a_runs.merge(
                    b_runs, on=join_cols, suffixes=("_a", "_b"), how="inner"
                )
                if merged.empty:
                    continue
                f1_a: list[float] = []
                f1_b: list[float] = []
                pooled_y_true: list[np.ndarray] = []
                pooled_pred_a: list[np.ndarray] = []
                pooled_pred_b: list[np.ndarray] = []
                for _, row in merged.iterrows():
                    f1_a.append(_safe_f1(row["y_true_a"], row["y_pred_a"]))
                    f1_b.append(_safe_f1(row["y_true_b"], row["y_pred_b"]))
                                                                         
                                                                         
                                                                     
                                                                          
                    if np.array_equal(row["y_true_a"], row["y_true_b"]):
                        pooled_y_true.append(row["y_true_a"])
                        pooled_pred_a.append(row["y_pred_a"])
                        pooled_pred_b.append(row["y_pred_b"])
                if not f1_a:
                    continue
                wilcoxon_res = wilcoxon_signed_rank(f1_a, f1_b)
                if pooled_y_true:
                    mcnemar_res = mcnemar_test(
                        np.concatenate(pooled_y_true),
                        np.concatenate(pooled_pred_a),
                        np.concatenate(pooled_pred_b),
                    )
                else:
                    mcnemar_res = {
                        "available": False,
                        "reason": "incompatible window/sequence alignment",
                    }
                test_rows.append(
                    {
                        "dataset": keys[0],
                        "scenario": keys[1],
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_paired_runs": len(f1_a),
                        "f1_a_mean": float(np.mean(f1_a)),
                        "f1_b_mean": float(np.mean(f1_b)),
                        "wilcoxon_p": wilcoxon_res.get("p_value"),
                        "wilcoxon_stat": wilcoxon_res.get("statistic"),
                        "mcnemar_p": mcnemar_res.get("p_value"),
                        "mcnemar_stat": mcnemar_res.get("statistic"),
                    }
                )

    if test_rows:
        out_path = paths.results_dir / "statistical_tests.csv"
        pd.DataFrame(test_rows).to_csv(out_path, index=False)
        logger.info("Wrote statistical test summary to %s", out_path)


def _safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(compute_classification_metrics(y_true, y_pred).f1)


def _serialize_config(cfg: ProjectConfig) -> dict:

    def _normalize(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (list, tuple)):
            return [_normalize(o) for o in obj]
        if isinstance(obj, dict):
            return {k: _normalize(v) for k, v in obj.items()}
        return obj

    return _normalize(asdict(cfg))


def run_bootstrap_pipeline(cfg: ProjectConfig) -> None:

    run_experiments(cfg)
