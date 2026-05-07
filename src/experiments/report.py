"""Build the Markdown experiment report from the persisted JSONL/CSV results.

The output mirrors the supplementary template (Tablo 1-5) so the user can
paste the generated report into the project's main report:

* Tablo 1 - Model F1 ± std per dataset
* Tablo 2 - Noise effect + Detection Rate + Mapping Accuracy
* Tablo 3 - Cross-dataset performance
* Tablo 4 - Automata window/alphabet sensitivity
* Tablo 5 - Training / Inference time per model
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from src.config import PathsConfig


def _load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_json(path, lines=True)
    if df.empty:
        return df
    if "metrics" in df.columns:
        metrics_df = pd.json_normalize(df["metrics"])
        df = pd.concat([df.drop(columns=["metrics"]), metrics_df], axis=1)
    return df


def _format_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no data)_"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    return "\n".join(lines)


def _scope(df: pd.DataFrame, scenario: str = "original") -> pd.DataFrame:
    if df.empty or "scenario" not in df.columns:
        return df
    return df[df["scenario"] == scenario]


def build_table1_performance(
    batadal_auto: pd.DataFrame,
    batadal_dl: pd.DataFrame,
    skab_auto: pd.DataFrame,
    skab_dl: pd.DataFrame,
) -> pd.DataFrame:
    """Tablo 1 - F1 mean ± std per dataset / model on the original scenario."""

    rows: list[dict] = []

    def _agg_f1(df: pd.DataFrame, model_label: str) -> Optional[tuple[str, str]]:
        df_scope = _scope(df, "original")
        if df_scope.empty or "f1" not in df_scope.columns:
            return None
        if "window_size" in df_scope.columns and "alphabet_size" in df_scope.columns:
            df_scope = df_scope[
                (df_scope["window_size"] == 4) & (df_scope["alphabet_size"] == 3)
            ]
        if df_scope.empty:
            return None
        return model_label, _format_mean_std(df_scope["f1"].mean(), df_scope["f1"].std())

    for label, dataset_pairs in [
        ("BATADAL", [("automata", batadal_auto)] + _dl_pairs(batadal_dl)),
        ("SKAB", [("automata", skab_auto)] + _dl_pairs(skab_dl)),
    ]:
        for model_label, df in dataset_pairs:
            agg = _agg_f1(df, model_label)
            if agg is None:
                continue
            rows.append({"Model": agg[0], "Dataset": label, "F1 ± std": agg[1]})

    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    return table.pivot(index="Model", columns="Dataset", values="F1 ± std").reset_index()


def _dl_pairs(dl_df: pd.DataFrame) -> List[tuple[str, pd.DataFrame]]:
    if dl_df.empty or "model" not in dl_df.columns:
        return []
    return [(model, dl_df[dl_df["model"] == model]) for model in sorted(dl_df["model"].unique())]


def build_table2_noise_unseen(
    batadal_auto: pd.DataFrame,
    batadal_dl: pd.DataFrame,
    skab_auto: pd.DataFrame,
    skab_dl: pd.DataFrame,
) -> pd.DataFrame:
    """Tablo 2 - Original F1 / Noisy F1 / Detection Rate / Mapping Accuracy."""

    rows: list[dict] = []

    def _scope_models(df: pd.DataFrame, model_label: str, dataset_label: str) -> Iterable[dict]:
        if df.empty:
            return []
        if "window_size" in df.columns and "alphabet_size" in df.columns:
            df = df[(df["window_size"] == 4) & (df["alphabet_size"] == 3)]
        if df.empty:
            return []

        def _f1_for(scenario: str) -> str:
            scope = df[df["scenario"] == scenario]
            if scope.empty:
                return "-"
            return _format_mean_std(scope["f1"].mean(), scope["f1"].std())

        unseen = df[df["scenario"] == "unseen"]
        det_rate = (
            _format_mean_std(unseen["detection_rate"].mean(), unseen["detection_rate"].std())
            if not unseen.empty and "detection_rate" in unseen.columns
            else "-"
        )
        map_acc = (
            _format_mean_std(unseen["mapping_accuracy"].mean(), unseen["mapping_accuracy"].std())
            if not unseen.empty and "mapping_accuracy" in unseen.columns
            else "-"
        )
        return [
            {
                "Dataset": dataset_label,
                "Model": model_label,
                "Original F1": _f1_for("original"),
                "Noisy F1": _f1_for("noise"),
                "Detection Rate (unseen)": det_rate,
                "Mapping Accuracy (unseen)": map_acc,
            }
        ]

    for dataset_label, auto_df, dl_df in [
        ("BATADAL", batadal_auto, batadal_dl),
        ("SKAB", skab_auto, skab_dl),
    ]:
        rows.extend(_scope_models(auto_df, "automata", dataset_label))
        for model_label, df in _dl_pairs(dl_df):
            rows.extend(_scope_models(df, model_label, dataset_label))

    return pd.DataFrame(rows)


def build_table3_cross_dataset(cross_df: pd.DataFrame) -> pd.DataFrame:
    """Tablo 3 - Cross-dataset transfer F1 (automaton)."""

    if cross_df.empty:
        return pd.DataFrame()
    df = cross_df.copy()
    if "metrics" in df.columns:
        metrics_df = pd.json_normalize(df["metrics"])
        df = pd.concat([df.drop(columns=["metrics"]), metrics_df], axis=1)
    df = df[df.get("scenario") == "original"]
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index="train_dataset", columns="test_dataset", values="f1", aggfunc="mean"
    )
    pivot = pivot.map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    pivot = pivot.reset_index().rename(columns={"train_dataset": "Train / Test"})
    return pivot


def build_table4_param_sensitivity(
    batadal_auto: pd.DataFrame, skab_auto: pd.DataFrame
) -> pd.DataFrame:
    """Tablo 4 - Window/Alphabet sensitivity F1 means averaged over datasets."""

    rows: list[dict] = []
    for dataset_label, df in (("BATADAL", batadal_auto), ("SKAB", skab_auto)):
        df_scope = _scope(df, "original")
        if df_scope.empty or "window_size" not in df_scope.columns:
            continue
        for value in [3, 4, 5, 6]:
            window_scope = df_scope[
                (df_scope["window_size"] == value) & (df_scope["alphabet_size"] == 3)
            ]
            alpha_scope = df_scope[
                (df_scope["alphabet_size"] == value) & (df_scope["window_size"] == 4)
            ]
            rows.append(
                {
                    "Dataset": dataset_label,
                    "Parameter": "Window Size",
                    "Value": value,
                    "F1": _format_mean_std(window_scope["f1"].mean(), window_scope["f1"].std())
                    if not window_scope.empty
                    else "-",
                }
            )
            rows.append(
                {
                    "Dataset": dataset_label,
                    "Parameter": "Alphabet Size",
                    "Value": value,
                    "F1": _format_mean_std(alpha_scope["f1"].mean(), alpha_scope["f1"].std())
                    if not alpha_scope.empty
                    else "-",
                }
            )
    return pd.DataFrame(rows)


def build_table5_runtime(
    batadal_auto: pd.DataFrame,
    batadal_dl: pd.DataFrame,
    skab_auto: pd.DataFrame,
    skab_dl: pd.DataFrame,
) -> pd.DataFrame:
    """Tablo 5 - Runtime (training + inference seconds) averaged across runs."""

    rows: list[dict] = []
    sources = [
        ("BATADAL", "automata", batadal_auto),
        ("SKAB", "automata", skab_auto),
    ]
    for dataset_label, dl_df in [("BATADAL", batadal_dl), ("SKAB", skab_dl)]:
        for model, df in _dl_pairs(dl_df):
            sources.append((dataset_label, model, df))
    for dataset_label, model_label, df in sources:
        if df.empty:
            continue
        df_scope = _scope(df, "original")
        if df_scope.empty:
            continue
        if "window_size" in df_scope.columns and "alphabet_size" in df_scope.columns:
            df_scope = df_scope[
                (df_scope["window_size"] == 4) & (df_scope["alphabet_size"] == 3)
            ]
        if df_scope.empty:
            continue
        rows.append(
            {
                "Dataset": dataset_label,
                "Model": model_label,
                "Train Time (s)": _format_mean_std(
                    df_scope.get("train_time_sec", pd.Series([np.nan])).mean(),
                    df_scope.get("train_time_sec", pd.Series([np.nan])).std(),
                ),
                "Inference Time (s)": _format_mean_std(
                    df_scope.get("inference_time_sec", pd.Series([np.nan])).mean(),
                    df_scope.get("inference_time_sec", pd.Series([np.nan])).std(),
                ),
            }
        )
    return pd.DataFrame(rows)


def build_markdown_report(paths: PathsConfig) -> Path:
    """Aggregate every persisted result into a single Markdown report."""

    batadal_auto = _load_jsonl(paths.results_dir / "batadal_automata_runs.jsonl")
    batadal_dl = _load_jsonl(paths.results_dir / "batadal_dl_runs.jsonl")
    skab_auto = _load_jsonl(paths.results_dir / "skab_automata_runs.jsonl")
    skab_dl = _load_jsonl(paths.results_dir / "skab_dl_runs.jsonl")
    cross_df = _load_jsonl(paths.results_dir / "cross_dataset_runs.jsonl")

    report_path = paths.results_dir / "experiment_report.md"
    sections: list[str] = ["# Yazlab2 - Deney Raporu", ""]

    sections.append("## Tablo 1: Model F1 ± std (Original senaryo, sabit parametreler)")
    sections.append(_df_to_markdown(build_table1_performance(batadal_auto, batadal_dl, skab_auto, skab_dl)))
    sections.append("")

    sections.append("## Tablo 2: Gürültü Etkisi ve Unseen Senaryo Analizi")
    sections.append(_df_to_markdown(build_table2_noise_unseen(batadal_auto, batadal_dl, skab_auto, skab_dl)))
    sections.append("")

    sections.append("## Tablo 3: Cross-Dataset Performans Karşılaştırması (F1)")
    sections.append(_df_to_markdown(build_table3_cross_dataset(cross_df)))
    sections.append("")

    sections.append("## Tablo 4: Automata Parametre Duyarlılık Analizi (F1)")
    sections.append(_df_to_markdown(build_table4_param_sensitivity(batadal_auto, skab_auto)))
    sections.append("")

    sections.append("## Tablo 5: Modellerin Çalışma Süresi (Training / Inference)")
    sections.append(_df_to_markdown(build_table5_runtime(batadal_auto, batadal_dl, skab_auto, skab_dl)))
    sections.append("")

    stats_path = paths.results_dir / "statistical_tests.csv"
    if stats_path.exists():
        sections.append("## İstatistiksel Test Sonuçları (Wilcoxon / McNemar)")
        sections.append(_df_to_markdown(pd.read_csv(stats_path)))
        sections.append("")

    report_path.write_text("\n".join(sections), encoding="utf-8")
    return report_path
