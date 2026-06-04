"""Tests for the static figure gallery builder."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.gallery import build_figure_gallery, categorise


def test_categorise_prefixes() -> None:
    assert categorise("cm_batadal_lstm_original_seed42.png") == "Confusion Matrix"
    assert categorise("heatmap_batadal_original_seed42_transitions.png") == (
        "Transition Heatmap"
    )
    assert categorise("other_plot.png") == "Other"


def test_build_figure_gallery_writes_html(tmp_path: Path) -> None:
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "cm_test.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    out = build_figure_gallery(figures)
    assert out is not None
    assert out.name == "gallery.html"
    text = out.read_text(encoding="utf-8")
    assert "cm_test.png" in text


def test_build_figure_gallery_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert build_figure_gallery(empty) is None
