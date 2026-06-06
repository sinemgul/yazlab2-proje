"""Tests for the results dashboard HTML builder."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.dashboard import build_results_dashboard, markdown_report_to_html


def test_markdown_report_to_html_table() -> None:
    md = "## Tablo 1\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    out = markdown_report_to_html(md)
    assert "<table>" in out
    assert "Tablo 1" in out
    assert "1" in out


def test_build_results_dashboard(tmp_path: Path) -> None:
    report = tmp_path / "results" / "experiment_report.md"
    report.parent.mkdir(parents=True)
    report.write_text(
        "## Tablo 1\n| Model | F1 |\n| --- | --- |\n| gru | 0.9 |\n",
        encoding="utf-8",
    )
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "cm_test.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    out = tmp_path / "dashboard.html"
    build_results_dashboard(report, figures, out)
    text = out.read_text(encoding="utf-8")
    assert "Deney Sonuçları" in text
    assert "cm_test.png" in text
    assert "gru" in text
