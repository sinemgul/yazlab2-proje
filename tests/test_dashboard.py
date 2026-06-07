from __future__ import annotations

from pathlib import Path

from src.evaluation.dashboard import build_results_dashboard, parse_report_sections


def test_parse_report_sections_order() -> None:
    md = "## Tablo 1: A\n| x |\n| --- |\n| 1 |\n\n## Tablo 2: B\n| y |\n| --- |\n| 2 |\n"
    sections = parse_report_sections(md)
    assert len(sections) == 2
    assert sections[0]["rows"] == [["1"]]
    assert sections[1]["rows"] == [["2"]]


def test_build_full_results_page(tmp_path: Path) -> None:
    report = tmp_path / "results" / "experiment_report.md"
    report.parent.mkdir(parents=True)
    report.write_text(
        "## Tablo 1: F1\n| Model | F1 |\n| --- | --- |\n| gru | 0.9 |\n",
        encoding="utf-8",
    )
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "cm_a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (figures / "roc_b.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    out = tmp_path / "dashboard.html"
    build_results_dashboard(report, figures, out)
    text = out.read_text(encoding="utf-8")
    assert "Tam Deney Sonuçları" in text
    assert "Tüm figürler" in text
    assert "cm_a.png" in text
    assert "roc_b.png" in text
    assert "gru" in text
    assert "0.9" in text
