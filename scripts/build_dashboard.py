"""Build the full HTML results page (all tables + all figures)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.dashboard import build_results_dashboard


def main() -> None:
    report = ROOT / "artifacts" / "results" / "experiment_report.md"
    figures = ROOT / "artifacts" / "figures"
    gallery = figures / "gallery.html"
    out = ROOT / "artifacts" / "dashboard.html"
    path = build_results_dashboard(
        report_md=report,
        figures_dir=figures,
        output_path=out,
        gallery_path=gallery if gallery.exists() else None,
    )
    print(f"Built {path}")


if __name__ == "__main__":
    main()
