"""Build a single static HTML gallery of every figure under a directory.

Usage:
    python scripts/build_gallery.py [figures_dir]

If ``figures_dir`` is omitted, both ``artifacts/figures`` and
``artifacts/smoke/figures`` are scanned. The resulting ``gallery.html`` lands
next to the figures and can be opened directly in any browser.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from html import escape
from pathlib import Path


CATEGORY_PATTERNS = {
    "Confusion Matrix": "cm_",
    "ROC": "roc_",
    "Precision-Recall": "pr_",
    "Transition Heatmap": "heatmap_",
    "State Diagram": "diagram_",
    "Parameter Sensitivity": "sensitivity_",
}


def categorise(filename: str) -> str:
    for label, prefix in CATEGORY_PATTERNS.items():
        if filename.startswith(prefix):
            return label
    return "Other"


def build_gallery(figures_dir: Path) -> Path | None:
    figures_dir = figures_dir.resolve()
    if not figures_dir.exists():
        return None

    images = sorted(p for p in figures_dir.glob("*.png"))
    if not images:
        return None

    grouped: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        grouped[categorise(image.name)].append(image)

    html_lines: list[str] = []
    html_lines.append("<!DOCTYPE html>")
    html_lines.append("<html lang='tr'><head><meta charset='utf-8'>")
    html_lines.append(f"<title>Yazlab2 figures – {escape(figures_dir.name)}</title>")
    html_lines.append(
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:0;padding:24px;background:#0f172a;color:#e2e8f0}"
        "h1{margin:0 0 8px}h2{margin-top:32px;border-bottom:1px solid #334155;padding-bottom:6px}"
        "section{margin-bottom:32px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}"
        ".card{background:#1e293b;border-radius:10px;padding:10px;box-shadow:0 1px 3px rgba(0,0,0,.2)}"
        ".card img{width:100%;height:auto;border-radius:6px;background:#fff}"
        ".card p{font-size:12px;margin:6px 0 0;word-break:break-all;color:#cbd5e1}"
        "summary{cursor:pointer;font-weight:600;font-size:14px;color:#f8fafc}"
        ".count{color:#94a3b8;font-weight:400;margin-left:8px}"
        "</style></head><body>"
    )
    html_lines.append(f"<h1>Yazlab2 — {escape(figures_dir.name)}</h1>")
    html_lines.append(
        f"<p>{sum(len(v) for v in grouped.values())} figür · "
        f"{len(grouped)} kategori · klasör: <code>{escape(str(figures_dir))}</code></p>"
    )

    for label in [*CATEGORY_PATTERNS.keys(), "Other"]:
        files = grouped.get(label)
        if not files:
            continue
        html_lines.append("<section>")
        html_lines.append(
            f"<details open><summary>{escape(label)}<span class='count'>({len(files)})</span></summary>"
        )
        html_lines.append("<div class='grid'>")
        for image in files:
            rel = image.name
            html_lines.append(
                "<div class='card'>"
                f"<a href='{escape(rel)}' target='_blank'>"
                f"<img src='{escape(rel)}' alt='{escape(rel)}'></a>"
                f"<p>{escape(rel)}</p></div>"
            )
        html_lines.append("</div></details></section>")

    html_lines.append("</body></html>")

    output = figures_dir / "gallery.html"
    output.write_text("\n".join(html_lines), encoding="utf-8")
    return output


def main() -> None:
    targets: list[Path]
    if len(sys.argv) > 1:
        targets = [Path(sys.argv[1])]
    else:
        root = Path(__file__).resolve().parent.parent
        targets = [
            root / "artifacts" / "figures",
            root / "artifacts" / "smoke" / "figures",
        ]

    any_built = False
    for figures_dir in targets:
        gallery = build_gallery(figures_dir)
        if gallery is not None:
            print(f"Built {gallery}")
            any_built = True
        else:
            print(f"Skipped (no PNGs): {figures_dir}")
    if not any_built:
        sys.exit(1)


if __name__ == "__main__":
    main()
