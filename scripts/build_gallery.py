"""Build static HTML galleries for experiment figures.

Usage:
    python scripts/build_gallery.py [figures_dir ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.gallery import build_figure_gallery


def main() -> None:
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        targets = [
            ROOT / "artifacts" / "figures",
            ROOT / "artifacts" / "smoke" / "figures",
        ]

    any_built = False
    for figures_dir in targets:
        gallery = build_figure_gallery(figures_dir)
        if gallery is not None:
            print(f"Built {gallery}")
            any_built = True
        else:
            print(f"Skipped (no PNGs): {figures_dir}")
    if not any_built:
        sys.exit(1)


if __name__ == "__main__":
    main()
