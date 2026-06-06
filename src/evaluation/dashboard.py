"""Build a single HTML dashboard from experiment_report.md + figure gallery."""

from __future__ import annotations

import html
import re
from pathlib import Path


def _md_table_to_html(lines: list[str]) -> str:
    rows: list[list[str]] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.match(r"^[-:\s]+$", c) for c in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""
    head, body = rows[0], rows[1:]
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{html.escape(c)}</th>" for c in head)
    parts.append("</tr></thead><tbody>")
    for row in body:
        parts.append("<tr>")
        parts.extend(f"<td>{html.escape(c)}</td>" for c in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def markdown_report_to_html(md_text: str) -> str:
    """Convert our experiment_report.md subset to HTML sections."""

    sections: list[str] = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            title = line[3:].strip()
            i += 1
            block: list[str] = []
            while i < len(lines) and not lines[i].startswith("## "):
                block.append(lines[i])
                i += 1
            inner: list[str] = [f"<h2>{html.escape(title)}</h2>"]
            j = 0
            while j < len(block):
                b = block[j]
                if b.strip().startswith("|"):
                    table_lines: list[str] = []
                    while j < len(block) and block[j].strip().startswith("|"):
                        table_lines.append(block[j])
                        j += 1
                    inner.append(_md_table_to_html(table_lines))
                    continue
                if b.strip().startswith(">"):
                    inner.append(f"<blockquote>{html.escape(b.lstrip('> ').strip())}</blockquote>")
                    j += 1
                    continue
                if b.strip():
                    inner.append(f"<p>{html.escape(b.strip())}</p>")
                j += 1
            sections.append(f"<section>{''.join(inner)}</section>")
            continue
        if line.startswith("# "):
            i += 1
            continue
        i += 1
    return "\n".join(sections)


def _featured_figures(figures_dir: Path) -> list[tuple[str, Path]]:
    """Pick a small set of representative PNGs if they exist."""

    patterns = [
        "cm_skab_gru_original_seed42.png",
        "roc_skab_gru_original_seed42.png",
        "pr_skab_gru_original_seed42.png",
        "heatmap_batadal_original_seed42_transitions.png",
        "diagram_batadal_original_seed42_transitions.png",
        "sensitivity_batadal_automata_summary_f1_mean.png",
    ]
    found: list[tuple[str, Path]] = []
    for name in patterns:
        path = figures_dir / name
        if path.exists():
            found.append((name, path))
    if found:
        return found
    return [(p.name, p) for p in sorted(figures_dir.glob("*.png"))[:6]]


def build_results_dashboard(
    report_md: Path,
    figures_dir: Path,
    output_path: Path,
    gallery_path: Path | None = None,
) -> Path:
    """Write ``output_path`` HTML combining tables, highlights and gallery link."""

    report_md = report_md.resolve()
    figures_dir = figures_dir.resolve()
    output_path = output_path.resolve()

    md_text = report_md.read_text(encoding="utf-8") if report_md.exists() else ""
    body = markdown_report_to_html(md_text) if md_text else "<p>Rapor bulunamadi.</p>"

    n_figs = len(list(figures_dir.glob("*.png"))) if figures_dir.exists() else 0
    gallery_href = "../figures/gallery.html"
    if gallery_path is not None:
        try:
            gallery_href = Path(gallery_path).resolve().relative_to(output_path.parent).as_posix()
        except ValueError:
            gallery_href = str(gallery_path)

    featured_html: list[str] = []
    for name, path in _featured_figures(figures_dir):
        rel = path.relative_to(output_path.parent).as_posix()
        featured_html.append(
            f"<div class='card'><a href='{html.escape(rel)}' target='_blank'>"
            f"<img src='{html.escape(rel)}' alt='{html.escape(name)}'></a>"
            f"<p>{html.escape(name)}</p></div>"
        )

    page = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Yazlab2 — Sonuç Özeti</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;padding:24px;background:#0f172a;color:#e2e8f0;line-height:1.5}}
header{{margin-bottom:28px;padding-bottom:16px;border-bottom:1px solid #334155}}
h1{{margin:0 0 8px;font-size:1.75rem}}
.sub{{color:#94a3b8;font-size:14px}}
.actions{{margin:16px 0;display:flex;flex-wrap:wrap;gap:10px}}
.btn{{display:inline-block;padding:10px 16px;border-radius:8px;background:#2563eb;color:#fff;text-decoration:none;font-weight:600;font-size:14px}}
.btn.secondary{{background:#334155}}
.summary{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin:20px 0}}
.stat{{background:#1e293b;border-radius:10px;padding:14px}}
.stat strong{{display:block;font-size:1.4rem;color:#f8fafc}}
.stat span{{font-size:12px;color:#94a3b8}}
h2{{font-size:1.1rem;margin:0 0 12px;color:#f1f5f9}}
section{{margin:28px 0;padding:20px;background:#1e293b;border-radius:12px;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #334155;padding:8px 10px;text-align:left}}
th{{background:#0f172a;color:#cbd5e1}}
tr:nth-child(even){{background:#172033}}
blockquote{{margin:12px 0;padding:10px 14px;border-left:3px solid #2563eb;background:#0f172a;color:#cbd5e1;font-size:13px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px;margin-top:12px}}
.grid .card{{background:#0f172a;border-radius:8px;padding:8px}}
.grid img{{width:100%;height:auto;border-radius:6px;background:#fff}}
.grid p{{font-size:11px;margin:6px 0 0;word-break:break-all;color:#94a3b8}}
</style></head><body>
<header>
<h1>Yazlab2 — Deney Sonuçları</h1>
<p class="sub">Olasılıksal otomata vs derin öğrenme · BATADAL + SKAB · tam koşum</p>
<div class="actions">
<a class="btn" href="{html.escape(gallery_href)}">Tüm figürler ({n_figs} PNG)</a>
<a class="btn secondary" href="results/experiment_report.md">Ham rapor (Markdown)</a>
</div>
<div class="summary">
<div class="stat"><strong>SKAB DL</strong><span>GRU/LSTM F1 ~ 0.87</span></div>
<div class="stat"><strong>Otomata</strong><span>Açıklanabilir, F1 düşük</span></div>
<div class="stat"><strong>Cross-dataset</strong><span>Transfer zayıf (~0.04–0.09)</span></div>
<div class="stat"><strong>{n_figs}</strong><span>üretilen figür</span></div>
</div>
</header>
<section>
<h2>Öne çıkan figürler</h2>
<div class="grid">
{''.join(featured_html) if featured_html else '<p>Figür klasörü bos.</p>'}
</div>
</section>
{body}
<footer class="sub" style="margin-top:32px">Eğitim amaçlı ders projesi · Sinem Gül & Elif Aysan</footer>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path
