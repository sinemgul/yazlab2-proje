from __future__ import annotations

import html
import re
from collections import defaultdict
from pathlib import Path

from src.evaluation.gallery import CATEGORY_PATTERNS, categorise


def _parse_md_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    header: list[str] = []
    rows: list[list[str]] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.match(r"^[-:\s]+$", c) for c in cells):
            continue
        if not header:
            header = cells
        else:
            rows.append(cells)
    return header, rows


def _table_html(header: list[str], rows: list[list[str]]) -> str:
    if not header:
        return ""
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{html.escape(c)}</th>" for c in header)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        parts.extend(f"<td>{html.escape(c)}</td>" for c in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def parse_report_sections(md_text: str) -> list[dict]:
    """Return every ``##`` section from experiment_report.md in order."""

    sections: list[dict] = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].startswith("## "):
            i += 1
            continue
        title = lines[i][3:].strip()
        i += 1
        block: list[str] = []
        while i < len(lines) and not lines[i].startswith("## "):
            block.append(lines[i])
            i += 1
        table_lines: list[str] = []
        blockquote = ""
        j = 0
        while j < len(block):
            if block[j].strip().startswith("|"):
                while j < len(block) and block[j].strip().startswith("|"):
                    table_lines.append(block[j])
                    j += 1
                continue
            if block[j].strip().startswith(">"):
                blockquote = block[j].lstrip("> ").strip()
            j += 1
        header, rows = _parse_md_table(table_lines)
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48] or f"sec-{len(sections)}"
        sections.append(
            {
                "title": title,
                "slug": slug,
                "header": header,
                "rows": rows,
                "blockquote": blockquote,
                "html": _table_html(header, rows),
            }
        )
    return sections


def _figures_html(figures_dir: Path, page_parent: Path) -> tuple[str, int]:
    """Render every PNG grouped by category."""

    if not figures_dir.exists():
        return "<p>Figür klasörü bulunamadı.</p>", 0

    images = sorted(figures_dir.glob("*.png"))
    if not images:
        return "<p>PNG figür yok.</p>", 0

    grouped: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        grouped[categorise(image.name)].append(image)

    parts: list[str] = []
    for label in [*CATEGORY_PATTERNS.keys(), "Other"]:
        files = grouped.get(label)
        if not files:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower())
        parts.append(f'<div class="fig-group" id="fig-{slug}">')
        parts.append(f"<h3>{html.escape(label)} <span class='count'>({len(files)})</span></h3>")
        parts.append("<div class='grid'>")
        for image in files:
            rel = image.relative_to(page_parent).as_posix()
            parts.append(
                "<div class='card'>"
                f"<a href='{html.escape(rel)}' target='_blank'>"
                f"<img loading='lazy' src='{html.escape(rel)}' alt='{html.escape(image.name)}'></a>"
                f"<p>{html.escape(image.name)}</p></div>"
            )
        parts.append("</div></div>")

    return "\n".join(parts), len(images)


def _file_links(results_dir: Path, page_parent: Path) -> str:
    names = [
        "experiment_report.md",
        "statistical_tests.csv",
        "batadal_automata_summary.csv",
        "batadal_dl_summary.csv",
        "skab_automata_summary.csv",
        "skab_dl_summary.csv",
        "cross_dataset_runs.jsonl",
        "experiment_summary.json",
    ]
    items: list[str] = []
    for name in names:
        path = results_dir / name
        if path.exists():
            rel = path.relative_to(page_parent).as_posix()
            items.append(f"<li><a href='{html.escape(rel)}'>{html.escape(name)}</a></li>")
    if not items:
        return ""
    return "<ul class='file-list'>" + "".join(items) + "</ul>"


def build_results_dashboard(
    report_md: Path,
    figures_dir: Path,
    output_path: Path,
    gallery_path: Path | None = None,
    explanations_dir: Path | None = None,
) -> Path:
    """Write ``dashboard.html`` with the complete experiment output."""

    del explanations_dir  # reserved; explanations listed via results links
    report_md = report_md.resolve()
    figures_dir = figures_dir.resolve()
    output_path = output_path.resolve()
    page_parent = output_path.parent
    results_dir = report_md.parent

    md_text = report_md.read_text(encoding="utf-8") if report_md.exists() else ""
    sections = parse_report_sections(md_text) if md_text else []

    fig_body, n_figs = _figures_html(figures_dir, page_parent)
    file_links = _file_links(results_dir, page_parent)

    nav_items = [
        f"<a href='#{html.escape(s['slug'])}'>{html.escape(s['title'][:60])}</a>"
        for s in sections
    ]
    nav_items.append("<a href='#tum-figurler'>Tüm figürler</a>")
    if file_links:
        nav_items.append("<a href='#dosyalar'>Ham dosyalar</a>")

    table_sections: list[str] = []
    for sec in sections:
        bq = (
            f"<blockquote>{html.escape(sec['blockquote'])}</blockquote>"
            if sec["blockquote"]
            else ""
        )
        table_sections.append(
            f"<section class='block' id='{html.escape(sec['slug'])}'>"
            f"<h2>{html.escape(sec['title'])}</h2>"
            f"<div class='table-wrap'>{sec['html'] or '<p>Tablo yok.</p>'}</div>{bq}</section>"
        )

    gallery_href = ""
    if gallery_path and gallery_path.exists():
        try:
            gallery_href = gallery_path.resolve().relative_to(page_parent).as_posix()
        except ValueError:
            gallery_href = str(gallery_path)

    gallery_link = (
        f"<a class='btn secondary' href='{html.escape(gallery_href)}'>Ayrı galeri sayfası</a>"
        if gallery_href
        else ""
    )

    page = f"""<!DOCTYPE html>
<html lang="tr"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Yazlab2 — Tam Deney Sonuçları</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:system-ui,sans-serif;margin:0;background:#0f172a;color:#e2e8f0;line-height:1.5}}
.top{{position:sticky;top:0;z-index:20;background:rgba(15,23,42,.96);border-bottom:1px solid #334155;padding:14px 20px;backdrop-filter:blur(8px)}}
.top h1{{margin:0 0 6px;font-size:1.35rem}}
.meta{{color:#94a3b8;font-size:13px}}
.nav{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}}
.nav a{{color:#93c5fd;text-decoration:none;font-size:12px;padding:4px 8px;border-radius:6px;background:#1e293b;border:1px solid #334155}}
.nav a:hover{{background:#334155}}
.wrap{{max-width:1400px;margin:0 auto;padding:20px}}
.block{{margin:28px 0;padding:22px;background:#1e293b;border-radius:12px;border:1px solid #334155}}
.block h2{{margin:0 0 14px;font-size:1.15rem;color:#f8fafc}}
.block h3{{margin:20px 0 10px;font-size:1rem;color:#e2e8f0}}
.table-wrap{{overflow:auto;max-height:none}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #334155;padding:8px 10px;text-align:left;vertical-align:top}}
th{{background:#0f172a;color:#cbd5e1;position:sticky;top:0}}
tr:nth-child(even){{background:#172033}}
blockquote{{margin:14px 0 0;padding:10px 14px;border-left:3px solid #2563eb;background:#0f172a;color:#cbd5e1;font-size:13px}}
.count{{color:#64748b;font-weight:400;font-size:.9em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px;margin-top:10px}}
.card{{background:#0f172a;border-radius:8px;padding:8px;border:1px solid #334155}}
.card img{{width:100%;height:auto;border-radius:6px;background:#fff;display:block}}
.card p{{font-size:11px;margin:6px 0 0;word-break:break-all;color:#94a3b8}}
.fig-group{{margin-top:24px;padding-top:8px;border-top:1px solid #334155}}
.file-list{{margin:0;padding-left:1.2rem;font-size:14px}}
.file-list a{{color:#60a5fa}}
.actions{{margin-top:8px;display:flex;gap:8px;flex-wrap:wrap}}
.btn{{display:inline-block;padding:8px 14px;border-radius:8px;background:#2563eb;color:#fff;text-decoration:none;font-size:13px;font-weight:600}}
.btn.secondary{{background:#475569}}
footer{{color:#64748b;font-size:12px;padding:24px 20px 40px;text-align:center}}
</style></head><body>
<header class="top">
<h1>Yazlab2 — Tam Deney Sonuçları</h1>
<p class="meta">BATADAL + SKAB · tam koşum · {len(sections)} tablo bölümü · {n_figs} figür · Sinem Gül & Elif Aysan</p>
<nav class="nav">{' '.join(nav_items)}</nav>
<div class="actions">{gallery_link}</div>
</header>
<main class="wrap">
{''.join(table_sections)}
<section class="block" id="tum-figurler">
<h2>Tüm figürler ({n_figs})</h2>
<p class="meta">Confusion matrix, ROC/PR, heatmap, state diagram, sensitivity — tam koşumdan.</p>
{fig_body}
</section>
<section class="block" id="dosyalar">
<h2>Ham çıktı dosyaları</h2>
<p class="meta">CSV, JSONL ve Markdown kaynakları.</p>
{file_links or '<p>Dosya bulunamadı.</p>'}
</section>
</main>
<footer>Yazılım Laboratuvarı II · github.com/sinemgul/yazlab2-proje</footer>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path
