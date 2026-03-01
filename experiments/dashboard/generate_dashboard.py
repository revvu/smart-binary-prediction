from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "experiments"
OUTPUT_HTML = ROOT / "experiments" / "dashboard" / "index.html"
SMART_ALGO_MD = ROOT / "smart_algorithm.md"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"}


@dataclass
class FigureItem:
    filename: str
    rel_src: str
    title: str


@dataclass
class ExperimentItem:
    slug: str
    display_name: str
    readme: str
    figures: list[FigureItem]


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _first_heading(markdown_text: str) -> str | None:
    for line in markdown_text.splitlines():
        m = re.match(r"^#\s+(.*)$", line.strip())
        if m:
            return m.group(1).strip()
    return None


def _nice_name_from_slug(slug: str) -> str:
    parts = slug.split("_", 1)
    if len(parts) == 2:
        return parts[1].replace("_", " ").title()
    return slug.replace("_", " ").title()


def _parse_figure_titles_from_index(index_md: str) -> dict[str, str]:
    title_by_file: dict[str, str] = {}
    lines = index_md.splitlines()
    current_title: str | None = None
    current_file: str | None = None

    for line in lines:
        t = re.search(r"Title:\s*`([^`]+)`", line)
        if t:
            current_title = t.group(1).strip()
        f = re.search(r"File:\s*`([^`]+)`", line)
        if f:
            current_file = f.group(1).strip()
        if current_title and current_file:
            title_by_file[current_file] = current_title
            current_title = None
            current_file = None

    return title_by_file


def _discover_experiments() -> list[ExperimentItem]:
    exp_dirs = sorted(
        [p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir() and re.match(r"^exp\d+_", p.name)],
        key=lambda p: p.name,
    )

    experiments: list[ExperimentItem] = []
    for exp_dir in exp_dirs:
        readme = _read_text(exp_dir / "README.md")
        heading = _first_heading(readme)
        display_name = heading or _nice_name_from_slug(exp_dir.name)

        figures_dir = exp_dir / "figures"
        title_map: dict[str, str] = {}
        if (figures_dir / "INDEX.md").exists():
            title_map = _parse_figure_titles_from_index(_read_text(figures_dir / "INDEX.md"))

        figures: list[FigureItem] = []
        if figures_dir.exists():
            for f in sorted(figures_dir.iterdir(), key=lambda p: p.name):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    rel_src = os.path.relpath(f, OUTPUT_HTML.parent).replace("\\", "/")
                    figures.append(
                        FigureItem(
                            filename=f.name,
                            rel_src=rel_src,
                            title=title_map.get(f.name, f.name),
                        )
                    )

        experiments.append(
            ExperimentItem(
                slug=exp_dir.name,
                display_name=display_name,
                readme=readme,
                figures=figures,
            )
        )

    return experiments


def _to_dict(experiments: list[ExperimentItem]) -> list[dict[str, Any]]:
    return [
        {
            "slug": e.slug,
            "displayName": e.display_name,
            "readme": e.readme,
            "figures": [
                {
                    "filename": f.filename,
                    "src": f.rel_src,
                    "title": f.title,
                }
                for f in e.figures
            ],
        }
        for e in experiments
    ]


def _build_html(payload_json: str) -> str:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SMART Experiment Reports</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --ink: #101827;
      --ink-soft: #4a5568;
      --line: #d7dee8;
      --card: #ffffff;
      --accent: #0f7a64;
      --accent-soft: #d9f4ed;
      --shadow: 0 14px 36px rgba(17, 30, 50, 0.10);
      --r: 16px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(900px 520px at -5% -10%, #e8f8f2 12%, transparent 65%),
        radial-gradient(1000px 520px at 110% 0%, #e7efff 10%, transparent 62%),
        var(--bg);
    }
    .container {
      max-width: 1160px;
      margin: 0 auto;
      padding: 24px 18px 56px;
    }
    .top {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--r);
      padding: 20px 22px;
      box-shadow: var(--shadow);
      margin-bottom: 14px;
    }
    .title {
      margin: 0;
      font-size: clamp(1.35rem, 2.8vw, 2rem);
      letter-spacing: -0.02em;
    }
    .subtitle {
      margin: 8px 0 0;
      color: var(--ink-soft);
      line-height: 1.5;
    }
    .tabbar-wrap {
      position: sticky;
      top: 0;
      z-index: 20;
      padding: 8px 0;
      backdrop-filter: blur(6px);
    }
    .tabbar {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(255,255,255,0.88);
      padding: 8px;
      box-shadow: var(--shadow);
    }
    .tab {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 13px;
      white-space: nowrap;
      font-size: 0.92rem;
      cursor: pointer;
      transition: transform 160ms ease, background 160ms ease;
    }
    .tab:hover { transform: translateY(-1px); }
    .tab.active {
      background: var(--accent-soft);
      border-color: #9cd9c8;
      color: #065846;
      font-weight: 600;
    }
    .panel { display: none; margin-top: 14px; }
    .panel.active { display: block; }
    .report {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--r);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .report-head {
      padding: 18px 22px 12px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #fbfdfc 0%, #f7fbf9 100%);
    }
    .report-head h2 {
      margin: 0;
      font-size: 1.18rem;
      letter-spacing: -0.01em;
    }
    .report-body {
      padding: 18px 22px 22px;
    }
    .md {
      font-size: 0.97rem;
      line-height: 1.62;
      color: #1f2738;
      overflow-wrap: anywhere;
    }
    .md h1, .md h2, .md h3, .md h4, .md h5, .md h6 {
      margin: 1.05em 0 0.5em;
      line-height: 1.25;
      font-family: "IBM Plex Serif", "Georgia", serif;
      letter-spacing: -0.01em;
      color: #121b2d;
    }
    .md p { margin: 0.55em 0 0.9em; }
    .md ul, .md ol { margin: 0.45em 0 0.9em 1.2em; }
    .md li { margin-bottom: 0.3em; }
    .md code {
      font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 0.88em;
      background: #eef2f7;
      border: 1px solid #dee6f0;
      border-radius: 6px;
      padding: 0.08em 0.34em;
    }
    .md pre {
      margin: 0.9em 0;
      padding: 11px;
      border-radius: 10px;
      border: 1px solid #273148;
      background: #111827;
      color: #e5ecff;
      overflow: auto;
    }
    .md pre code {
      padding: 0;
      border: 0;
      background: transparent;
      color: inherit;
    }
    .md a { color: var(--accent); text-decoration: none; }
    .md a:hover { text-decoration: underline; }

    .fig-section {
      margin-top: 22px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
    }
    .fig-section h3 {
      margin: 0 0 10px;
      font-size: 1.05rem;
      letter-spacing: -0.01em;
    }
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }
    .figure {
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: #fff;
    }
    .figure a {
      display: block;
      background: #f5f8fb;
    }
    .figure img {
      width: 100%;
      display: block;
      aspect-ratio: 16/10;
      object-fit: contain;
    }
    .figure-meta {
      padding: 10px;
      border-top: 1px solid var(--line);
    }
    .figure-title {
      margin: 0 0 6px;
      font-size: 0.92rem;
      color: #22304a;
    }
    .figure-file {
      margin: 0;
      color: var(--ink-soft);
      font-size: 0.84rem;
    }
    .empty {
      color: var(--ink-soft);
      font-style: italic;
    }
  </style>
</head>
<body>
  <main class="container">
    <section class="top">
      <h1 class="title">SMART Experiment Reports</h1>
      <p class="subtitle">Overview tab renders <code>smart_algorithm.md</code>. Each experiment tab renders its README as a report with figures appended at the end.</p>
    </section>

    <div class="tabbar-wrap">
      <div id="tabs" class="tabbar"></div>
    </div>

    <div id="panels"></div>
  </main>

  <script>
    const payload = __PAYLOAD_JSON__;

    function escHtml(s) {
      return s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
    }

    function inlineMd(s) {
      let x = escHtml(s);
      x = x.replace(/`([^`]+)`/g, "<code>$1</code>");
      x = x.replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>");
      x = x.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
      return x;
    }

    function renderMarkdown(md) {
      if (!md || !md.trim()) return '<p class="empty">No markdown content found.</p>';
      const lines = md.replace(/\\r\\n/g, "\\n").split("\\n");
      let out = "";
      let inCode = false;
      let listType = null;
      let para = [];

      function flushPara() {
        if (para.length) {
          out += "<p>" + inlineMd(para.join(" ")) + "</p>";
          para = [];
        }
      }

      function closeList() {
        if (listType) {
          out += listType === "ol" ? "</ol>" : "</ul>";
          listType = null;
        }
      }

      for (const rawLine of lines) {
        const trimmed = rawLine.trim();
        if (trimmed.startsWith("```")) {
          flushPara();
          closeList();
          if (!inCode) {
            inCode = true;
            out += "<pre><code>";
          } else {
            inCode = false;
            out += "</code></pre>";
          }
          continue;
        }
        if (inCode) {
          out += escHtml(rawLine) + "\\n";
          continue;
        }
        if (!trimmed) {
          flushPara();
          closeList();
          continue;
        }
        const h = trimmed.match(/^(#{1,6})\\s+(.*)$/);
        if (h) {
          flushPara();
          closeList();
          out += "<h" + h[1].length + ">" + inlineMd(h[2]) + "</h" + h[1].length + ">";
          continue;
        }
        const ul = trimmed.match(/^[-*]\\s+(.*)$/);
        if (ul) {
          flushPara();
          if (listType !== "ul") {
            closeList();
            listType = "ul";
            out += "<ul>";
          }
          out += "<li>" + inlineMd(ul[1]) + "</li>";
          continue;
        }
        const ol = trimmed.match(/^\\d+\\.\\s+(.*)$/);
        if (ol) {
          flushPara();
          if (listType !== "ol") {
            closeList();
            listType = "ol";
            out += "<ol>";
          }
          out += "<li>" + inlineMd(ol[1]) + "</li>";
          continue;
        }
        para.push(trimmed);
      }

      flushPara();
      closeList();
      return out;
    }

    function renderFigures(figures) {
      if (!figures || figures.length === 0) {
        return '<p class="empty">No figures found in this experiment\\'s <code>figures/</code> folder.</p>';
      }
      let html = '<div class="gallery">';
      for (const fig of figures) {
        html += '<article class="figure">' +
          '<a href="' + fig.src + '" target="_blank" rel="noopener noreferrer">' +
          '<img src="' + fig.src + '" alt="' + escHtml(fig.title) + '" loading="lazy" />' +
          '</a>' +
          '<div class="figure-meta">' +
          '<p class="figure-title">' + escHtml(fig.title) + '</p>' +
          '<p class="figure-file"><code>' + escHtml(fig.filename) + '</code></p>' +
          '</div>' +
          '</article>';
      }
      html += '</div>';
      return html;
    }

    function overviewHtml() {
      return '<article class="report">' +
        '<header class="report-head"><h2>SMART Algorithm Overview</h2></header>' +
        '<section class="report-body md">' + renderMarkdown(payload.smartAlgorithm) + '</section>' +
        '</article>';
    }

    function experimentHtml(exp) {
      return '<article class="report">' +
        '<header class="report-head"><h2>' + escHtml(exp.displayName) + '</h2></header>' +
        '<section class="report-body">' +
        '<div class="md">' + renderMarkdown(exp.readme || "") + '</div>' +
        '<section class="fig-section">' +
        '<h3>Figures</h3>' +
        renderFigures(exp.figures) +
        '</section>' +
        '</section>' +
        '</article>';
    }

    const tabs = [{ key: "overview", label: "Overview" }];
    for (const exp of payload.experiments) {
      tabs.push({ key: exp.slug, label: exp.slug });
    }

    const tabsRoot = document.getElementById("tabs");
    const panelsRoot = document.getElementById("panels");

    function addPanel(key, html) {
      const panel = document.createElement("section");
      panel.className = "panel";
      panel.dataset.tab = key;
      panel.innerHTML = html;
      panelsRoot.appendChild(panel);
    }

    for (const tab of tabs) {
      const b = document.createElement("button");
      b.className = "tab";
      b.dataset.tab = tab.key;
      b.textContent = tab.label;
      tabsRoot.appendChild(b);
    }

    addPanel("overview", overviewHtml());
    for (const exp of payload.experiments) {
      addPanel(exp.slug, experimentHtml(exp));
    }

    function activate(key) {
      document.querySelectorAll(".tab").forEach((el) => el.classList.toggle("active", el.dataset.tab === key));
      document.querySelectorAll(".panel").forEach((el) => el.classList.toggle("active", el.dataset.tab === key));
    }

    tabsRoot.addEventListener("click", (ev) => {
      const t = ev.target;
      if (!(t instanceof HTMLElement)) return;
      if (!t.classList.contains("tab")) return;
      activate(t.dataset.tab || "overview");
    });

    activate("overview");
  </script>
</body>
</html>
"""
    return template.replace("__PAYLOAD_JSON__", payload_json)


def main() -> None:
    experiments = _discover_experiments()
    payload = {
        "smartAlgorithm": _read_text(SMART_ALGO_MD),
        "experiments": _to_dict(experiments),
    }

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(_build_html(json.dumps(payload, ensure_ascii=False)), encoding="utf-8")

    print(f"Wrote dashboard: {OUTPUT_HTML}")
    print(f"Experiments indexed: {len(experiments)}")


if __name__ == "__main__":
    main()
