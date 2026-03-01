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
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      },
      options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }
    };
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,500&family=Fira+Code:wght@400;500&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f5f0e8;
      --bg-paper: #fffdf7;
      --bg-warm: #faf6ed;
      --bg-inset: #f0ebe0;
      --text-ink: #2c2418;
      --text-body: #4a3f30;
      --text-faded: #8a7e6b;
      --text-ghost: #b8ad98;
      --accent-terracotta: #c4634a;
      --accent-sage: #6b8f71;
      --accent-ochre: #c49a3c;
      --accent-navy: #364b6b;
      --border-rule: #d4cbb8;
      --border-light: #e5ded0;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: "Libre Baskerville", Georgia, serif;
      background: var(--bg);
      color: var(--text-body);
      min-height: 100vh;
      font-size: 18px;
    }
    body::after {
      content: "";
      position: fixed;
      inset: 0;
      background-image: url("data:image/svg+xml,%3Csvg width='200' height='200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
      pointer-events: none;
      z-index: 9999;
    }

    .container {
      max-width: 1220px;
      margin: 0 auto;
      padding: 2.5rem;
    }

    .masthead {
      text-align: center;
      padding: 3.8rem 2.4rem 2.5rem;
      border-bottom: 2px solid var(--text-ink);
      animation: fadeIn 0.8s ease;
    }
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    .masthead-overline {
      font-family: "Fira Code", monospace;
      font-size: 0.9rem;
      letter-spacing: 0.38em;
      text-transform: uppercase;
      color: var(--text-faded);
      margin-bottom: 0.8rem;
    }
    .masthead h1 {
      font-family: "Cormorant Garamond", serif;
      font-size: 4rem;
      font-weight: 700;
      color: var(--text-ink);
      letter-spacing: 0.04em;
      line-height: 1.1;
      margin-bottom: 0.5rem;
    }
    .masthead-sub {
      font-family: "Cormorant Garamond", serif;
      font-size: 1.5rem;
      font-style: italic;
      color: var(--text-faded);
      font-weight: 400;
    }
    .masthead-sub code {
      font-family: "Fira Code", monospace;
      font-size: 0.95rem;
      font-style: normal;
      color: var(--accent-terracotta);
      background: rgba(196, 99, 74, 0.08);
      padding: 0.1em 0.4em;
      border-radius: 2px;
    }

    .nav-bar {
      display: flex;
      justify-content: center;
      gap: 0;
      border-bottom: 1px solid var(--border-rule);
      background: var(--bg-warm);
      animation: fadeIn 0.8s ease 0.15s both;
    }
    .nav-item {
      font-family: "Cormorant Garamond", serif;
      font-size: 1.2rem;
      font-weight: 600;
      padding: 1.2rem 1.9rem;
      color: var(--text-faded);
      cursor: pointer;
      border: none;
      background: none;
      position: relative;
      transition: all 0.3s ease;
      letter-spacing: 0.02em;
      white-space: nowrap;
    }
    .nav-item:hover { color: var(--text-body); }
    .nav-item.active {
      color: var(--accent-terracotta);
      background: var(--bg-paper);
      font-weight: 600;
    }
    .nav-item.active::before {
      content: "§";
      margin-right: 0.4rem;
      opacity: 0.6;
    }

    .panel { display: none; }
    .panel.active { display: block; animation: fadeIn 0.35s ease; }

    .report {
      background: var(--bg-paper);
      padding: 3.5rem 4.25rem;
      min-height: 60vh;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .title-block {
      text-align: center;
      margin-bottom: 3rem;
      padding-bottom: 2.4rem;
      border-bottom: 1px solid var(--border-light);
    }
    .exp-number {
      font-family: "Fira Code", monospace;
      font-size: 0.8rem;
      letter-spacing: 0.4em;
      text-transform: uppercase;
      color: var(--accent-terracotta);
      display: inline-block;
      border: 1px solid rgba(196, 99, 74, 0.3);
      padding: 0.3em 1em;
      border-radius: 2px;
      margin-bottom: 1rem;
    }
    .title-block h2 {
      font-family: "Cormorant Garamond", serif;
      font-size: 3rem;
      font-weight: 700;
      color: var(--text-ink);
      margin-bottom: 0.6rem;
      line-height: 1.15;
    }
    .title-block .origin {
      font-size: 1.2rem;
      font-style: italic;
      color: var(--text-faded);
    }
    .title-block .origin code {
      font-family: "Fira Code", monospace;
      font-size: 0.95rem;
      font-style: normal;
      color: var(--accent-navy);
    }

    .md {
      font-size: 1.1rem;
      line-height: 2;
      color: var(--text-body);
      overflow-wrap: anywhere;
      text-align: justify;
      hyphens: auto;
    }
    .md h1, .md h2, .md h3, .md h4, .md h5, .md h6 {
      font-family: "Cormorant Garamond", serif;
      color: var(--text-ink);
      line-height: 1.2;
      letter-spacing: 0.01em;
      margin: 1.4em 0 0.45em;
    }
    .md h1 { font-size: 2.65rem; border-bottom: 1px solid var(--border-light); padding-bottom: 0.35em; }
    .md h2 { font-size: 2rem; }
    .md h3 { font-size: 1.5rem; }
    .md p { margin: 0 0 1.3rem; }
    .md ul, .md ol { margin: 0.35rem 0 1.2rem 1.5rem; }
    .md li { margin-bottom: 0.45rem; }

    .md code {
      font-family: "Fira Code", monospace;
      font-size: 0.88em;
      color: var(--accent-navy);
      background: rgba(54, 75, 107, 0.06);
      border: 1px solid rgba(54, 75, 107, 0.12);
      border-radius: 2px;
      padding: 0.15em 0.45em;
    }
    .md pre {
      margin: 1rem 0 1.4rem;
      padding: 0.9rem;
      background: var(--bg-inset);
      border: 1px solid var(--border-light);
      border-radius: 4px;
      color: var(--text-ink);
      overflow: auto;
    }
    .md pre code {
      color: inherit;
      padding: 0;
      border: 0;
      background: transparent;
    }
    .md a { color: var(--accent-navy); text-decoration: none; border-bottom: 1px dotted rgba(54, 75, 107, 0.35); }
    .md a:hover { text-decoration: underline; }

    .fig-section {
      margin-top: 2.8rem;
      padding-top: 1.5rem;
      border-top: 1px solid var(--border-light);
    }
    .fig-section h3 {
      font-family: "Cormorant Garamond", serif;
      font-size: 1.9rem;
      color: var(--text-ink);
      margin: 0 0 1rem;
    }
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 1.2rem;
    }
    .figure {
      background: var(--bg-warm);
      border: 1px solid var(--border-light);
      border-radius: 4px;
      overflow: clip;
    }
    .figure a { display: block; background: #f7f2e8; }
    .figure img {
      width: 100%;
      display: block;
      aspect-ratio: 16/10;
      object-fit: contain;
    }
    .figure-meta {
      padding: 0.95rem 1.05rem;
      border-top: 1px solid var(--border-light);
    }
    .figure-title {
      margin: 0 0 0.35rem;
      font-family: "Cormorant Garamond", serif;
      font-size: 1.3rem;
      color: var(--text-ink);
    }
    .figure-file {
      margin: 0;
      color: var(--text-faded);
      font-size: 0.9rem;
    }
    .empty { color: var(--text-faded); font-style: italic; }
    .paper-end {
      text-align: center;
      margin-top: 3.4rem;
      color: var(--text-ghost);
      font-family: "Cormorant Garamond", serif;
      font-size: 1.8rem;
      letter-spacing: 0.5em;
    }

    @media (max-width: 768px) {
      body { font-size: 17px; }
      .container { padding: 1.2rem; }
      .report { padding: 2.4rem 1.8rem; }
      .masthead { padding: 2.8rem 1.2rem 1.8rem; }
      .masthead h1 { font-size: 2.7rem; }
      .masthead-sub { font-size: 1.2rem; }
      .title-block h2 { font-size: 2.25rem; }
      .md { font-size: 1.03rem; line-height: 1.85; }
      .md h1 { font-size: 2.1rem; }
      .md h2 { font-size: 1.7rem; }
      .md h3 { font-size: 1.35rem; }
      .nav-bar { overflow-x: auto; justify-content: flex-start; }
      .nav-item { font-size: 1.1rem; padding: 1.05rem 1.3rem; }
      .gallery { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="container">
    <header class="masthead">
      <div class="masthead-overline">Experiment Reports · SMART Algorithm</div>
      <h1>SMART Experiment<br>Reports</h1>
      <p class="masthead-sub">Overview renders <code>smart_algorithm.md</code> — each experiment tab renders its README as a report with appended figures.</p>
    </header>
    <nav id="tabs" class="nav-bar"></nav>
    <div id="panels"></div>
  </div>

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

    function titleFromSlug(slug) {
      return slug.replace(/^exp\\d+_/, "").replaceAll("_", " ").replace(/\\b\\w/g, (c) => c.toUpperCase());
    }

    function expNumber(slug) {
      const m = slug.match(/^exp(\\d+)_/);
      return m ? m[1] : "--";
    }

    function overviewHtml() {
      return '<article class="report">' +
        '<header class="title-block">' +
        '<div class="exp-number">Overview</div>' +
        '<h2>SMART Algorithm Deep Dive</h2>' +
        '<p class="origin">Source: <code>smart_algorithm.md</code></p>' +
        '</header>' +
        '<section class="md">' + renderMarkdown(payload.smartAlgorithm) + '</section>' +
        '<div class="paper-end">· · ·</div>' +
        '</article>';
    }

    function experimentHtml(exp) {
      return '<article class="report">' +
        '<header class="title-block">' +
        '<div class="exp-number">Experiment ' + expNumber(exp.slug) + '</div>' +
        '<h2>' + escHtml(titleFromSlug(exp.slug)) + '</h2>' +
        '<p class="origin">' + escHtml(exp.displayName || exp.slug) + '</p>' +
        '</header>' +
        '<section class="md">' + renderMarkdown(exp.readme || "") + '</section>' +
        '<section class="fig-section">' +
        '<h3>Figures</h3>' +
        renderFigures(exp.figures) +
        '</section>' +
        '<div class="paper-end">· · ·</div>' +
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
      b.className = "nav-item";
      b.dataset.tab = tab.key;
      b.textContent = tab.key === "overview" ? "Overview" : titleFromSlug(tab.label);
      tabsRoot.appendChild(b);
    }

    addPanel("overview", overviewHtml());
    for (const exp of payload.experiments) {
      addPanel(exp.slug, experimentHtml(exp));
    }

    function activate(key) {
      document.querySelectorAll(".nav-item").forEach((el) => el.classList.toggle("active", el.dataset.tab === key));
      document.querySelectorAll(".panel").forEach((el) => el.classList.toggle("active", el.dataset.tab === key));
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetClear();
        window.MathJax.typesetPromise().catch(() => {});
      }
    }

    tabsRoot.addEventListener("click", (ev) => {
      const t = ev.target;
      if (!(t instanceof HTMLElement)) return;
      if (!t.classList.contains("nav-item")) return;
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
