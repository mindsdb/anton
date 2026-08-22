---
name: build-html-dashboard
description: 'MANDATORY reading before building ANY HTML dashboard, chart, plot, interactive
  report, or browser-based visualization (create_artifact type="html-app", or the
  frontend of a fullstack app). Contains the full HTML output contract: self-contained
  file rules, Apache ECharts setup, dark theme, layout/design standards, and large-dataset
  handling. Recall it BEFORE writing the first line of dashboard HTML. When in doubt,
  recall it.'
metadata:
  display_name: HTML dashboard & visualization output format
  provenance: builtin
---
LIST THE INSIGHTS (terse — one line each, not an essay):
Before coding, list the insights you want to present/convey/highlight as `1 - <chart/infographic/etc>: <insight it conveys and why it matters>..`
Example: `1 - Line chart of weekly signups: shows growth inflection after the March launch, flags whether momentum is sustained.`
This is a checklist, not a brief — no narrative prose, no design discussion.

BUILD THE DASHBOARD — use multiple scratchpad cells, but produce ONE single self-contained HTML file:

Before the first write, call `create_artifact(type="html-app", name=..., description=..., primary="dashboard.html")` and use the returned `<artifact_path>` for every file you write (the HTML, any sibling data files, images, etc.). All paths below referring to "the output directory" mean `<artifact_path>`. The final dashboard MUST be a single .html file with all data, CSS, and JS inlined, with exactly two exceptions, both covered below: an oversized JSON payload, and binary assets such as an image the user uploaded. Both live as sibling files in the SAME directory as the HTML. Never reference a local file OUTSIDE `<artifact_path>` — browsers block local file:// cross-references across directories, and the publisher will not bundle it.

  REROUND DISCIPLINE (critical — most "round-cap exhaustion" failures we've seen on real dashboards come from drifting off one or more of these):
  1. ONE scratchpad, ONE name. Pick a name on the first cell (e.g. `dash`) and reuse it for the entire build. Switching names (`build_pres` → `write_html` → `pres1` …) creates *separate isolated environments* — variables in one don't exist in another — and burns rounds on recovery.
  2. WRITE TO DISK INCREMENTALLY. Open the output `.html` once in 'w' mode, then `open(path, 'a')` to append head → body skeleton → each chart section → nav/JS → closing tags. Each cell appends a small chunk you can sanity-check. Do NOT build a single 20KB+ HTML string in memory and write it at the end.
  3. CAP STRING SIZE PER CELL at ~5KB. Large-string scratchpad calls are the single biggest cause of silent failures (the tool occasionally drops the `code` payload on oversized inputs and the cell comes back with an empty-code error, which still counts against the round cap). If a section is too big, split it.
  4. NEVER re-emit the full HTML mid-build. Append deltas, don't re-print the world. Assembly is a one-line concat at the end, not a re-render of everything you've written so far.
  5. KEEP READS SMALL. To verify what landed, `os.path.getsize(path)` or `open(path).read(2000)` — never `open(path).read()` on a multi-KB HTML.

  SECURITY (critical): Dashboards may be published to the web. NEVER embed API keys, tokens, passwords, connection strings, or any credentials in the HTML, JS, or inline data. Fetch data in scratchpad cells using credentials from environment variables, then serialize only the resulting data into the dashboard. If the user explicitly asks to embed a credential (e.g. for a live-updating dashboard), warn them that publishing will expose it and get confirmation before proceeding.

  Build the parts in separate cells, then assemble at the end:

  CELL 1 — Serialize data to a JS string variable (programmatic, no HTML):
  Serialize all computed data (dataframes, metrics, KPIs) into a Python string. Build a Python dict with keys like "kpis", "tables", "charts" — each containing the relevant data. Convert DataFrames with df.to_dict(orient='records'). Use json.dumps(data, default=str) to handle dates, Decimal, numpy types. Store as a Python variable: `data_js = 'const D = ' + json_string + ';'` — do NOT write to a separate file.

  CELL 2 — Build CSS + HTML structure as a Python string variable:
  Write the HTML head (styles, CDN script tags) and body structure (header, KPIs, chart divs, tabs, tables) as a Python string variable `html_body`. This cell builds the template.

  CELL 3+ — Build JS chart rendering logic as Python string variables:
  Write the JavaScript that initializes charts, populates tables, handles tabs, etc. Split across multiple cells if needed to avoid token limits. Store as `js_charts` etc.

  FINAL CELL — Assemble and write the HTML file:
  Combine: `html = html_body.replace('</body>', f'<script>{data_js}{js_charts}</script></body>')` or similar.

  SELF-CONTAINED OUTPUT (critical):
  Prefer inlining everything — CSS in `<style>`, JS in `<script>`, data as JS variables. A single .html file is the most portable and publishable format. Anything larger than 100KB is the exception: write it to a separate file in the SAME directory and reference it with a relative path — a JSON dataset as `<script src="dashboard_data.js">`, a binary asset as `<img src="screenshot.png">`. The publisher auto-bundles sibling files referenced in the HTML, so both keep working after /publish. Never reference files outside the output directory.

  UPLOADED IMAGES AND BINARY ASSETS:
  Files the user attached or pasted appear with ABSOLUTE paths in the conversation
  context (`.cowork/files/<uuid>/<name>` in the app, `.anton/uploads/clipboard_*.png`
  for a CLI paste). Use the path you were given — never guess one or scan directories.
  If the user refers to a file whose path you cannot see, ask them for it.

  Never point the HTML at that original path: it sits outside `<artifact_path>`, and
  only relative references to files INSIDE that folder are bundled on publish, so the
  artifact would render locally and break once published. Bring the file in instead —
  one scratchpad cell, branching on the 100KB rule above, into the folder that holds
  the primary HTML you're writing (`<artifact_path>` for a standalone dashboard,
  `<artifact_path>/static/` inside a fullstack build):

      import base64, mimetypes, os, shutil
      from pathlib import Path
      name = Path(src).name
      dest_dir = Path(artifact_path)  # or <artifact_path>/static in a fullstack build
      if os.path.getsize(src) < 100_000:
          mime = mimetypes.guess_type(src)[0] or "application/octet-stream"
          b64 = base64.b64encode(Path(src).read_bytes()).decode()
          img_src = f"data:{mime};base64,{b64}"
      else:
          shutil.copy(src, dest_dir / name)
          img_src = name

  A copied sibling MUST be referenced as a literal `src="<name>"` in the HTML. A path
  assembled in JS at runtime is invisible to the publisher and will not be bundled.
  For uploaded DATA files (CSV/JSON) do not use a data: URI — read them in CELL 1 and
  inline the result as a JS variable like any other dataset.

  WHY: (1) Browsers block local file:// cross-references across directories. (2) Splitting the build across cells catches JS/CSS errors early — if a cell has a syntax issue in a string, you'll see it before the final assembly. (3) Large datasets in single cells timeout. (4) Self-contained files can be published to the web via /publish without missing assets.

  PYTHON → JS STRING SAFETY (critical):
  When building JS code inside Python strings, escape sequences get resolved by Python BEFORE writing to the file. This means '\n' in Python becomes a literal newline in the output, which breaks JavaScript string literals. Rules:
  - Use '\\n' in Python if you need a literal \n in the JS output
  - Use raw strings (r"...") for JS code blocks when possible
  - NEVER use '\n', '\t', or '\"' inside JS strings within Python — double-escape them
  - After writing the file, sanity-check that no string literals span multiple lines

Output format:
- Unless the user explicitly asks for a different format, always output visualizations as polished, single-file HTML pages — never raw PNGs or bare image files.

Visual design:
- Make it look good by default. Use a dark theme (#0d1117 background, #e6edf3 text), clean typography (system sans-serif stack), generous padding, and responsive layout.
- ALWAYS use Apache ECharts for interactive charts. Load it via CDN: `<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>`. No Python dependencies needed — just write the HTML with inline JS. Use ECharts' built-in dark theme: `echarts.init(dom, 'dark')`, then customize colors to match #0d1117 background.
- NEVER use Plotly, matplotlib, or other charting libraries unless the user explicitly asks.

Line smoothing (critical — smooth: true misrepresents volatile data):
- DEFAULT: `smooth: false` on ALL line series. Straight segments between data points are the honest representation — they show actual volatility, drawdowns, and inflection points.
- EXCEPTION: Use `smooth: true` ONLY for cumulative/monotonic series (cumulative returns, running totals, growth curves) where the trend matters more than point-to-point moves.
- Decision heuristic: Does the line ever reverse direction meaningfully? If yes → smooth: false. Is it a running sum, cumulative metric, or long-horizon trend? → smooth: true is acceptable.
- Line widths: 2.5 for hero/primary lines, 1.5 for multi-line comparisons, 1 for secondary/reference lines.

Chart readability (critical — labels must NEVER overlap):
- Use `axisLabel: { rotate: -45 }` or `{ rotate: 45 }` on crowded axes. Set `grid: { containLabel: true }` so labels never clip. Use `legend: { type: 'scroll', bottom: 0 }` to place scrollable legends below the chart. For pie/donut charts use `label: { show: true, position: 'outside' }` with `labelLayout: { hideOverlap: true }`. For bar charts with many categories, use horizontal bars (`yAxis` as category) or abbreviate labels with `axisLabel: { formatter }`. Always configure rich `tooltip` with `formatter` functions for precise value display on hover. Use `dataZoom` for time series so users can zoom into ranges.

Multi-tab / multi-view dashboards (critical — charts fail silently on hidden containers):
- ECharts, Chart.js, and Plotly all render nothing when called on a container with `display: none` or 0×0 dimensions — no error, no warning, just a blank chart. NEVER call `echarts.init()` inside `DOMContentLoaded` for tabs/pages that start hidden.
- Initialize charts lazily, gated on first visibility: in the tab-click handler, check a `Set` of already-rendered tabs and call the page's init function only on first visit. Example pattern: `const _rendered = new Set(['overview']); function showPage(name) { /* toggle classes */ if (!_rendered.has(name)) { _rendered.add(name); initChartsFor(name); } }` — only the default-visible page initializes on load.

Layout and composition:
- For non-chart visualizations (tables, reports, dashboards), write clean HTML/CSS directly. Use CSS grid or flexbox. Add subtle styling: rounded corners, soft shadows, hover effects.
- When showing multiple related visuals, combine them into a single page with sections, not separate files. Ensure each chart has enough height (min 400px) and breathing room between them so nothing feels cramped.
- Hero KPI cards at the top (large numbers, color-coded positive/negative, with delta arrows).
- Main narrative chart immediately below the KPIs — this is the chart that tells the story.
- Supporting charts below, each with a clear subtitle explaining what it reveals.
- Annotations on charts: use ECharts `markLine` for thresholds, `markPoint` for outliers, and `markArea` for highlighted regions. A chart without annotations is a missed opportunity.
- The goal: every visualization should look like a polished product page, not a homework assignment. Think dark-mode dashboard, not Jupyter default.

Responsive layout (critical — dashboards must work on phones too):
- ALWAYS include `<meta name="viewport" content="width=device-width, initial-scale=1.0">` in `<head>`. Without this, mobile browsers render at desktop width and the user pinch-zooms.
- Multi-card sections use `grid-template-columns: repeat(auto-fit, minmax(360px, 1fr))` (or 300px on dense layouts). This lets the browser reflow to single-column on narrow viewports without a media query — cards stack vertically instead of getting squashed into unreadable columns.
- Chart containers use `width: 100%` and `height: min(420px, 60vh)` (NOT fixed pixel widths). For each ECharts instance, register a window resize hook so it refits: `window.addEventListener('resize', () => myChart.resize());` — without this, rotating a phone or resizing the window leaves charts the wrong size.
- Tables wrap in `<div style="overflow-x: auto;">` so they scroll horizontally on narrow screens rather than overflowing the page. Do NOT set fixed table widths.
- Default to one column on narrow viewports unless the user explicitly asks for a fixed multi-column layout (e.g. for a printable PDF).