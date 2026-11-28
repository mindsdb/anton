---
name: build-html-dashboard
description: 'ONLY for writing dashboard or chart HTML BY HAND. NOT needed on the normal
  path — create_artifact(type="html-app") followed by generate_artifact writes the
  dashboard itself and carries this contract internally. NOT needed to change the content
  of a report or add sections (report_tools does that; see DIRECT REPORTS). Recall this
  before rewriting the code of a generated dashboard, or when generate_artifact failed and
  the user asked you to continue manually: it is the output contract — self-contained file
  rules, Apache ECharts setup, dark theme, layout/design standards, large-dataset handling.
  Also recall it to write an inline chart (create_artifact with display="inline") or a
  widget the user picks, confirms or fills in (display="widget"): its Inline section says
  how that page differs and documents the window.anton state and submit API.
  Not needed to repair a page generate_artifact returned as browser_check_failed: that
  result''s instruction says how.'
metadata:
  display_name: HTML dashboard & visualization output format
  provenance: builtin
---
LIST THE INSIGHTS (terse — one line each, not an essay):
Before coding, list the insights you want to present/convey/highlight as `1 - <chart/infographic/etc>: <insight it conveys and why it matters>..`
Example: `1 - Line chart of weekly signups: shows growth inflection after the March launch, flags whether momentum is sustained.`
This is a checklist, not a brief — no narrative prose, no design discussion.

For an Earth map or globe, follow the always-available MAPS AND GLOBES guidance: use real coastlines and country geometry, never invented or hand-transcribed outlines, coordinate arrays, decorative blobs or a blank sphere. SVG paths generated from authentic geographic data are allowed, as is authentic user-supplied GeoJSON; non-geographic maps and explicitly requested schematics remain valid. Load the atlas in the page's JavaScript, not in Python or a scratchpad. During repairs, preserve the real geography and requested views and interactions while fixing the failure; this also applies when no skill recall is needed. Keep library names and loading details out of the user-facing brief.

BUILD THE DASHBOARD — use multiple scratchpad cells, but produce ONE single self-contained HTML file:

Before the first write, call `create_artifact(type="html-app", name=..., description=..., primary="index.html")` (with `display="inline"` for a chart shown inside the chat reply, or `display="widget"` for a page that collects the user's input there: see INLINE AND WIDGET at the end) and use the returned `<artifact_path>` for every file you write. All paths below referring to "the output directory" mean `<artifact_path>`. The final dashboard MUST be a single .html file with all data, CSS, JS and images inlined, with no exception: the web app preview shows only the page itself, so a separate file next to it (`dashboard_data.js`, `logo.png`) does not load there, whatever its size. Build every page this way so it renders on every deployment. Keep the whole page under 8 MB; a larger page cannot be checked in a browser.

  REROUND DISCIPLINE (critical — most "round-cap exhaustion" failures we've seen on real dashboards come from drifting off one or more of these):
  1. ONE scratchpad, ONE name. Pick a name on the first cell (e.g. `dash`) and reuse it for the entire build. Switching names (`build_pres` → `write_html` → `pres1` …) creates *separate isolated environments* — variables in one don't exist in another — and burns rounds on recovery.
  2. WRITE TO DISK INCREMENTALLY. Open the output `.html` once in 'w' mode, then `open(path, 'a')` to append head → body skeleton → each chart section → nav/JS → closing tags. Each cell appends a small chunk you can sanity-check. Do NOT build a single 20KB+ HTML string in memory and write it at the end.
  3. CAP STRING SIZE PER CELL at ~5KB. Large-string scratchpad calls are the single biggest cause of silent failures (the tool occasionally drops the `code` payload on oversized inputs and the cell comes back with an empty-code error, which still counts against the round cap). If a section is too big, split it.
  4. NEVER re-emit the full HTML mid-build. Append deltas, don't re-print the world. Assembly is a one-line concat at the end, not a re-render of everything you've written so far.
  5. KEEP READS SMALL. To verify what landed, `os.path.getsize(path)` or the `read_text_file` tool with `start_line=-20` — never read a multi-KB HTML whole.

  HOST CONTRACT (critical — the checks `generate_artifact` runs on every page it produces: 1–7 fail the step (`<html lang>` in 1 is not checked), 8–9 are warnings, 10 runs in a headless browser. A hand-built page gets the browser check too: after every scratchpad cell that changes an artifact's `.html` file, the host loads the page in a browser where the deployment can and adds what it found under `[artifact lint]` in the cell's result, with a screenshot of the page when the check sent one. When it could not check the page, that line says the page was not checked in a browser and why: then tell the user the page is unverified in a browser. Items 1–9 are not checked for a hand-built page, so verify them yourself before you finish):
  1. A complete HTML document with an explicit `<body>`...`</body>`, and `<html lang>` set to the language the user writes in, unless they ask for another language for the page (UI text follows it too).
  2. `<meta name="viewport" content="width=device-width, initial-scale=1.0">` in `<head>`.
  3. No absolute URL in any `fetch()` call except a data file from `https://cdn.jsdelivr.net/npm/<package>@<exact version>/...`; a request to any other host fails once the page is published. A map loads its atlas, `https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json`, with `d3.json(...)` (item 9). No root-relative path either, in `src`, `href`, `fetch()` or `new EventSource()`: a value that starts with a single `/` (`/logo.png`, `/api/export`) breaks in the app preview and once published, because the page is served under a path prefix that a path from the root drops. Use a relative path for a link to another page (`details.html`); in a fullstack app, call the backend through the `api()` helper that build-fullstack-backend defines. `https://`, `//`, `data:`, `blob:`, `#` and `mailto:` values are fine, so `href`/`src` links to external sources stay allowed. Moving the same root-relative path into a JS string does not fix it.
  4. Every opened `<script>` block is closed with `</script>` — an unclosed one silently disables all JS.
  5. The global name `window.__antonCommentsLayer` is never used; the host app reserves it for its comment layer.
  6. No universal `* { ... !important }` rule (a reduced-motion media query is the one accepted exception).
  7. Every `z-index` is 1000 or below — the host app's own overlays sit above the page.
  8. Significant block containers (`div`, `section`, `table`, `main`, `article`) carry stable `id` attributes — the host app attaches comments to them.
  9. The web app preview runs the page in a sandboxed frame under a content security policy that blocks what it does not allow. Build to that policy so the page renders on every deployment. Scripts load only from `https://cdn.jsdelivr.net/npm/<package>@<exact version>/...`: pin an exact version, since a floating `@4`, `@5` or `@latest` cannot be checked. Styles are inline only (`<style>` or `style=`): no external stylesheet, so no Leaflet CSS and no web fonts. No external images or map tiles: embed an image as a `data:` URI. No `eval`/`new Function` and no web workers. The default external resources are the two CDN scripts named under Visual design (Tailwind, ECharts); another library only when the user asked for it, from jsDelivr at an exact version. For a map or a globe, use a tested recipe: d3@7.9.0 with topojson-client@3.1.0 and world-atlas@2.0.2, loading `countries-110m.json` with `d3.json` (item 3) and turning it into GeoJSON with `topojson.feature(world, world.objects.countries)`. Draw that GeoJSON with d3 into an `<svg>`; or with ECharts 5.5.1 through `echarts.registerMap(name, geoJson)` and a `map` series; or on a globe with globe.gl@2.34.2 with `polygonsData`. Also tested: three@0.169.0 as an ES module, 3d-force-graph@1.73.4. For motion, use CSS transitions and keyframes, the Web Animations API (`element.animate()`), or the animation ECharts and d3 already have; load no animation library. Leaflet, raster tiles, MapLibre (it needs a worker) and echarts-gl (it needs `eval`) break that policy, so they do not render in the web app preview.
  10. Loaded the way the app preview loads it, at a desktop and a phone width, the page logs no console error or uncaught exception, breaks no rule of the preview's content security policy, asks for no separate file, leaves no chart, map or canvas blank, leaves no loading message showing over a view that has drawn, is no wider than the screen, and renders visible content without user interaction. Start every view through this one helper: each chart, map, globe or other view that runs setup code, on load or on its tab's first visit. Put the helper alone in the page's first inline `<script>`: `const viewSetups = new Map(); function defineView(name, setup) { viewSetups.set(name, viewSetups.has(name) ? () => { throw new Error('two scripts defined this view'); } : setup); } function runView(name, el) { const fail = (err) => { const why = err && err.message ? err.message : String(err); if (el) { el.setAttribute('data-view-failed', name); el.textContent = name + ' could not load.'; } console.error(name + ' failed: ' + why); if (typeof reportError === 'function') reportError(new Error(name + ': ' + why)); }; const setup = viewSetups.get(name); if (typeof setup !== 'function') { fail(new Error('no script defined this view')); return; } try { const done = setup(el); if (done && typeof done.then === 'function') done.catch(fail); } catch (err) { fail(err); } }`. Give each view an inline `<script>` of its own that holds only its setup, registered under the view's title as the page shows it. Give each view a title no other view on the page uses: two scripts that register one title fail both views. The setup takes the view's element and may be `async`: `<script>defineView('Shipment map', (el) => { /* draw the map */ });</script>`. Declare nothing else at a view script's top level: every function and constant the view uses goes inside its setup. A page's scripts share one global scope, so two views' top-level `function render` silently swap, and one view draws the other's chart with no error. Shared code, such as the data and formatters, goes in scripts before the views'. The page's last script starts each view by its title once the page has parsed, on load or on its tab's first visit: `addEventListener('DOMContentLoaded', () => { runView('Shipment map', document.getElementById('world-map')); });`. A `<script type="module">` runs after every classic script, so a view it registers exists only by then. Keep the functions the page's HTML calls, such as a tab's `showPage`, outside that listener, where an `onclick` attribute finds them. The browser parses each `<script>` on its own, so a syntax error in one view's script stops only that view: it shows a plain-text message in its place, and every other view still draws. A view whose setup throws or whose promise rejects fails alone the same way. Never put two views' code in one script, and never pass a setup function to `runView`. Await or return every promise the setup starts, such as the atlas's `d3.json(...)`. Don't end it with a `.catch` of your own: the helper must see the failure to name and mark the view. Keep the helper as written: its `console.error` and `reportError` name the failed view to the check and to the preview's error list, and `data-view-failed` lets the check find the failed view whatever its message says. A view's own `try`/`catch` that only writes a message hides the error from the check. Log with `console.error` only a view that failed to draw. An expected failure stays silent or uses `console.warn`: the storage helper in item 11 catches a throw on every load in the preview frame, and a `console.error` there fails the check.
  Not checked by the pipeline, but the page breaks inside the frame without them:
  11. BROWSER STORAGE (critical): `localStorage` / `sessionStorage` / `indexedDB` / `document.cookie` may be unavailable — the in-app preview runs the page in a sandboxed frame. Never touch them directly during top-level initialisation: a throw there aborts the script before any listener is bound, and the page renders with every control dead. Go through one guarded helper with an in-memory fallback.
  12. LINKS: the page runs inside a frame, in the app preview and on the published link alike. `target="_top"` and `target="_parent"` do not work; give a link to another artifact or an external site `target="_blank"`, or it opens inside the frame.

  SECURITY (critical): Dashboards may be published to the web. NEVER embed API keys, tokens, passwords, connection strings, or any credentials in the HTML, JS, or inline data. Fetch data in scratchpad cells using credentials from environment variables, then serialize only the resulting data into the dashboard. If the user explicitly asks to embed a credential (e.g. for a live-updating dashboard), warn them that publishing will expose it and get confirmation before proceeding.

  Build the parts in separate cells, then assemble at the end:

  CELL 1 — Serialize data to a JS string variable (programmatic, no HTML):
  Serialize all computed data (dataframes, metrics, KPIs) into a Python string. Build a Python dict with keys like "kpis", "tables", "charts" — each containing the relevant data. Convert DataFrames with df.to_dict(orient='records'). Use json.dumps(data, default=str) to handle dates, Decimal, numpy types. Store as a Python variable: `data_js = 'const D = ' + json_string + ';'` — do NOT write to a separate file.

  CELL 2 — Build CSS + HTML structure as a Python string variable:
  Write the HTML head (styles, CDN script tags) and body structure (header, KPIs, chart divs, tabs, tables) as a Python string variable `html_body`. This cell builds the template.

  CELL 3+ — Build JS chart rendering logic as Python string variables:
  Store item 10's helper as `js_helper`. Write each view's setup as its own string, one `defineView(...)` call per view as item 10 says, views behind tabs included, and collect them in a list `view_js`. Code the views share, such as formatters, goes in `js_shared`. Write the code that populates tables, handles tabs and starts each view by its title as `js_start`: the `runView` calls for the views shown on load go inside `addEventListener('DOMContentLoaded', ...)`, and the tab functions stay outside it. Split across multiple cells if needed to avoid token limits.

  FINAL CELL — Assemble and write the HTML file:
  Give each part a `<script>` of its own, in this order: `scripts = ''.join(f'<script>{js}</script>' for js in [js_helper, data_js, js_shared, *view_js, js_start])`, then `html = html_body.replace('</body>', scripts + '</body>')`. Never join two views' code into one `<script>`: a syntax error in it would stop both.

  SELF-CONTAINED OUTPUT (critical):
  Inline everything, however large: CSS in `<style>`, JS in `<script>`, data as JS variables, images as `data:` URIs. A big dataset goes into the page as a `const D = ...;` script, not into a `dashboard_data.js` beside it, and an image into an `<img src="data:...">`, not a `screenshot.png` beside it. The web app preview loads only the page itself, so a separate file does not show there. If the data makes the page too heavy, aggregate it in the cell first instead of splitting it out.

  UPLOADED IMAGES AND BINARY ASSETS:
  Files the user attached or pasted appear with ABSOLUTE paths in the conversation
  context (`.cowork/files/<uuid>/<name>` in the app, `.anton/uploads/clipboard_*.png`
  for a CLI paste). Use the path you were given — never guess one or scan directories.
  If the user refers to a file whose path you cannot see, ask them to attach it
  to the conversation — never ask for a filesystem path.

  Never point the HTML at that original path: it sits outside `<artifact_path>`, and
  the web app preview loads only the page itself. Embed the image instead, as a `data:` URI,
  in one scratchpad cell:

      import base64, mimetypes
      from pathlib import Path
      mime = mimetypes.guess_type(src)[0] or "application/octet-stream"
      b64 = base64.b64encode(Path(src).read_bytes()).decode()
      img_src = f"data:{mime};base64,{b64}"

  A photo of several MB makes the page heavy: shrink it first (Pillow, a smaller
  width, JPEG) when the page would pass a few MB.
  For uploaded DATA files (CSV/JSON) do not use a data: URI — read them in CELL 1 and
  inline the result as a JS variable like any other dataset.

  WHY: (1) The web app preview loads only the page itself: a separate file does not reach it there. (2) Splitting the build across cells catches JS/CSS errors early: if a cell has a syntax issue in a string, you'll see it before the final assembly. (3) Large datasets in single cells timeout. (4) Self-contained files can be published to the web via /publish without missing assets.

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
- STYLING: Tailwind CSS is the recommended way to style the page. Load it once in `<head>` via CDN: `<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.3.3"></script>` and use utility classes for layout, spacing, typography and colour (`bg-[#0d1117] text-[#e6edf3]` for the theme). Hand-written CSS in a `<style>` block stays allowed wherever utilities fall short — keyframes, chart containers, complex selectors — or when the page is too small to need a framework. No other CSS or JS library unless the user asked for one (host contract item 9): the two CDN scripts named in these rules (Tailwind, ECharts), at exactly the versions shown, are the default external resources. Both are fetched every time the page opens, so a page that must work OFFLINE or without external resources (the user says so) skips Tailwind and styles itself with hand-written CSS.
- ALWAYS use Apache ECharts for interactive charts. Load it via CDN: `<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>`. No Python dependencies needed — just write the HTML with inline JS. Use ECharts' built-in dark theme: `echarts.init(dom, 'dark')`, then customize colors to match #0d1117 background.
- NEVER use Plotly, matplotlib, or other charting libraries unless the user explicitly asks.

Chart type by purpose (the same rule `generate_artifact` gives its generators):
- Line for change over time; bar for comparing categories (horizontal bars when labels are long); stacked bar for composition across categories; scatter for the relation between two measures.
- Pie or donut ONLY for a simple part-of-a-whole with few slices — never for comparison or change over time.
- A value with no dimension is a KPI card, not a chart. Titles, legends and axis labels name what the data IS. No chart junk: 3D, gradients, decorative markers.

Axes and units (critical — a number without a unit is a guess for the reader):
- Every axis label names the measure AND its unit (`Revenue, USD`, `Latency, ms`, `Share, %`).
- Every number in a tooltip, axis or KPI goes through a `formatter`: thousands separators, fixed decimals, currency or percent sign, unit suffix — never a raw float.
- One date format per page, consistent between axes and tooltips. Say when a value is estimated, sampled or converted, and from which unit.

Line smoothing (critical — smooth: true misrepresents volatile data):
- DEFAULT: `smooth: false` on ALL line series. Straight segments between data points are the honest representation — they show actual volatility, drawdowns, and inflection points.
- EXCEPTION: Use `smooth: true` ONLY for cumulative/monotonic series (cumulative returns, running totals, growth curves) where the trend matters more than point-to-point moves.
- Decision heuristic: Does the line ever reverse direction meaningfully? If yes → smooth: false. Is it a running sum, cumulative metric, or long-horizon trend? → smooth: true is acceptable.
- Line widths: 2.5 for hero/primary lines, 1.5 for multi-line comparisons, 1 for secondary/reference lines.

Chart readability (critical — labels must NEVER overlap):
- Use `axisLabel: { rotate: -45 }` or `{ rotate: 45 }` on crowded axes. Set `grid: { containLabel: true }` so labels never clip. Use `legend: { type: 'scroll', bottom: 0 }` to place scrollable legends below the chart. For pie/donut charts use `label: { show: true, position: 'outside' }` with `labelLayout: { hideOverlap: true }`. For bar charts with many categories, use horizontal bars (`yAxis` as category) or abbreviate labels with `axisLabel: { formatter }`. Always configure rich `tooltip` with `formatter` functions for precise value display on hover. Use `dataZoom` for time series so users can zoom into ranges.

Multi-tab / multi-view dashboards (critical — charts fail silently on hidden containers):
- ECharts, Chart.js, and Plotly all render nothing when called on a container with `display: none` or 0×0 dimensions — no error, no warning, just a blank chart. NEVER call `echarts.init()` inside `DOMContentLoaded` for tabs/pages that start hidden.
- Initialize charts lazily, gated on first visibility: in the tab-click handler, check a `Set` of already-rendered tabs and call the page's init function only on first visit. Example pattern: `const _rendered = new Set(['overview']); function showPage(name) { /* toggle classes */ if (!_rendered.has(name)) { _rendered.add(name); initChartsFor(name); } }` — only the default-visible page initializes on load. `initChartsFor(name)` starts each view on that tab by its title with `runView(title, el)` (item 10), so a view whose script fails on a tab fails alone and the tab's other views still draw.

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
- Multi-card sections use `grid-template-columns: repeat(auto-fit, minmax(min(360px, 100%), 1fr))` (or 300px on dense layouts; the `min()` keeps one card inside a 390 px phone). This lets the browser reflow to single-column on narrow viewports without a media query — cards stack vertically instead of getting squashed into unreadable columns.
- Chart containers use `width: 100%` and `height: min(420px, 60vh)` (NOT fixed pixel widths). For each ECharts instance, register a window resize hook so it refits: `window.addEventListener('resize', () => myChart.resize());` — without this, rotating a phone or resizing the window leaves charts the wrong size.
- A table of more than 4 columns shows each row as a card below 640 px: every `td` carries `data-label="<column name>"`, and a media query hides the `thead` and sets `display: block` on `table`, `tbody`, `tr` and `td` (keep `tr[hidden]` hidden), with `td::before { content: attr(data-label) }`. Squeezed into a phone, a wide table's columns end up a letter or two wide, and a sideways-scrolling one hides most of its columns.
- A narrower table wraps in `<div style="overflow-x: auto;">`. Do NOT set fixed table widths.
- Numbers, ids and dates never break inside a value: `white-space: nowrap` on those cells. Never `overflow-wrap: anywhere` or `word-break: break-all` on table cells.
- A map fits its projection to the plotted points (with a margin), not the whole world, when they sit in one region. At phone width its labels and legend stay inside the view and no smaller than 11 px: shorten or drop labels that would overlap.
- Default to one column on narrow viewports unless the user explicitly asks for a fixed multi-column layout (e.g. for a printable PDF).

INLINE AND WIDGET (display="inline" or "widget") — a small page shown live inside the chat reply:
Two kinds, one set of rules. `display="inline"` is an interactive view worth keeping or sharing: a chart with range tabs or toggles, a small dashboard. `display="widget"` is a page whose point is the user's input in this conversation: pick, confirm, approve, rank, fill in, choose between options. A widget is not a deliverable: the app offers no Share or publishing for it and keeps it out of the Artifacts list. If the user later wants to keep or share a widget, `update_artifact(slug=..., display="inline")`: same slug, nothing rebuilt.
Register it with `create_artifact(type="html-app", primary="index.html", display="inline" or "widget", ...)` and write it yourself; generate_artifact builds full pages. Everything above still holds (one self-contained file, the HOST CONTRACT with its view helper, SECURITY, chart rules) except where these rules differ:
- FITS IN A MESSAGE: one chart and its controls (range tabs such as 1D/5D/1M, a series toggle, a select), or two small linked views. No page header, KPI rows, multi-section layout, tabbed pages or long tables. The insight list is one line. Build it in one or two cells; REROUND DISCIPLINE is for full dashboards.
- HEIGHT: the host measures the page and sizes the frame to it, so the page sizes to its content: aim for 600 px tall or less. No `100vh`, `height: 100%` on `html`/`body`, `position: fixed`, or a wrapper with its own scrolling (`overflow: auto`/`scroll`). `body { margin: 0; }`, chart container `width: 100%; height: 320px` (420 px at most), and the resize hook from Responsive layout.
- THEME: the host sets `data-theme="light"` or `"dark"` and `color-scheme` on `<html>`, and changes them live when the app's theme changes. Style from CSS custom properties, not the fixed #0d1117 theme: light tokens on `:root`, dark ones on `:root[data-theme="dark"]`, and the same dark tokens under `@media (prefers-color-scheme: dark) { :root:not([data-theme]) { ... } }` for the moment before the host sets it. Keep the page background transparent so it sits in the reply. Start ECharts with `echarts.init(el, dark ? 'dark' : null)` and `backgroundColor: 'transparent'`, reading colours from the tokens (`getComputedStyle(document.documentElement).getPropertyValue('--fg')`). Watch the attribute with `new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })`, where `redraw` disposes the chart and starts it again in the new theme with the current range and toggles.
- NO NETWORK: the frame makes no requests, so all data is inline as `const D = ...;` and nothing calls `fetch`, `XMLHttpRequest`, `d3.json` or `EventSource`. No web fonts, external images, map atlases or tiles, and no Tailwind: style with a `<style>` block. The one external resource is the pinned ECharts script `https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js`, the same `<script src>` every html-app preview loads; no other library.
- SMALL DATA: aggregate in the scratchpad to what the chart shows, a few hundred points per view at most, with every range a tab offers computed ahead, and keep the file under 200 KB. Larger data, or a page the user will open on its own, is a full artifact without `display`.
- No `localStorage`. A chart starts in its default view every time; a page the user works in keeps its state through `window.anton` (below).
- WIDGETS (review, pick, edit or confirm several items: people to contact, rows to approve, a plan to reorder): same rules, with checkboxes, fields, buttons or drag handles instead of a chart, and no ECharts unless it also shows a chart. Give it a visible title that says what the user is deciding. The host injects a small page API (an inline chart may use it too):
  - The host captures the page's form controls (inputs, selects, sliders, checkboxes) and their values on its own, so label every control clearly with a `<label for>`; use `anton.setState` for any choice that is not a form control (a clicked chart bar, a drag order).
  - `anton.setState(obj)`: saves the page state (JSON, at most 32 KB) as `state.json` in the artifact folder. Call it on every edit.
  - `anton.getState()` returns the saved state or `null`; `anton.onState(cb)` calls `cb(state)` when the saved state arrives or changes. Render from the data in the page first, then apply the state; ignore keys that no longer match the data (a newer revision may have changed the items).
  - `anton.submit(obj, message)`: sends the user's confirmation to the chat as a new user turn tagged with this artifact; both arguments are optional (`obj` defaults to the current state). Give the page one clear confirm button that calls it.
  - Guard every call with `if (window.anton)`: the full preview and the browser check have no API, and the page must still render there.
  - When the conversation keeps working on the same list, rewrite this page under the same slug (a new revision) instead of creating another artifact; the chat shows only the newest copy.
- VOICE: the user may be on a voice call. A voice assistant reads the page's title, visible text and current state, and can set the state or submit for the user. So:
  - Keep the state a small, self-describing JSON with readable keys and labels: `{"people":[{"name":"Ana Li","picked":true}]}`, not `{"p":["u_83f2"]}`. Someone reading only the state knows what the user chose.
  - Render fully from `onState`: any valid state can be pushed from outside at any time, so every visible control follows the state it is given, not only the user's clicks.
  - Make the `submit` message short and human: "Contact Ana Li and Li Wei", not a dump of the state.
- Example widget, a checklist in that state shape (the view helper from item 10 is left out for brevity; a real page still starts its view through it):
  ```html
  <h3>Who should we contact?</h3>
  <ul id="list"></ul><button id="ok">Confirm</button>
  <script>
  const D = [{name: "Ana Li", company: "Acme"}, {name: "Li Wei", company: "Globex"}];
  let state = {people: D.map(d => ({name: d.name, picked: true}))};
  const api = window.anton;
  function render() {  // DOM nodes and textContent: names are data, never HTML
    document.getElementById("list").replaceChildren(...state.people.map((p, i) => {
      const d = D.find(x => x.name === p.name), li = document.createElement("li");
      const label = document.createElement("label"), box = document.createElement("input");
      box.type = "checkbox"; box.dataset.i = i; box.checked = p.picked;
      label.append(box, ` ${p.name}, ${d.company}`); li.append(label); return li;
    }));
  }
  function apply(s) {  // any valid state, from the user or pushed from outside: render all of it
    if (!s || !Array.isArray(s.people)) return;
    const picked = new Map(s.people.map(p => [p.name, !!p.picked]));
    state = {people: D.map(d => ({name: d.name, picked: picked.has(d.name) ? picked.get(d.name) : true}))};
    render();
  }
  document.getElementById("list").addEventListener("change", e => {
    state.people[e.target.dataset.i].picked = e.target.checked;
    if (api) api.setState(state);
  });
  document.getElementById("ok").addEventListener("click", () => {
    const names = state.people.filter(p => p.picked).map(p => p.name);
    if (api) api.submit(state, names.length ? `Contact ${names.join(" and ")}` : "Contact no one");
  });
  render();
  if (api) { apply(api.getState()); api.onState(apply); }
  </script>
  ```
