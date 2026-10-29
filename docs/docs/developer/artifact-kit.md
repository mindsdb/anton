---
title: Artifact kit and page builder
description: How html-app pages are built — the kit's block API, the runtime that draws pages, the page-builder agent behind generate_artifact, and how to extend or test them.
---

# Artifact kit and page builder

Every `html-app` artifact (reports, dashboards, maps, globes, timelines, briefs, slide decks) is
built by the **page builder**: one focused agent loop that `generate_artifact` runs for pages. The
builder fetches the data in the scratchpad, computes every figure with code, composes the page with
the **artifact kit** and saves it. Fullstack apps still go through the generator pipeline
(brief, PRD, spec, code, launch).

```
main agent ──generate_artifact(name=…, user_request=…)──▶ page builder (kit/builder.py)
                                                        │  tools: scratchpad, recall_skill, finish
                                                        ▼
                                  scratchpad cell: fetch → profile → compute → kit.Page(...) → save
                                                        ▼
                                  index.html = spec JSON (data embedded) + static rendition
                                               + design system CSS + the runtime modules it uses
```

## Why a kit, not free-form HTML

A page that a model writes from scratch every time is slow (thousands of tokens of markup and
script), re-types numbers (transcription errors), and fails in ways that only show in a browser
(blocked CDNs, console errors, blank charts, phone overflow). The kit moves the hard, repeatable
parts into tested code:

- **Correct data by construction.** Blocks take DataFrames; values are embedded straight from the
  data. The model writes a short composition, not the numbers.
- **Rendering that works everywhere.** Every block is pre-tested in the Cowork preview (sandboxed
  srcdoc frame, preview CSP) at desktop and phone sizes; libraries load only from
  `cdn.jsdelivr.net`; a failing view says so in place instead of throwing.
- **Design quality by default.** One design system (tokens, light and dark themes, typography,
  spacing, motion that respects reduced-motion), consistent number and date formatting per locale,
  localized UI strings.
- **Freedom where needed.** `page.echarts(option)` takes any ECharts option (radar, treemap,
  sunburst, funnel, gauge, calendar, …) and `page.html(html, js=...)` takes any custom markup and
  script, with the page's data and theme available as `AK.data(...)`, `AK.fmt(...)`, `AK.palette()`.

## The block API (`anton.core.artifacts.kit`)

| Block | Use |
|---|---|
| `kpis([kit.kpi(...)])` | headline figures with deltas and sparklines |
| `compare(columns, [kit.metric(...)])` | metrics side by side across options/scenarios, with the change |
| `chart(kind, df, x, y, ...)` | bar, hbar, line, area, combo, scatter, pie, donut, heatmap, waterfall; thresholds, reference lines |
| `table(df, columns, ...)` | sort, search, select filters, status pills, inline bars, totals, row detail, linked selection; one card per row on narrow screens |
| `map(*layers)` / `globe(*layers)` | `kit.points`, `kit.routes` (great circles, optional via point), `kit.regions` (choropleth); maps re-center on the data's longitude so routes never break at the date line, and zoom with buttons, keys, double-click or pinch; globes are canvas-drawn, turn slowly and can be dragged |
| `network(links, nodes=..., style=)` | layered node-link diagram or sankey; tiers from a group order or longest path; vertical layout on phones; click or Enter on a node highlights its links (`controls=True` adds buttons for it) |
| `timeline(bars=[...], milestones=[...], markers=[...])` | Gantt lanes per layer (plan vs actual), milestones with a detail panel and stepper, dashed markers, a collapsed list of milestones (`milestone_list=True` opens it) |
| `text`, `callout`, `list`, `image`, `section`, `grid`, `tabs` | layout and narrative; `tabs(style="segmented")` is a scenario switch |
| `Deck` + `slide(...)` | 16:9 slides with title, content, big-number, split and section layouts |

Formats (`format=`): `int`, `number:1`, `percent` (12.5 → 12.5%), `fraction` (0.125 → 12.5%),
`currency`, `currency_compact` ($2.4M), `compact`, `days`, `date`, `date_short`, `date_iso`,
`month`, `datetime`, `datetime_short`, `time`, `text`. Timestamps keep their times of day unless a
date format is chosen (the page warns when that would hide them).

Linked views: give a table and map/globe/chart layers the same `key` and `link`; selecting a row
highlights the same key everywhere (the globe turns to it).

Edits: `page = kit.load(folder)`, `page.outline()`, then `page.after(id)`, `page.before(id)`,
`page.beside(id)`, `page.inside(id)`, `page.update(id, ...)`, `page.remove(id)`, `page.set(...)`,
`page.save(folder)`.

Helpers: `kit.geocode("City, Country")` (offline GeoNames towns; `None` when unsure),
`kit.profile(df, name)` (columns, distinct values with counts, ranges).

`page.save()` prints the outline with its figures (KPI values, chart sums and ranges, table sizes)
and any warnings (missing coordinates, summed duplicates, unknown fields), which the builder reads
before it finishes.

## The page file

`render.render_html(spec)` writes, in order: the stylesheet and the spec as JSON with its data (both
in the head), a static rendition of the page (headings, text, KPI
values, tables) for readers without JavaScript (the runtime replaces it when it boots), then only
the runtime modules the page's blocks use
(`core.js` and `blocks.js` always; `table.js`, `chart.js`, `geo.js`, `flow.js`, `timeline.js`,
`deck.js` on demand). Custom block scripts are emitted as their own `<script>` tags because the
preview's policy forbids `eval`.

Places are stored in one shape whatever the source columns are called (`{"lat": …, "lon": …}`, a
route's `from`/`via`/`to` each its own), so the runtime reads every layer the same way; coordinate
columns a block does not draw are not embedded, which keeps pages small.

## Runtime rules worth knowing

- Continuous motion repaints only small or canvas layers, on an adaptive frame budget: a slow device
  (or a software-rendered browser) gets fewer frames, never an unresponsive page.
- Text that matters (labels, values) is real DOM text; canvas-drawn globe labels are mirrored in a
  visually hidden list for screen readers.
- Views redraw on width changes (ResizeObserver), so tabs and grids that reveal a view size it then.

## The page builder (`kit/builder.py`)

- System prompt: `BUILDER_PROMPT` (workflow, data-correctness rules, design rules, `KIT_REFERENCE`),
  plus the connected data sources (credential variable names only), the skills it may recall and the
  project's rules.
- Kickoff: the request, the page language, the main agent's notes and known data, the artifact
  folder, the existing page's outline when it is an edit, the session's scratchpads, and the bodies of
  skills already recalled in the conversation or written by the user (saves it a round).
- Loop: up to `MAX_ROUNDS` calls on the planning model (or the `ANTON_ARTIFACT_MODEL` route); tools
  `scratchpad`, `recall_skill` (always returns the full body) and `finish` (accepted only once the
  page was saved in this build).
- A page made only of kit blocks skips the per-cell browser check (the runtime is tested); pages
  with custom HTML/JS or raw ECharts options are checked as before.

## Testing

- `tests/test_artifact_kit.py` covers the builder API, page assembly and editing.
- `tests/test_pages_prompt.py` keeps the documented API in `KIT_REFERENCE` in step with the code.
- To see a page the way the Cowork preview shows it, render it in a sandboxed `srcdoc` iframe under
  the preview CSP across widths (320, 390, 768, 1024, 1440 and 1920 px) and look for console errors,
  blank views, clipped labels and horizontal overflow.
