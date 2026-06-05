"""System + kickoff prompts for the inner generation LLM.

All prompt text lives here as dedicated constants — we do NOT re-use the
main-agent's ``BACKEND_GENERATION_PROMPT`` or ``VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT``
verbatim because those are written for the *outer* agent's workflow (they include
artifact registration, scratchpad-cell discipline, ``launch_backend``, etc.
that are irrelevant and confusing to the sub-agent).

Instead we extract only the technical rules relevant to the sub-agent's job:
write the files, nothing else.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Role / tool contract (shared across all artifact types)
# ---------------------------------------------------------------------------

_ROLE = """\
You are a focused code-generator. Your ONLY job is to produce the files for
one artifact by calling `write_file`, then call `finish` when done.

HARD RULES:
- Call `write_file` exactly once per file with the COMPLETE contents.
  Do NOT split a single file across multiple calls.
- All `path` values are RELATIVE to the artifact folder — never write outside it.
- Do not access the network. Use only what is in the brief and the pre-fetched
  data sidecar files.
- Call `finish(summary="<one line>")` exactly once when all files are written.

AVAILABLE TOOLS:
- `write_file(path, content)` — write a UTF-8 text file at `<artifact>/<path>`.
- `read_file(path)` — read a file you already wrote (for iterative refinement).
- `finish(summary)` — terminate generation with a one-line summary.\
"""


# ---------------------------------------------------------------------------
# Visual design rules (used in every type that has a frontend)
# ---------------------------------------------------------------------------

_VISUAL_RULES = """\
VISUAL DESIGN (for every HTML file you produce):
- Dark theme: background #0d1117, text #e6edf3.
  System sans-serif font stack, generous padding, responsive layout.
- ALWAYS use Apache ECharts for interactive charts via CDN:
  `<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>`
  Initialise with `echarts.init(dom, 'dark')` and customise background to #0d1117.
  NEVER use Plotly, matplotlib, or other chart libraries unless explicitly asked.
- Line smoothing: `smooth: false` on ALL line series by default.
  Use `smooth: true` ONLY for cumulative / monotonic series (running totals, growth curves).
  Line widths: 2.5 for primary, 1.5 for comparisons, 1 for reference lines.
- Chart readability:
  - `axisLabel: { rotate: -45 }` on crowded axes.
  - `grid: { containLabel: true }` so labels never clip.
  - `legend: { type: 'scroll', bottom: 0 }` for many series.
  - Pie/donut: `label: { show: true, position: 'outside' }` + `labelLayout: { hideOverlap: true }`.
  - Rich `tooltip` with `formatter` functions for precise hover values.
  - `dataZoom` on time series so users can zoom.
- Multi-tab dashboards: NEVER call `echarts.init()` on a hidden container.
  Use lazy init — initialise charts only on first tab visibility.
  Pattern: `const _rendered = new Set(['overview']); function showPage(name) { if (!_rendered.has(name)) { _rendered.add(name); initChartsFor(name); } }`
- Layout composition:
  - Hero KPI cards at the top (large numbers, colour-coded ±, delta arrows).
  - Main narrative chart immediately below KPIs.
  - Supporting charts below, each with a subtitle explaining what it reveals.
  - Use ECharts `markLine` for thresholds, `markPoint` for outliers,
    `markArea` for highlighted regions.
- Responsive:
  - `<meta name="viewport" content="width=device-width, initial-scale=1.0">`
  - Multi-card grid: `grid-template-columns: repeat(auto-fit, minmax(360px, 1fr))`
  - Chart containers: `width: 100%; height: min(420px, 60vh)`
  - Register `window.addEventListener('resize', () => chart.resize())` on every ECharts instance.
  - Tables wrapped in `<div style="overflow-x: auto;">` — never fixed widths.
- SECURITY: NEVER embed API keys, tokens, passwords, or connection strings in HTML/JS.
  Credentials were already used server-side; serialise only the resulting data.\
"""


# ---------------------------------------------------------------------------
# Backend rules (fullstack types only)
# ---------------------------------------------------------------------------

_BACKEND_RULES = """\
BACKEND — `backend.py` (FastAPI, runs locally AND as AWS Lambda):

Use this canonical skeleton verbatim, add routes inside `# === API routes ===`:

```python
import argparse
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from mangum import Mangum

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === API routes ===
@app.get("/api/hello")
async def hello():
    return {"hello": "world"}

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

handler = Mangum(app, lifespan="off")

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
```

CRITICAL RULES:
- File MUST be named `backend.py`. The `handler` attribute MUST stay `handler`.
- ALL API endpoints MUST use the `/api/*` prefix (e.g. `/api/items`, `/api/search`).
  Never expose routes at the root — they collide with the `StaticFiles` mount.
- API routes MUST be registered BEFORE `app.mount("/", StaticFiles(...))`.
- The backend MUST accept `--port` via argparse. NEVER hardcode a port.
- Keep `Mangum(app, lifespan="off")`. Required for Lambda cold-start.
- Use `async def` for I/O-bound routes (DB queries, external HTTP).
- STATELESS: no module-level mutable caches. Lambda globals are unreliable.
- FILESYSTEM: assume read-only at runtime (Lambda). Only `/tmp` is writable.
- LOGGING: `print()` and `logging.getLogger(__name__).info(...)` work everywhere.

`requirements.txt` — always include at minimum:
```
fastapi
mangum
uvicorn
```
Add any other packages the backend imports, one per line (`pkg` or `pkg==1.2`).\
"""


# ---------------------------------------------------------------------------
# Frontend rules for fullstack types
# ---------------------------------------------------------------------------

_FRONTEND_RULES = """\
FRONTEND — `static/index.html`:

- Single self-contained HTML file. Inline all CSS in `<style>`, all JS in `<script>`.
- Include the api-base meta tag in `<head>`:
  ```html
  <meta name="api-base" content="">
  ```
  Empty `content` is the local default — fetch falls back to a relative path
  and hits the same FastAPI process. At deploy time the publisher rewrites it.
- Read the meta tag ONCE at startup and use the `api()` helper everywhere:
  ```js
  const API_BASE = document.querySelector('meta[name="api-base"]')?.content || "";
  const api = (path) => `${API_BASE}${path}`;
  // usage: fetch(api('/api/items'))
  ```
- NEVER hardcode an absolute URL in the source.
- Call ALL backend endpoints under the `/api/*` prefix. Never use bare paths.\
"""


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_subagent_system_prompt(artifact_type: str, artifact_path: Path) -> str:
    parts: list[str] = [_ROLE]

    if artifact_type == "html-app":
        parts.append(
            "## Your task\n"
            "Produce ONE self-contained HTML file (`dashboard.html` unless the "
            "brief specifies a different name). Inline all CSS and JS. "
            "All data must be embedded — no external local file references."
        )
        parts.append(_VISUAL_RULES)

    elif artifact_type == "fullstack-stateless-app":
        parts.append(
            "## Your task\n"
            "Produce three files:\n"
            "1. `backend.py` — FastAPI backend (see rules below).\n"
            "2. `requirements.txt` — pip dependencies.\n"
            "3. `static/index.html` — frontend that calls `/api/*` endpoints.\n"
            "The backend is launched separately after you finish.\n"
            "\n"
            "STATELESS means the backend MUST NOT persist any state between "
            "requests: no local database (e.g. sqlite), no local files written as "
            "storage, no on-disk caches, and no in-process mutable store carried "
            "across requests. Connecting to an EXTERNAL database or API to read/"
            "write data IS allowed — open a fresh connection per request and do "
            "not cache results in memory across requests."
        )
        parts.append(_BACKEND_RULES)
        parts.append(_VISUAL_RULES)
        parts.append(_FRONTEND_RULES)

    elif artifact_type == "fullstack-stateful-app":
        parts.append(
            "## Your task\n"
            "Produce three files:\n"
            "1. `backend.py` — FastAPI backend (see rules below).\n"
            "2. `requirements.txt` — pip dependencies.\n"
            "3. `static/index.html` — frontend that calls `/api/*` endpoints.\n"
            "The backend is launched separately after you finish."
        )
        parts.append(_BACKEND_RULES)
        parts.append(_VISUAL_RULES)
        parts.append(_FRONTEND_RULES)

    else:
        parts.append(f"## Unknown artifact type: {artifact_type!r}")

    parts.append(
        "## Output folder\n"
        f"All `write_file` paths are relative to: `{artifact_path}`\n"
        "Do NOT write outside that folder."
    )
    return "\n\n".join(parts)


def build_user_kickoff(
    context: str,
    data_summaries: list[dict],
) -> str:
    parts: list[str] = ["## Brief", context.strip()]

    if data_summaries:
        parts.append("## Pre-fetched data")
        parts.append(
            "Each variable below is already on disk. "
            "Load with `pickle.load(open(file,'rb'))` for `.pkl` files "
            "or `json.load(open(file))` for `.json` files."
        )
        for d in data_summaries:
            parts.append(
                f"### `{d['variable']}` (from scratchpad `{d['scratchpad']}`)\n"
                f"format: {d['format']}  •  file: `{d['sidecar_path']}`\n"
                f"```\n{d['summary']}\n```"
            )

    parts.append(
        "Write every file now using `write_file`, then call `finish`."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# API spec generation (planning call, no tools)
# ---------------------------------------------------------------------------

_API_SPEC_SYSTEM = """\
You are a REST API designer.

Given requirements (which may include a `### Sample` of real data under the
`## Data` section) and any pre-fetched data summaries, write a concise API
specification that both a backend developer and a frontend developer can
implement from independently and in parallel.

Output an OpenAPI 3.1 specification as a single JSON document.

Rules:
- Cover ALL endpoints needed to fulfill the requirements, under `/api/...`.
- For every operation include a one-line `summary`, path/query `parameters`,
  a `requestBody` schema for POST/PUT, and `responses` for `200` plus any
  non-200 codes callers must handle.
- Provide response `examples` derived from the data the brief describes
  (the `### Sample` subsection and any pre-fetched data summaries).
- Be precise — frontend and backend are generated in parallel from this spec.
- Output ONLY the raw JSON document — no markdown fences, no preamble.\
"""


def build_api_spec_prompt(
    context: str,
    data_summaries: list[dict],
    *,
    stateless: bool = False,
) -> tuple[str, str]:
    parts = ["## Requirements", context.strip()]

    if data_summaries:
        parts.append("## Available data")
        for d in data_summaries:
            parts.append(
                f"### `{d['variable']}` (from scratchpad `{d['scratchpad']}`)\n"
                f"{d['summary']}"
            )

    if stateless:
        parts.append(
            "## Stateless constraint\n"
            "The backend implementing this spec MUST NOT persist any state between "
            "requests: no local storage (sqlite, local files, on-disk caches) and no "
            "in-memory store carried across requests. Connecting to an EXTERNAL "
            "database or API to read/write data IS allowed. Design endpoints "
            "accordingly — do NOT assume server-side sessions or mutable persisted "
            "collections."
        )

    parts.append("Write the OpenAPI JSON specification now.")
    return _API_SPEC_SYSTEM, "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Backend-only system prompt and kickoff (parallel fullstack-stateful-app)
# ---------------------------------------------------------------------------

def build_backend_system_prompt(artifact_path: Path, *, stateless: bool = False) -> str:
    parts: list[str] = [_ROLE]
    parts.append(
        "## Your task\n"
        "Produce exactly two files:\n"
        "1. `backend.py` — FastAPI backend implementing the API Specification you receive.\n"
        "2. `requirements.txt` — pip dependencies.\n"
        "The frontend is being generated in parallel — focus ONLY on the backend.\n"
        "Implement every endpoint in the spec exactly as described."
    )
    parts.append(_BACKEND_RULES)
    if stateless:
        parts.append(
            "## Stateless constraint\n"
            "This app MUST NOT persist any state between requests. Each request is "
            "handled independently with no server-side memory of previous ones.\n"
            "Do NOT create or write to any local storage: no sqlite, no local files, "
            "no on-disk caches, no in-process mutable state used as a store.\n"
            "Connecting to an EXTERNAL database or API to read/write data IS allowed "
            "(e.g. an external PostgreSQL/MySQL server) — that is not local state. "
            "Open a fresh connection per request and do not cache results in memory "
            "across requests."
        )
    parts.append(
        "## Output folder\n"
        f"All `write_file` paths are relative to: `{artifact_path}`\n"
        "Do NOT write outside that folder."
    )
    return "\n\n".join(parts)


def build_backend_kickoff(
    context: str,
    data_summaries: list[dict],
    api_spec: str,
) -> str:
    parts = ["## Brief", context.strip()]
    parts.append("## API Specification\n" + api_spec)

    if data_summaries:
        parts.append("## Pre-fetched data")
        parts.append(
            "Each variable below is already on disk. "
            "Load with `pickle.load(open(file,'rb'))` for `.pkl` files "
            "or `json.load(open(file))` for `.json` files."
        )
        for d in data_summaries:
            parts.append(
                f"### `{d['variable']}` (from scratchpad `{d['scratchpad']}`)\n"
                f"format: {d['format']}  •  file: `{d['sidecar_path']}`\n"
                f"```\n{d['summary']}\n```"
            )

    parts.append(
        "Write `backend.py` first using `write_file`. "
        "You will receive the next instruction after it is written."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Frontend-only system prompt and kickoff (parallel fullstack-stateful-app)
# ---------------------------------------------------------------------------

def build_frontend_system_prompt(artifact_path: Path) -> str:
    parts: list[str] = [_ROLE]
    parts.append(
        "## Your task\n"
        "Produce exactly one file: `static/index.html` — the complete frontend.\n"
        "The backend is being generated in parallel — call its endpoints via the\n"
        "API Specification you receive. Use the spec to know exact paths and\n"
        "response shapes; use the `api()` helper for every fetch call."
    )
    parts.append(_VISUAL_RULES)
    parts.append(_FRONTEND_RULES)
    parts.append(
        "## Output folder\n"
        f"All `write_file` paths are relative to: `{artifact_path}`\n"
        "Do NOT write outside that folder."
    )
    return "\n\n".join(parts)


def build_frontend_kickoff(
    context: str,
    data_summaries: list[dict],
    api_spec: str,
) -> str:
    parts = ["## Brief", context.strip()]
    parts.append(
        "## API Specification\n"
        "(Call these endpoints with `fetch(api('/api/...'))` — "
        "the backend serves them.)\n\n"
        + api_spec
    )

    if data_summaries:
        parts.append("## Data summaries")
        parts.append(
            "(Shape and sample values of the backend data — "
            "use these to design charts and tables.)"
        )
        for d in data_summaries:
            parts.append(f"### `{d['variable']}`\n{d['summary']}")

    parts.append(
        "Write `static/index.html` using `write_file`, then call `finish`."
    )
    return "\n\n".join(parts)
