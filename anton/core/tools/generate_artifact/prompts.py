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

from .discovery.notes import EXEC_OUTPUT_MAX

# The body markers are quoted to the model in several places here and parsed in
# `sub_tools.extract_file_body`; every surface reads the one constant, because a
# literal that drifts breaks the protocol silently.
from .sub_tools import FILE_BEGIN_MARKER as BEGIN, FILE_END_MARKER as END

# How much file content fits in one reply, in characters — the unit the model
# can compare against what it is about to write. Derived from the output budget
# in state.py; quoted from the constant so the two cannot drift.
from .state import REPLY_BODY_CHARS


# ---------------------------------------------------------------------------
# Canonical FSM graph — embedded in decision/generation prompts so every LLM
# call understands the whole pipeline and where its step sits. English only.
# ---------------------------------------------------------------------------

FSM_DIGRAPH = """\
digraph artifact_generation {
    rankdir=TB;
    is_data_enough       [shape=diamond, label="Is there enough data to solve the task?"];
    define_required_data [shape=box,     label="Determine the required data"];
    is_possible_to_fetch [shape=diamond, label="Is it possible to fetch the data?"];
    fetch_data_sample    [shape=box,     label="Fetch a data sample"];
    not_enough_data      [shape=ellipse, label="Error: not enough data"];
    make_tech_spec       [shape=box,     label="Write a detailed technical specification (spec.md)"];
    is_fullstack         [shape=diamond, label="Is a backend required? (derived from artifact type)"];
    make_api_spec        [shape=box,     label="Design the REST API specification (openapi.json)"];
    generate_backend     [shape=box,     label="Generate backend in a subagent"];
    verify_backend       [shape=box,     label="Verify and unit-test the backend"];
    generate_frontend    [shape=box,     label="Generate frontend in a subagent"];
    verify_frontend      [shape=box,     label="Verify the frontend"];
    run_app              [shape=box,     label="Launch the application"];
    verify_fullstack     [shape=box,     label="Verify the application is running"];

    // entry node = is_data_enough
    is_data_enough       -> make_tech_spec       [label="yes"];
    is_data_enough       -> define_required_data [label="no"];
    define_required_data -> is_possible_to_fetch;
    is_possible_to_fetch -> fetch_data_sample    [label="yes"];
    is_possible_to_fetch -> not_enough_data      [label="no"];
    fetch_data_sample    -> is_data_enough;
    make_tech_spec       -> is_fullstack;
    is_fullstack         -> make_api_spec        [label="yes"];
    is_fullstack         -> generate_frontend    [label="no"];
    make_api_spec        -> generate_backend;
    make_api_spec        -> generate_frontend;
    generate_backend     -> verify_backend;
    generate_frontend    -> verify_frontend;
    verify_backend       -> run_app;
    verify_frontend      -> run_app;
    run_app              -> verify_fullstack;
}
"""


# ---------------------------------------------------------------------------
# Role / tool contract (shared across all artifact types)
# ---------------------------------------------------------------------------

# Common half: fits both the nodes that write files and the fetch node, which
# does not. Everything about write_file lives in _ROLE_WRITE below.
_ROLE_COMMON = """\
You are a focused, single-purpose worker inside an artifact-generation pipeline.
You do exactly the job your task section describes, then call `finish`.

ALWAYS-AVAILABLE TOOLS:
- `scratchpad(action, name, ...)` — drive a persistent Python scratchpad
  (`exec`, `view`, `dump`, `install`, `reset`, `remove`). Use it to reach the
  real data described in the brief's `## Data` section.
- `finish(summary)` — terminate with a one-line summary.

SCRATCHPAD DISCIPLINE (the same rules the main agent works under):
- The scratchpad starts with a clean namespace — nothing is pre-imported. Put
  every import the cell needs at the top of THAT cell. Re-importing is free and
  makes the cell work even if an earlier one failed.
- Each cell has a hard timeout of 120 seconds. On timeout the process is killed
  and ALL state is lost — variables, imports, loaded data. Keep cells small;
  split anything heavier across cells.
- Always `print(...)` what you want to see: the tool captures stdout, and a bare
  expression at the end of a cell returns nothing.
- Connected data-source credentials arrive as environment variables named
  `DS_<ENGINE>_<NAME>__<FIELD>` — read them from `os.environ`. NEVER read the
  `data_vault` files directly.
- If a cell fails the same way twice, change strategy instead of re-running it:
  different library, different query shape, a smaller batch. Repeating an
  identical failing cell only burns the round budget.

USING DATA:
- The brief's `## Data` section names the scratchpads and cells the main agent
  already used, and what was done in them. `## Data gathered so far`, when
  present, already contains those cells — read it before running anything.
- Use `scratchpad(action="exec", name="<pad>", code=...)` to pull or rebuild the
  data you need (re-query, aggregate, reshape). Provide
  `one_line_description` and `estimated_execution_time_seconds` on every `exec`.
  NEVER create a scratchpad with a new name: reuse the pad named in the brief
  or the PRD — a new name is an isolated empty environment with none of the
  existing variables, imports or connection code, and the call may be rejected.\
"""

# Write half: only for nodes that actually produce files. NOT mixed into the
# fetch node — there the role is immediately followed by "Do NOT write any
# artifact files", and the full _ROLE would contradict that instruction.
_ROLE_WRITE = f"""\
YOUR OUTPUT IS FILES. A file is written in TWO parts of the SAME reply:

  1. the file's content, as plain text between two marker lines;
  2. a `write_file` call naming the path — with NO content argument.

Like this, in one reply:

{BEGIN}
<!DOCTYPE html>
<html lang="en">
…the entire file…
</html>
{END}

…and in the same reply, the tool call `write_file(path="index.html")`.

HARD RULES:
- Write the content as TEXT between the markers. `write_file` takes `path` and
  `mode` only; there is no `content` argument.
- ONE file per reply. A reply carries one body, so a second `write_file` in the
  same reply has nothing to write and is refused.
- Nothing but the file goes between the markers, and the closing marker ends
  the file. If your reply has no closing marker, nothing is written.
- All `path` values are RELATIVE to the artifact folder — never write outside it.
- Call `finish(summary="<one line>")` exactly once when all files are written.
- VERIFICATION IS NOT YOUR JOB. After you call `finish`, a deterministic
  verifier checks your output (structure, required tags, forbidden patterns),
  and on failure you get another attempt with the exact errors. Do NOT spend
  rounds re-reading, re-counting or re-checking what you wrote — the moment the
  last chunk closes every open tag, call `finish`.

FILE TOOLS:
- `write_file(path, mode="w"|"a")` — write the body from THIS reply to
  `<artifact>/<path>`. `"w"` creates or overwrites, `"a"` appends (creating the
  file when absent). Default is `"w"`. It reports back the CHARACTERS and
  LINES the body added and the file's new totals, so after an append you
  already know where your part landed without reading anything.
- `read_file(path)` — check a file you already wrote. Returns its size, its
  line count and its tail: enough to see that your chunk landed and that the
  file is closed.
- `read_file(path, full=true)` — pulls the ENTIRE file into your context and
  keeps it there for every remaining round (a 25 KB page is ~7k tokens, re-sent
  each round). Use it ONLY when you must re-read content in order to keep
  WRITING. Never to check finished work: the tail plus what `write_file` told
  you already answers "did it land and is it closed", and everything beyond
  that is the verifier's job after `finish`.

DATA INTO FILES:
- For an html-app, the real data goes INTO the output file — but as its own
  part: print the serialised data in a scratchpad cell, then send it as its own
  body with `write_file(path, mode="a")` — a single `<script>` block, separate
  from the markup parts. For a large dataset, aggregate it in the scratchpad
  first; a dashboard almost never needs raw rows.
- For a fullstack app, the generated backend queries the live source itself —
  use the scratchpad mainly to confirm the schema and a sample.\
"""

# The name is kept: generator prompts mix it in whole, and three stage-1c tests
# read `_ROLE` directly.
_ROLE = _ROLE_COMMON + "\n\n" + _ROLE_WRITE


# ---------------------------------------------------------------------------
# Visual design rules (used in every type that has a frontend)
# ---------------------------------------------------------------------------

# Two halves, quoted under their own headings by both frontend builders
# (html-app and the fullstack `static/index.html`).
_DESIGN_RULES = """\
VISUAL DESIGN (for every HTML file you produce):
- Dark theme: background #0d1117, text #e6edf3.
  System sans-serif font stack, generous padding, responsive layout.
- STYLING: Tailwind CSS is the recommended way to style the page. Load it
  once in `<head>` via CDN:
  `<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>`
  and use utility classes for layout, spacing, typography and colour
  (`bg-[#0d1117] text-[#e6edf3]` for the theme). Hand-written CSS in a
  `<style>` block stays allowed wherever utilities fall short — keyframes,
  chart containers, complex selectors — or when the page is too small to
  need a framework. No other CSS or JS library: the two CDN scripts named in
  these rules (Tailwind, ECharts) are the ONLY external resources a page
  may load. Both are fetched every time the page opens, so a page that must
  work OFFLINE or without external resources (brief or PRD says so) skips
  Tailwind and styles itself with hand-written CSS.
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
  - Multi-card grid: `grid-template-columns: repeat(auto-fit, minmax(360px, 1fr))`
  - Chart containers: `width: 100%; height: min(420px, 60vh)`
  - Register `window.addEventListener('resize', () => chart.resize())` on every ECharts instance.
  - Tables wrapped in `<div style="overflow-x: auto;">` — never fixed widths.
- SECURITY: NEVER embed API keys, tokens, passwords, or connection strings in HTML/JS.
  Credentials were already used server-side; serialise only the resulting data.\
"""

_VERIFIER_CONTRACT = """\
A static verifier checks each of these after `finish`; a violation fails the
step and costs a regeneration:
- A complete HTML document with an explicit `<body>`...`</body>`.
- `<meta name="viewport" content="width=device-width, initial-scale=1.0">`.
- No absolute URL in any `fetch()` call — relative paths only.
- The global name `window.__antonCommentsLayer` is never used; the host app
  reserves it.
- Every opened `<script>` block must be closed with `</script>`.
- No universal `* { ... !important }` rule.
- Every `z-index` is 1000 or below.
- Significant block containers (`div`, `section`, `table`, `main`,
  `article`) carry stable `id` attributes — the host app attaches comments
  to them.
- Where a headless browser is available (single-file `html-app` pages only),
  the page is also loaded once: a console error, a crashed renderer or a
  request for a local file that does not exist fails the step; a page with
  no visible text or elements is a warning.\
"""


# ---------------------------------------------------------------------------
# Backend rules (fullstack types only)
# ---------------------------------------------------------------------------

_BACKEND_RULES = """\
BACKEND — `backend.py` (FastAPI, runs locally AND as AWS Lambda):

Use this canonical skeleton verbatim, add routes inside `# === API routes ===`:

```python
import argparse
import os
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

# === Secrets ===
# Keys are the canonical DS_<ENGINE>_<NAME>__<FIELD> names from the
# `## Connected Data Sources` section. Locally each value comes from
# os.environ; in the cloud the shared runner overlays the decrypted values
# onto this dict before every request. READ a secret AT ITS POINT OF USE
# inside the route — never copy it into a module-level variable at import
# time. Leave SECRETS empty if the backend uses none.
SECRETS = {
    # "DS_POSTGRES_PROD_DB__PASSWORD": os.environ.get("DS_POSTGRES_PROD_DB__PASSWORD"),
}

# === API routes ===
@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/api/hello")
async def hello():
    # Example secret use (read at point of use, not at import):
    #   pw = SECRETS["DS_POSTGRES_PROD_DB__PASSWORD"]
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
- Target Python >= 3.12 — that is the runtime the artifact is launched with.
- File MUST be named `backend.py`. The `handler` attribute MUST stay `handler`.
- ALL API endpoints MUST use the `/api/*` prefix (e.g. `/api/items`, `/api/search`).
  Never expose routes at the root — they collide with the `StaticFiles` mount.
- MUST expose a health-check endpoint `GET /api/health` returning
  `200 {"status": "ok"}`. The launcher uses it as the readiness probe.
- API routes MUST be registered BEFORE `app.mount("/", StaticFiles(...))`.
- The backend MUST accept `--port` via argparse. NEVER hardcode a port.
- Keep `Mangum(app, lifespan="off")`. Required for Lambda cold-start.
- SECRETS: expose a module-level `SECRETS` dict keyed by the canonical
  `DS_<ENGINE>_<NAME>__<FIELD>` name, each entry initialised from
  `os.environ.get(...)`. Read a secret AT ITS POINT OF USE — `SECRETS["DS_..."]`
  inside the route — and NEVER hoist it into a module-level variable at import
  time: the import runs before the cloud overlay, so the value would be missed.
  If a credential-backed resource is needed (DB pool, API client), build it
  LAZILY on first request, never at module level.
- Use `async def` for I/O-bound routes (DB queries, external HTTP).
- LOGGING: `print()` and `logging.getLogger(__name__).info(...)` work everywhere.
- DATA SOURCE CREDENTIALS: the user's connected data sources are exposed as
  environment variables named `DS_<ENGINE>_<NAME>__<FIELD>` (uppercase), e.g.
  `DS_POSTGRES_PROD_DB__HOST`, `DS_HUBSPOT_MAIN__ACCESS_TOKEN`. Do NOT derive
  these names yourself — the `## Connected Data Sources` section below lists the
  full variable names verbatim (it writes the same pattern as
  `DS_<ENGINE_NAME>__<FIELD>`; both describe the names printed there). Copy them
  exactly. An invented `DS_*` key fails verification and cannot be recovered.
  If no such section is present, the backend must not read any `DS_*` variable.

`requirements.txt` — always include at minimum:
```
fastapi
mangum
uvicorn
```
Add any other packages the backend imports, one per line. Extras and version
specifiers are fine (`uvicorn[standard]`, `fastapi>=0.100`). Only simple
requirement lines are supported — `-r`, `-e`, `--index-url`, blank lines and
`#` comments are ignored by the installer.

DEPLOYMENT — why the SECRETS rules are shaped this way:
- LOCAL: `python backend.py --port=NNN`. uvicorn serves the app and the
  `static/` mount; secrets come from the `DS_*` env vars in SECRETS' defaults.
- CLOUD: a shared runner overlays the decrypted secrets onto `backend.SECRETS`
  and invokes `backend.handler` per request. The overlay happens AFTER import,
  which is exactly why an import-time copy of a secret is empty in the cloud.
  Statics are served separately, so the StaticFiles mount sits unused there.\
"""


# The ONE rule that differs between the two fullstack types. Selected by
# `stateless` in build_backend_system_prompt — never both, never neither.
_STATELESS_RULES = """\
LOCAL STATE — this app MUST NOT persist anything between requests:
- No local database (no sqlite), no local files used as storage, no on-disk
  caches, no module-level mutable store carried across requests. In Lambda,
  module globals may or may not survive an invocation — never rely on them.
- Treat the filesystem as read-only and non-persistent. NEVER write into the
  artifact folder at runtime. If a request genuinely needs scratch space, use
  the OS temp dir via `tempfile` and treat it as ephemeral — gone the moment the
  request ends.
- Connecting to an EXTERNAL database or API to read/write data IS allowed — that
  is a data source, not local state. Open a fresh connection per request and do
  not cache results in memory across requests.
- Do NOT import `anton_state` — the platform STATE store belongs to
  `fullstack-stateful-app` backends only; this app persists nothing of its own.\
"""

_STATEFUL_RULES = """\
DURABLE STATE — this app persists data through the platform `STATE` store:
- Declare a module-level `STATE = None` right after `SECRETS`. It mirrors
  SECRETS: in the cloud a shared runner overlays `{url, token}` (a short-lived
  capability for the trusted state broker) onto `backend.STATE` before each
  request; run locally the value stays `None` and the SDK falls back to a
  SQLite file next to `backend.py`. One code path serves both — never branch
  on the environment yourself.
- Build the store AT POINT OF USE, inside a route — NEVER at import time (the
  import runs before the cloud overlay, so a module-level store would bind to
  the wrong driver, exactly like an import-time SECRETS copy). Use this helper:
  ```python
  from anton_state import open_store, Collection
  _STATE_DIR = Path(__file__).resolve().parent

  def get_store():
      return open_store(
          state=STATE,
          manifest_path=str(_STATE_DIR / "state_manifest.json"),
          local_path=str(_STATE_DIR / ".anton_state.db"),
      )
  ```
- `anton_state` is an internal SDK injected at runtime. NEVER list `anton_state`
  in `requirements.txt` — it is not a published package, so the dependency
  install FAILS on it. The SDK needs pydantic v2, which the mandatory `fastapi`
  line already provides. `from anton_state import open_store` just works.
- STATE is a document/key-value store keyed by `(pk, sk)` — use it for LIGHT
  state: counters, settings, sessions, simple documents keyed by id. For HEAVY
  or relational needs (joins, transactions, analytics, large datasets) use an
  EXTERNAL database via a connected data source instead — do not force it
  into STATE.
- Write `state_manifest.json` into the artifact root, next to `backend.py`.
  It declares ONLY the key schema and the collection registry — never data
  fields — as one FLAT JSON object:
  ```json
  {"version": 1,
   "pk": {"name": "pk", "type": "S"},
   "sk": {"name": "sk", "type": "S"},
   "collections": ["todos", "counters"]}
  ```
  List EVERY `Collection(store, "<name>")` name used in the code under
  `collections`. Keys are strings (`"type": "S"`) in v1. Do NOT wrap the
  object in a DynamoDB-CreateTable shape (`entities`, `attributes`,
  `partition_key` — all fail validation) and do NOT declare item fields:
  values passed to `store.put({...})` need no schema entry.
- PREFER the `Collection` helper for light state — it manages the keys:
  ```python
  todos = Collection(get_store(), "todos")   # built inside the route
  await todos.put("id1", {"text": "buy milk"})
  items = await todos.list()
  n = await Collection(get_store(), "counters").increment("visits", field="n")
  ```
  Low-level `store` methods (all async; there is NO `scan()` and NO secondary
  indexes — `query` has no `index=` argument):
  * `await store.get(pk, sk=None)` → one item or `None`
  * `await store.put(item)` — the dict MUST include `pk` (and `sk` if the
    schema declares one); `_v` is set by the store, never set it yourself
  * `await store.delete(pk, sk=None)`
  * `await store.query(pk, *, sk_prefix=None, filters=None, limit=None)`
  * `await store.increment(pk, sk=None, *, field, by=1)` — atomic counter;
    do NOT hand-roll read-modify-write
  * `await store.update(pk, sk=None, *, set_fields=None, add_fields=None,
    if_version=None)` — atomic partial update
- DESIGN KEYS AROUND ACCESS PATTERNS: every "list" endpoint must map to ONE
  `query(pk=...)` call (or `Collection.list()`). Never call the store in a
  loop to assemble a listing.
- Do NOT wrap a STATE mutation (`put`/`delete`/`increment`/`update`) in your
  own retry loop: on a timeout the outcome is unknown and a retry can
  double-apply — surface the error instead.
- Never keep state ONLY in module-level Python variables, and never invent
  your own on-disk persistence (sqlite files, JSON files in the artifact
  folder): the STATE store is the single durable layer, working both locally
  and in AWS Lambda.\
"""


# ---------------------------------------------------------------------------
# Frontend rules for fullstack types
# ---------------------------------------------------------------------------

_FRONTEND_RULES = """\
Two of these are verifier checks as well: a page without the api-base meta
tag, or one that calls the backend outside `/api/*`, fails the step.

- Single self-contained HTML file: your own CSS in `<style>`, all JS in
  `<script>`. The only external resources are the CDN scripts named in the
  design rules (Tailwind, ECharts).
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
- Call ALL backend endpoints under the `/api/*` prefix. Never use bare paths.
- `static/` is the ONLY folder the backend serves. ANY additional frontend asset
  (separate CSS, JS, images, fonts, large data payloads) MUST live under
  `static/` too — never at the artifact root, or it will 404 at runtime.\
"""


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

# The entry-point filename for an html-app whose `create_artifact` call set no
# `primary`. Was "dashboard.html" until 2026-09-16 — a leftover from the
# dashboard-only origin of the tool; the tenth live run wrote a card game to
# that name. `index.html` is what a browser, a static host and a reader expect.
HTML_APP_DEFAULT_PRIMARY = "index.html"


# ---------------------------------------------------------------------------
# html-app generator prompt (`generate_frontend`, non-fullstack branch)
#
# Ordered the way the model needs it: the task and its done-criterion first,
# then what the kickoff message carries, then the workflow, and only then the
# mechanics. Each rule is stated once. The section names below quote the
# headings `_spec_context` actually renders, so the model is never pointed at
# a section that does not exist in its input.
# ---------------------------------------------------------------------------

_GEN_HTML_INPUTS = """\
## What you receive
The user message carries these sections, in this order (some may be absent):
- `## Product requirements` (prd.md) — reviewed and accepted by the user.
  This is the authoritative source; where anything else disagrees with it,
  the PRD wins. It ends at `{prd_footer}`.
- `## Brief` — only when there is no PRD: the request as the calling agent
  understood it.
- `## Data` — scratchpad cells already run earlier in this pipeline: pad
  name, code, printed output. The schema and samples you need are here.
- `### Sources read from the web` — notes and quotes from web pages, when the
  task used any.
- `## Technical specification` — `spec.md`: insights and implementation notes
  that complement the PRD rather than repeat it.
- `## Progress journal` — one line per pipeline step done so far.\
"""

# Appended for the fullstack frontend: `build_frontend_kickoff` renders this
# section after the shared context.
_GEN_API_SPEC_INPUT = """\
- `## API Specification` — the backend's `openapi.json`, generated just
  before this step: the exact paths and response shapes your `fetch` calls
  must match.\
"""


def _gen_html_inputs(*, fullstack: bool = False) -> str:
    text = _GEN_HTML_INPUTS.format(prd_footer=PRD_SECTION_FOOTER)
    return text + "\n" + _GEN_API_SPEC_INPUT if fullstack else text


def _gen_html_task(target: str, *, fullstack: bool = False) -> str:
    if fullstack:
        body = f"""\
Produce ONE self-contained HTML file named exactly `{target}`: the
complete frontend of a fullstack app. Inline your own CSS and all JS; the
only external resources allowed are the Tailwind and ECharts `<script>` tags
from the CDNs named in the design rules. The backend is generated in parallel
and serves the endpoints listed under `## API Specification`: take exact
paths and response shapes from there and call every one of them through the
`api()` helper (see Fullstack rules). Embed no data — the page fetches it
from the backend at run time."""
    else:
        body = f"""\
Produce ONE self-contained HTML file named exactly `{target}`, at the root
of the artifact folder. Inline your own CSS and all JS and embed all data in
the file. Do not reference any other file of the artifact; the only external
resources allowed are the Tailwind and ECharts `<script>` tags from the CDNs
named in the design rules."""
    return f"""\
You are a single-purpose worker inside an artifact-generation pipeline. This
step is `generate_frontend`.

## Your task
{body}

UI text and the `<html lang>` attribute follow the language of the PRD.

You are done when the file is written and closed (`</body></html>`) and you
have called `finish(summary="<one line>")`. A deterministic verifier then
checks the file; on failure you get another attempt with the exact errors.
Checking your own output is NOT part of the job.\
"""


def _gen_html_workflow(target: str, *, fullstack: bool = False) -> str:
    if fullstack:
        data_step = """\
2. Data: none to embed — the page fetches it from the backend. A scratchpad
   cell is warranted only when the API Specification leaves a response
   shape unclear and `## Data` holds no sample of it; otherwise skip."""
    else:
        data_step = """\
2. Data, only if needed: when the dataset to embed is not already printed
   under `## Data` in a usable shape, run ONE scratchpad cell that
   aggregates it and prints it as JSON (a dashboard almost never needs raw
   rows). Skip this step when the data is already visible or the artifact
   needs none."""
    return f"""\
## Workflow
A typical run takes one or two rounds.
1. Read the PRD, `## Data` and the specification. Plan the page.
{data_step}
3. Write the whole file in one reply: the body between the markers plus one
   `write_file(path="{target}")` call — see Output protocol.
4. Call `finish`. Do not read the file back and do not run checks.\
"""


def _gen_html_output_protocol(target: str) -> str:
    return f"""\
## Output protocol
A file is written in TWO parts of the SAME reply:
  1. the file content, as plain text between two marker lines;
  2. a `write_file` call naming the path — with NO content argument.

{BEGIN}
<!DOCTYPE html>
<html lang="…">
…the entire file…
</html>
{END}

…and in the same reply: `write_file(path="{target}")`.

Rules:
- The body is TEXT between the markers. `write_file` takes `path` and `mode`
  only; there is no `content` argument.
- One body per reply, so one `write_file` per reply.
- Nothing but the file goes between the markers; the closing marker ends the
  file. A reply without the closing marker writes nothing.
- All paths are relative to the artifact root. Never write outside it.\
"""


_GEN_SIZE_RULES = f"""\
### Size and splitting
One reply holds about {REPLY_BODY_CHARS:,} characters of file content. Almost
every artifact fits, so the default is the whole file in a single body with
`write_file(path, mode="w")`. Split ONLY a file that will clearly exceed it:
- Each part costs a round; splitting a file that would have fit wastes it.
- Continue exactly where the file now ends and append with `mode="a"`.
  The `write_file` result reports the CHARACTERS and LINES added and the
  file's new totals, so you know where the part landed without reading.
- If a reply is cut off before the closing marker, nothing is written and
  the tool result tells you what to send next. Follow it; never re-emit the
  whole file to "fix" something.
- The last part must close every open tag, `</body></html>` included.\
"""


_GEN_TOOLS = """\
## Tools
- `scratchpad(action="exec", name="<pad>", code=...)` — a persistent Python
  pad; needed only for step 2 of the workflow. Reuse the pad named in
  `## Data`; NEVER create a scratchpad with a new name — it is an empty,
  isolated environment and the call may be rejected. Provide
  `one_line_description` and `estimated_execution_time_seconds` on every
  `exec`. The namespace starts clean: put every import at the top of the
  cell. A cell has a hard 120-second timeout; on timeout all state is lost,
  so keep cells small. Always `print(...)` what you want to see. Data-source
  credentials are environment variables `DS_<ENGINE>_<NAME>__<FIELD>`; read
  them from `os.environ`, never from `data_vault` files. If a cell fails the
  same way twice, change strategy instead of re-running it.
- `write_file(path, mode="w"|"a")` — writes the body of THIS reply.
- `read_file(path)` — size, line count and tail of a file you wrote.
  `read_file(path, full=true)` pulls the ENTIRE file into your context and
  keeps it there for every remaining round. Use either only when you must
  re-read content in order to keep WRITING — never to check finished work.
- `finish(summary)` — call exactly once, after the file is written.

Python → JS: if you ever build file text inside a scratchpad cell, escape
sequences resolve in Python first, so `'\\n'` breaks a JS string literal; use
raw strings. Writing the text directly between the markers avoids this.\
"""


def build_subagent_system_prompt(
    artifact_type: str,
    artifact_path: Path,
    *,
    primary: str | None = None,
) -> str:
    """System prompt for the single-generator path — html-app only.

    The fullstack types never reach here: `orchestrator._gen_verify_frontend`
    calls this only in its non-fullstack branch, and fullstack generation uses
    `build_backend_system_prompt` / `build_frontend_system_prompt` instead. The
    fullstack branches that used to live here were a third copy of the backend
    contract that nothing executed and no test covered.

    `primary` is the filename the artifact was registered with. It may be None
    (`Artifact.primary: str | None`, `artifacts/models.py:153`) — then the shared
    default applies, the same one the orchestrator's cleanup step uses, so the
    two never disagree about which file is the entry point.
    """
    parts: list[str] = [_ROLE]

    if artifact_type != "html-app":
        parts.append(
            f"## Unsupported artifact type: {artifact_type!r}\n"
            "This builder serves `html-app` only. Fullstack types use "
            "`build_backend_system_prompt` / `build_frontend_system_prompt`."
        )
        return "\n\n".join(parts)

    target = primary or HTML_APP_DEFAULT_PRIMARY
    # `artifact_path` is deliberately not quoted: every path the model writes
    # is relative to the artifact root, and an absolute path in the prompt is
    # only something to misuse. The parameter stays for signature stability.
    return "\n\n".join(
        [
            _gen_html_task(target),
            _gen_html_inputs(),
            _gen_html_workflow(target),
            _gen_html_output_protocol(target),
            _GEN_SIZE_RULES,
            "## Verifier contract\n" + _VERIFIER_CONTRACT,
            "## Design rules\n" + _DESIGN_RULES,
            _GEN_TOOLS,
        ]
    )


def _kickoff_closing(*, fullstack: bool) -> str:
    data_clause = (
        "the page fetches its data from the backend, so a scratchpad cell is "
        "warranted only for a response shape the API Specification leaves "
        "unclear"
        if fullstack
        else "run a scratchpad cell only if the data to embed is not already "
        "usable under `## Data`"
    )
    return (
        "Read the sections above, then follow the workflow from your "
        f"instructions: {data_clause}; write the file as text between "
        f"`{BEGIN}` and `{END}` plus one `write_file` call in the same reply; "
        "then call `finish`."
    )


def build_user_kickoff(context: str) -> str:
    # The context renders its own section headers (`## Product requirements`,
    # or `## Brief` when there is no PRD) — see `orchestrator._spec_context`.
    return "\n\n".join([context.strip(), _kickoff_closing(fullstack=False)])


# ---------------------------------------------------------------------------
# API spec generation (planning call, no tools)
#
# Like `make_tech_spec` (0c4767d6), the step's rules travel in the step
# message: on the hot path the node continues the shared history under the
# pipeline system prompt, so a system prompt of its own is never seen there.
# Until 2026-09-17 the rules lived only in `_API_SPEC_SYSTEM`, and every live
# run got a fenced OpenAPI 3.0 document with tags, header schemas and two
# examples per operation — 4 000 characters for one endpoint — plus the PRD
# and spec.md restated in the message although both were already in the
# history as the model's own replies.
# ---------------------------------------------------------------------------

_API_SPEC_STATELESS = (
    "## Stateless constraint\n"
    "The backend implementing this spec MUST NOT persist any state between "
    "requests: no local storage (sqlite, local files, on-disk caches) and no "
    "in-memory store carried across requests. Connecting to an EXTERNAL "
    "database or API to read/write data IS allowed. Design endpoints "
    "accordingly — do NOT assume server-side sessions or mutable persisted "
    "collections."
)

_API_SPEC_STATEFUL = (
    "## Durable state constraint\n"
    "The backend implementing this spec persists its own data through the "
    "platform STATE store — a document/key-value store keyed by "
    "(partition key, sort key), organised into named collections. It has "
    "NO scan operation and NO secondary indexes, so design every listing "
    "endpoint to map onto ONE partition-key query (one collection = one "
    "listing); an endpoint that would need to read \"everything across "
    "partitions\" cannot be implemented. Counters must be served by an "
    "atomic increment, not read-modify-write. Keep the stored shapes to "
    "LIGHT state: settings, sessions, counters, simple documents keyed by "
    "id. If the requirements need joins, transactions or analytics over "
    "large data, design those endpoints against an EXTERNAL connected "
    "database instead of the STATE store."
)


def build_api_spec_instruction(*, stateless: bool) -> str:
    """The step message for `make_api_spec`, complete in itself.

    On the hot path this is the ONLY step-specific text the model sees;
    `build_api_spec_prompt` prepends the assembled context for the cold
    start, where there is no history to have seen it.
    """
    return (
        "## Your task\n"
        "Write the API contract (`openapi.json`) that the backend and frontend "
        "generators build from, independently and in parallel. Do not call any "
        "tool. Reply with ONE OpenAPI 3.1 JSON document and nothing else: no "
        "markdown fence, no preamble, no commentary.\n\n"
        "## Content\n"
        "- The endpoints are the ones `spec.md` lists under `## Backend`: "
        "formalise exactly those, every path under `/api/...`; add none, drop "
        "none. Where `spec.md` lists none, derive them from the PRD's "
        "functional requirements. `/api/health` is added by the backend "
        "generator on its own and may be left out.\n"
        "- For every operation: a one-line `summary`, its path/query "
        "`parameters`, a `requestBody` schema for POST/PUT/PATCH, the `200` "
        "response schema, and any non-200 status the frontend must handle.\n"
        "- One response `example` per operation, taken from the `### Sample` "
        "data when the requirements carry one.\n"
        "- Compact: no `tags`, no `info.description`, no header schemas, no "
        "per-field `pattern` or `example` where the type already says it, no "
        "prose the generators do not need. Every line is context both "
        "generators carry on every round.\n\n"
        + (_API_SPEC_STATELESS if stateless else _API_SPEC_STATEFUL)
        + "\n\nWrite the OpenAPI JSON document now."
    )


def build_api_spec_prompt(
    context: str,
    *,
    stateless: bool = False,
) -> tuple[str, str]:
    """Cold-start form: the assembled context plus the same instruction the
    hot path sends, so both paths ask for the same document."""
    system = (
        _DATA_CONTEXT_HEADER
        + "You are the `make_api_spec` node. Your document is saved to "
        "`openapi.json` and handed to the backend and frontend generators "
        "next to the material below; the instruction at the end of the user "
        "message says what goes into it."
    )
    user = "## Requirements\n" + context.strip() + "\n\n" + build_api_spec_instruction(
        stateless=stateless
    )
    return system, user


# ---------------------------------------------------------------------------
# Backend-only system prompt and kickoff (parallel fullstack-stateful-app)
# ---------------------------------------------------------------------------

def build_backend_system_prompt(
    artifact_path: Path,
    *,
    stateless: bool = False,
    datasource_context: str = "",
) -> str:
    parts: list[str] = [_ROLE]
    if stateless:
        task = (
            "## Your task\n"
            "Produce exactly two files:\n"
            "1. `backend.py` — FastAPI backend implementing the API Specification you receive.\n"
            "2. `requirements.txt` — pip dependencies.\n"
            "The frontend is being generated in parallel — focus ONLY on the backend.\n"
            "Implement every endpoint in the spec exactly as described."
        )
    else:
        task = (
            "## Your task\n"
            "Produce exactly three files:\n"
            "1. `backend.py` — FastAPI backend implementing the API Specification you receive.\n"
            "2. `state_manifest.json` — the STATE key schema and collection registry "
            "(see DURABLE STATE below).\n"
            "3. `requirements.txt` — pip dependencies.\n"
            "The frontend is being generated in parallel — focus ONLY on the backend.\n"
            "Implement every endpoint in the spec exactly as described."
        )
    parts.append(task)
    parts.append(_BACKEND_RULES)
    parts.append(_STATELESS_RULES if stateless else _STATEFUL_RULES)
    if datasource_context.strip():
        parts.append(datasource_context.strip())
    parts.append(
        "## Output folder\n"
        f"All `write_file` paths are relative to: `{artifact_path}`\n"
        "Do NOT write outside that folder."
    )
    return "\n\n".join(parts)


def build_backend_kickoff(
    context: str,
    api_spec: str,
) -> str:
    parts = [context.strip()]
    parts.append("## API Specification\n" + api_spec)
    parts.append(
        "Use the `scratchpad` tool to confirm the schema/sample of any data "
        "described under `## Data`. Then write `backend.py` first: its content "
        f"as text between `{BEGIN}` and `{END}`, plus `write_file(path=\"backend.py\")` "
        "in the same reply. You will receive the next instruction after it is "
        "written."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Fullstack frontend (`static/index.html`) — system prompt and kickoff
#
# Built from the same blocks as the html-app prompt, in the same order, with
# the fullstack-only material in two places: the task paragraph and a
# `## Fullstack rules` section. Until 2026-09-17 this builder stacked the
# shared `_ROLE` (scratchpad discipline, an html-app data recipe, a second
# copy of the write protocol) on top of its own blocks — 32 % more text than
# the html-app prompt, no workflow, no input map, no UI-language rule, stale
# "brief" wording, and every protocol fix had to be made twice.
# ---------------------------------------------------------------------------

FULLSTACK_FRONTEND_TARGET = "static/index.html"


def build_frontend_system_prompt(artifact_path: Path) -> str:
    # `artifact_path` is not quoted, for the same reason as in the html-app
    # builder: every path the model writes is relative to the artifact root.
    target = FULLSTACK_FRONTEND_TARGET
    return "\n\n".join(
        [
            _gen_html_task(target, fullstack=True),
            _gen_html_inputs(fullstack=True),
            _gen_html_workflow(target, fullstack=True),
            _gen_html_output_protocol(target),
            _GEN_SIZE_RULES,
            "## Verifier contract\n" + _VERIFIER_CONTRACT,
            "## Fullstack rules\n" + _FRONTEND_RULES,
            "## Design rules\n" + _DESIGN_RULES,
            _GEN_TOOLS,
        ]
    )


def build_frontend_kickoff(
    context: str,
    api_spec: str,
) -> str:
    parts = [context.strip()]
    parts.append(
        "## API Specification\n"
        "(Call these endpoints with `fetch(api('/api/...'))` — "
        "the backend serves them.)\n\n"
        + api_spec
    )
    parts.append(_kickoff_closing(fullstack=True))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Data-phase prompts (decisions + fetch loop)
# ---------------------------------------------------------------------------

_DATA_CONTEXT_HEADER = (
    "You are one step of a strict artifact-generation state machine. The full "
    "pipeline is this graph:\n\n"
    f"{FSM_DIGRAPH}\n"
)


PRD_SECTION_HEADER = (
    "## Product requirements (prd.md — reviewed and accepted by the user; "
    "this is the authoritative requirements source)"
)
PRD_SECTION_FOOTER = "--- end of prd.md ---"


def prd_section(state) -> str:
    """The PRD block, or "" when this run has no PRD.

    The header states the document's standing rather than leaving the node to
    infer it from position: the same context carries `spec.md` and the data
    record, and on any disagreement the accepted PRD wins. (The brief travels
    only when there is no PRD — see `orchestrator._spec_context`.) One
    renderer for every node, so no node reads a differently-framed PRD.

    The footer marks where the quoted document ends. `prd.md` carries its own
    `##` headings, which land as siblings of the wrapper's — without a closing
    marker the PRD's last section and whatever follows it in the context are
    indistinguishable. Demoting those headings instead would corrupt the
    fenced connection-code examples the PRD is required to include.
    """
    body = (getattr(state, "prd", "") or "").strip()
    return f"{PRD_SECTION_HEADER}\n{body}\n{PRD_SECTION_FOOTER}" if body else ""


def _brief_and_notes(state) -> str:
    parts = [f"## Brief\n{state.brief.strip()}"]
    prd = prd_section(state)
    if prd:
        parts.append(prd)
    if state.data_notes.strip():
        parts.append(f"## Data gathered so far\n{state.data_notes.strip()}")
    else:
        parts.append("## Data gathered so far\n(nothing gathered yet)")
    if getattr(state, "web_notes", "").strip():
        parts.append(state.web_notes.strip())
    journal = state.journal()
    if journal:
        parts.append(f"## Progress journal (steps completed so far)\n{journal}")
    return "\n\n".join(parts)


def build_required_data_prompt(state) -> tuple[str, str]:
    system = (
        _DATA_CONTEXT_HEADER
        + "You are the `define_required_data` node. List exactly which data is "
        "missing and where each item can be obtained (user's connected data "
        "sources or public/documented APIs). Be concrete and minimal."
    )
    user = _brief_and_notes(state) + "\n\nList the required data items."
    return system, user


def build_can_fetch_prompt(state, required: str) -> tuple[str, str]:
    system = (
        _DATA_CONTEXT_HEADER
        + "You are the `is_possible_to_fetch` decision node. Given the required "
        "data and the available sources, decide whether the data can actually be "
        "obtained. If a needed source simply does not exist among the user's "
        "connections or public sources, it is NOT possible."
    )
    user = (
        _brief_and_notes(state)
        + f"\n\n## Required data\n{required}\n\nIs it possible to fetch this data?"
    )
    return system, user


def build_fetch_data_system_prompt(
    artifact_path, *, datasource_context: str = "", public_sources: str = ""
) -> str:
    parts = [
        _ROLE_COMMON,
        _DATA_CONTEXT_HEADER
        + "You are the `fetch_data_sample` node. Use the `scratchpad` tool to "
        "run Python that pulls a small SAMPLE of the required data (query the "
        "DB, call the API, read the file). Confirm the shape and types. Do NOT "
        "write any artifact files.\n\n"
        "Fetch ONLY WHAT IS MISSING. `## Data gathered so far` may already show "
        "cells the main agent ran — do not repeat them.\n\n"
        "Work in the SAME scratchpad the brief names, not a new one: a fresh "
        "name is an isolated environment, so its variables, imports and working "
        "connection code do not exist there and you would rebuild them from "
        "scratch.\n\n"
        "When done, call `finish(summary=...)` with a "
        "precise description of WHICH scratchpad(s)/cell(s) you used, what each "
        "produced, and the observed schema/sample — this summary is handed to "
        "the next steps.",
    ]
    if public_sources.strip():
        parts.append(public_sources.strip())
    if datasource_context.strip():
        parts.append(datasource_context.strip())
    parts.append(
        "## Output folder\n"
        f"(You will NOT write files here in this step.) Artifact folder: `{artifact_path}`"
    )
    return "\n\n".join(parts)


def build_fetch_data_kickoff(state) -> str:
    return (
        _brief_and_notes(state)
        + "\n\nUse the `scratchpad` tool to fetch a data sample, then call "
        "`finish` with a precise summary of the scratchpads/cells and the "
        "schema/sample you observed."
    )


# ---------------------------------------------------------------------------
# Tech-spec prompt (make_tech_spec → spec.md)
# ---------------------------------------------------------------------------

# The spec writer never sees _BACKEND_RULES/_FRONTEND_RULES (those go to the
# generator system prompts), so without this block it invents its own stack
# (Node/Express, fixed ports, …) and spec.md contradicts what gets built.
_TECH_SPEC_STACK = """\
FIXED TECHNOLOGY STACK — already decided by the pipeline, NOT yours to choose.
The spec MUST NOT contain a technology-selection section and MUST NOT propose
any other stack. Describe behaviour, screens, data flow, and endpoints on top of:
- Backend (fullstack types only): Python >= 3.12, FastAPI, everything in a
  single `backend.py`, dependencies in `requirements.txt`. It runs both
  locally and as an AWS Lambda.
- All API endpoints live under the `/api/*` prefix.
- The launcher assigns the port at run time: never mention a port number or
  an absolute URL anywhere in the spec.
- Frontend: one self-contained HTML file (`static/index.html` for fullstack
  types) — vanilla JavaScript, Tailwind CSS via CDN for styling (hand-written
  CSS where needed), Apache ECharts for charts. Both libraries need network
  access when the page opens; when the PRD requires offline use, the spec
  says so and the frontend is styled with hand-written CSS only.
- Durable state (`fullstack-stateful-app` ONLY): the platform STATE store — a
  document/key-value store keyed by (partition key, sort key) with named
  collections, no scan, no secondary indexes, atomic counters. The spec must
  name the collections and what each stores; every listing must come from one
  partition-key query. Do NOT propose sqlite or local files as storage. Heavy
  or relational data (joins, transactions, analytics) belongs in an EXTERNAL
  connected database, not in the STATE store. For every other artifact type
  there is no local persistence at all — data lives in external sources.\
"""


# Everything the spec writer is told lives in the step INSTRUCTION, not in a
# system prompt: on the hot path the node continues the phases A-D history
# under the shared pipeline system prompt (part of the cached prefix, so it
# cannot carry step-specific text), and the seventh live run of 2026-09-16
# showed what a bare "write the specification now" produces next to a
# confirmed PRD — a 5.5 KB retelling of a 2.1 KB PRD, eight sections of which
# five restated the PRD and one (acceptance criteria) restated it a third
# time, riding into every generation round as 60 % of the frontend context.
# The rules below used to sit in the cold-start system prompt only, where no
# live run ever read them.

# What the code-writing steps actually receive next to spec.md — see
# `orchestrator._spec_context`. Stated exactly, because the previous wording
# ("nothing else from above reaches them") was false and pushed the model to
# carry the PRD forward verbatim.
TECH_SPEC_CARRY_FORWARD = (
    "This is the LAST step that sees this conversation. Next to your "
    "document, the code-writing steps receive the PRD verbatim, "
    "the scratchpad cells run in this pipeline (code plus the first "
    f"{EXEC_OUTPUT_MAX} characters of each printed output) and short excerpts "
    "of the web pages read. They do NOT receive this conversation, full "
    "scratchpad outputs or full page texts. So carry forward verbatim only "
    "what the build step needs and cannot get elsewhere: exact figures and "
    "computed values, quotes to display, image URLs, the source link, the "
    "full text of anything the artifact must reproduce exactly. Do not "
    "paraphrase what must be reproduced exactly."
)

_TECH_SPEC_NO_RESTATE = (
    "- Do not restate the PRD: no retelling of its goal, requirements, data "
    "model, copy or structure. Where the PRD decides something, do not "
    "repeat it."
)

_TECH_SPEC_CONTENT = (
    "## Content\n"
    "- `## Insights`: only when the artifact shows data to a human (a "
    "dashboard, report or any charted view). One line each, per chart or "
    "element: `<element>: <what it conveys and why it matters>`. A checklist, not "
    "prose, no design discussion. It tells the frontend generator what each "
    "visual is FOR.\n"
    "- `## Implementation notes`: component breakdown, state, interaction "
    "details, timings, edge cases, exact UI strings. Terse bullets. Skip what "
    "the frontend's own design rules already decide (theme, fonts, chart "
    "library); name only what is specific to this artifact."
)

_TECH_SPEC_BACKEND = (
    "- `## Backend` (fullstack types): the endpoints under `/api/*`, the data "
    "flow between frontend, backend and the external sources, and for a "
    "stateful app the STATE collections with what each stores."
)

_TECH_SPEC_EXCLUSIONS = (
    "- No acceptance criteria, no restated constraints, no "
    "technology-selection section."
)


def build_tech_spec_instruction(state) -> str:
    """The step message for `make_tech_spec`, complete in itself.

    On the hot path this is the ONLY step-specific text the model sees (the
    system prompt is the shared pipeline one); `build_tech_spec_prompt`
    prepends the assembled context for the cold-start path, where there is
    no history to have seen it. Tolerates a bare object — the only two
    attributes read are optional.
    """
    has_prd = bool((getattr(state, "prd", "") or "").strip())
    fullstack = bool(getattr(state, "is_fullstack", False)) or str(
        getattr(state, "artifact_type", "")
    ).startswith("fullstack-")
    reach = ["## What reaches the build step", TECH_SPEC_CARRY_FORWARD]
    if has_prd:
        reach.append(_TECH_SPEC_NO_RESTATE)
    content = [_TECH_SPEC_CONTENT]
    if fullstack:
        content.append(_TECH_SPEC_BACKEND)
    content.append(_TECH_SPEC_EXCLUSIONS)
    return (
        "## Your task\n"
        "Write the technical specification (`spec.md`) for the build. Do not "
        "call any tool. Reply with the document only, as GitHub-flavoured "
        "markdown, no code fence around the whole document.\n\n"
        + "\n".join(reach)
        + "\n\n"
        + "\n".join(content)
        + "\n\n## Fixed stack\n"
        + _TECH_SPEC_STACK
        + "\n\nKeep the whole document short: it complements the PRD, and "
        "every line is context the build step carries on every round."
    )


def build_tech_spec_prompt(state) -> tuple[str, str]:
    """Cold-start form: the assembled context plus the same instruction the
    hot path sends, so both paths ask for the same document."""
    system = (
        _DATA_CONTEXT_HEADER
        + "You are the `make_tech_spec` node. Your document is saved to "
        "`spec.md` and handed to the backend and frontend generators next to "
        "the material below; the instruction at the end of the user message "
        "says what goes into it."
    )
    user = _brief_and_notes(state) + (
        f"\n\n## Artifact type\n{state.artifact_type}\n\n"
        + build_tech_spec_instruction(state)
    )
    return system, user
