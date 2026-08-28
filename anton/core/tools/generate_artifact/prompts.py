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

# The chunk limit is quoted to the model here and enforced in `write_file`;
# reading it from the one constant keeps the two from drifting apart.
from .sub_tools import CHUNK_SOFT_LIMIT


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
_ROLE_WRITE = """\
YOUR OUTPUT IS FILES: produce them by calling `write_file`, then call `finish`.

HARD RULES:
- Build every file with `write_file`. A large file MUST be written in several
  chunks — one `mode="w"` call followed by `mode="a"` calls. A single call
  carrying a whole file is cut off by the output limit and rejected.
- All `path` values are RELATIVE to the artifact folder — never write outside it.
- Call `finish(summary="<one line>")` exactly once when all files are written.
- VERIFICATION IS NOT YOUR JOB. After you call `finish`, a deterministic
  verifier checks your output (structure, required tags, forbidden patterns),
  and on failure you get another attempt with the exact errors. Do NOT spend
  rounds re-reading, re-counting or re-checking what you wrote — the moment the
  last chunk closes every open tag, call `finish`.

FILE TOOLS:
- `write_file(path, content, mode="w"|"a")` — write a UTF-8 text file at
  `<artifact>/<path>`. `"w"` creates or overwrites, `"a"` appends (creating the
  file when absent). Default is `"w"`.
- `read_file(path)` — read a file you already wrote (for iterative refinement).

DATA INTO FILES:
- For an html-app, the real data goes INTO the output file — but as its own
  chunk: print the serialised data in a scratchpad cell, then append it with
  `write_file(path, content, mode="a")` as a single `<script>` block, separate
  from the markup chunks. For a large dataset, aggregate it in the scratchpad
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
  Credentials were already used server-side; serialise only the resulting data.

HARD OUTPUT CONTRACT (a static verifier checks each of these; a violation
fails the step and costs a regeneration):
- Emit a complete HTML document with an explicit `<body>`...`</body>`.
- Include `<meta name="viewport" content="width=device-width, initial-scale=1.0">`.
- NEVER put an absolute URL in a `fetch()` call. Use relative paths only.
- NEVER use the global name `window.__antonCommentsLayer`; it is reserved by the
  host app.
- Write script tags as plain `<script>` and `</script>` — never any underscore
  variant, and every opened script block must be closed with `</script>`.
- NEVER write a universal `* { ... !important }` rule.
- Keep every `z-index` at 1000 or below.
- Give significant block containers (`div`, `section`, `table`, `main`,
  `article`) stable `id` attributes — they are anchors the host app attaches
  comments to.\
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
- Call ALL backend endpoints under the `/api/*` prefix. Never use bare paths.
- `static/` is the ONLY folder the backend serves. ANY additional frontend asset
  (separate CSS, JS, images, fonts, large data payloads) MUST live under
  `static/` too — never at the artifact root, or it will 404 at runtime.\
"""


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

# Mixed in everywhere _VISUAL_RULES is: chunked writing is needed by html-app and
# by the fullstack frontend alike. It cannot live in _FRONTEND_RULES — that block
# only goes to the fullstack branch.
_WRITE_DISCIPLINE = f"""\
WRITING A LARGE FILE (mandatory — a single big call cannot succeed):
Your reply has a hard output limit. A `write_file` call carrying a whole
dashboard exceeds it, gets cut off mid-argument, and is REJECTED — nothing is
written and the round is wasted. Build the file in chunks instead:
- First chunk: `write_file(path, content, mode="w")` — head, `<style>`, opening
  `<body>`.
- Every next chunk: `write_file(path, content, mode="a")` — one section per call:
  the data block, then each chart's markup, then the scripts, then the closing
  tags.
- HARD CHUNK LIMIT: at most {CHUNK_SOFT_LIMIT:,} characters of `content` per
  call — the FIRST `mode="w"` chunk and the first `mode="a"` chunk included;
  those are exactly where oversized calls fail. A 40 KB page is 3-4 chunks.
  Several small calls in one reply are fine and cost one round together.
- Do NOT re-emit the whole file to "fix" something — append the remaining part.
  To check what landed, `read_file` the path — it returns the size and the tail,
  which is all you need.
- The final chunk must close every tag you opened, `</body></html>` included.

PYTHON → JS STRING SAFETY (only when you build content inside a scratchpad cell):
Escape sequences resolve in Python BEFORE the text reaches the file, so `'\\n'`
inside a Python string becomes a real newline and breaks a JS string literal.
Use raw strings (`r"..."`) for JS blocks, or double-escape. Writing the text
directly through `write_file` avoids the problem entirely — prefer that.\
"""

HTML_APP_DEFAULT_PRIMARY = "dashboard.html"


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
    parts.append(
        "## Your task\n"
        f"Produce ONE self-contained HTML file named exactly `{target}`. "
        "Inline all CSS and JS. All data must be embedded — no external local "
        "file references."
    )
    parts.append(_VISUAL_RULES)
    parts.append(_WRITE_DISCIPLINE)
    parts.append(
        "## Output folder\n"
        f"All `write_file` paths are relative to: `{artifact_path}`\n"
        "Do NOT write outside that folder."
    )
    return "\n\n".join(parts)


def build_user_kickoff(context: str) -> str:
    parts: list[str] = ["## Brief", context.strip()]
    parts.append(
        "Use the `scratchpad` tool to reach any data described under `## Data`. "
        "Then write every file using `write_file`, and call `finish`."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# API spec generation (planning call, no tools)
# ---------------------------------------------------------------------------

_API_SPEC_SYSTEM = """\
You are a REST API designer.

Given requirements (which may include a `### Sample` of real data under the
`## Data` section), write a concise API specification that both a backend
developer and a frontend developer can implement from independently and in
parallel.

Output an OpenAPI 3.1 specification as a single JSON document.

Rules:
- Cover ALL endpoints needed to fulfill the requirements, under `/api/...`.
- For every operation include a one-line `summary`, path/query `parameters`,
  a `requestBody` schema for POST/PUT, and `responses` for `200` plus any
  non-200 codes callers must handle.
- Provide response `examples` derived from the data the brief describes
  (the `### Sample` subsection, when present).
- Be precise — frontend and backend are generated in parallel from this spec.
- Output ONLY the raw JSON document — no markdown fences, no preamble.\
"""


def build_api_spec_prompt(
    context: str,
    *,
    stateless: bool = False,
) -> tuple[str, str]:
    parts = ["## Requirements", context.strip()]

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
    else:
        parts.append(
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

    parts.append("Write the OpenAPI JSON specification now.")
    return _API_SPEC_SYSTEM, "\n\n".join(parts)


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
    parts = ["## Brief", context.strip()]
    parts.append("## API Specification\n" + api_spec)
    parts.append(
        "Use the `scratchpad` tool to confirm the schema/sample of any data "
        "described under `## Data`. Then write `backend.py` first using "
        "`write_file`. You will receive the next instruction after it is written."
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
    parts.append(_WRITE_DISCIPLINE)
    parts.append(_FRONTEND_RULES)
    parts.append(
        "## Output folder\n"
        f"All `write_file` paths are relative to: `{artifact_path}`\n"
        "Do NOT write outside that folder."
    )
    return "\n\n".join(parts)


def build_frontend_kickoff(
    context: str,
    api_spec: str,
) -> str:
    parts = ["## Brief", context.strip()]
    parts.append(
        "## API Specification\n"
        "(Call these endpoints with `fetch(api('/api/...'))` — "
        "the backend serves them.)\n\n"
        + api_spec
    )
    parts.append(
        "Use the `scratchpad` tool to inspect a data sample if you need it to "
        "design charts and tables. Then write `static/index.html` using "
        "`write_file`, and call `finish`."
    )
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
    infer it from position: the same context also carries `## Brief`, written
    by the calling agent, and on any disagreement the accepted PRD wins. One
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
    journal = state.journal()
    if journal:
        parts.append(f"## Progress journal (steps completed so far)\n{journal}")
    return "\n\n".join(parts)


def build_data_enough_prompt(state) -> tuple[str, str]:
    system = (
        _DATA_CONTEXT_HEADER
        + "You are the `is_data_enough` decision node. Decide whether there is "
        "ALREADY enough data to build the artifact. If the task needs no external "
        "data at all (e.g. 'show the current time'), that counts as ENOUGH. "
        "Otherwise it is enough only when the source, schema, and a concrete "
        "sample of every needed dataset are known.\n\n"
        "Anything under `## Data gathered so far` counts as already available, "
        "regardless of who obtained it — a scratchpad cell the main agent ran "
        "before calling this tool is just as good as one you would run yourself. "
        "Do NOT ask for a re-fetch of something already shown there.\n\n"
        "Answer strictly."
    )
    user = _brief_and_notes(state) + "\n\nIs there enough data? Answer with the verdict."
    return system, user


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
  types) with inline CSS/JS — vanilla JavaScript, Apache ECharts for charts.
- Durable state (`fullstack-stateful-app` ONLY): the platform STATE store — a
  document/key-value store keyed by (partition key, sort key) with named
  collections, no scan, no secondary indexes, atomic counters. The spec must
  name the collections and what each stores; every listing must come from one
  partition-key query. Do NOT propose sqlite or local files as storage. Heavy
  or relational data (joins, transactions, analytics) belongs in an EXTERNAL
  connected database, not in the STATE store. For every other artifact type
  there is no local persistence at all — data lives in external sources.\
"""


# html-app with a confirmed PRD: the PRD already fixes goal, data model,
# functional and UI/UX requirements, and `_spec_context` hands it to the
# generator verbatim NEXT TO this document. A full spec there is almost pure
# duplication — measured 2026-08-27: 190 s and 13k output tokens to restate a
# 20 KB PRD as a 35 KB spec, which then rode into every generation prompt.
_TECH_SPEC_COMPACT = """\
A user-confirmed PRD is provided below. It is the authoritative requirements
source, and the generator receives it VERBATIM alongside your document — so do
NOT restate it: no retelling of its content, requirements, data, copy or
structure. Write ONLY what the PRD does not already say:
- `## Insights` — as described above, when the artifact shows data.
- `## Implementation notes` — component breakdown, rendering and interaction
  details, tricky parts, edge cases. Terse bullet points, no prose.
Keep the whole document SHORT — it complements the PRD instead of replacing
it, and every line you write is context the generator must carry."""


def build_tech_spec_prompt(state) -> tuple[str, str]:
    system = (
        _DATA_CONTEXT_HEADER
        + "You are the `make_tech_spec` node. Write a detailed technical "
        "specification for building this artifact — the backend behaviour (if a "
        "backend is needed) and the frontend behaviour, screens, and data flow. "
        "Base it on the brief and the gathered data. Output GitHub-flavoured "
        "Markdown only (no code fences around the whole document). This document "
        "will be saved to `spec.md` and handed to the backend and frontend "
        "generators.\n\n"
        "If the artifact shows data to a human (a dashboard, report or any "
        "charted view), OPEN the specification with an `## Insights` section: "
        "one line each, `<chart or element>: <the insight it conveys and why it "
        "matters>`. Terse — a checklist, not prose, no design discussion. It "
        "tells the frontend generator what each visual is FOR, which is the "
        "difference between a polished page and a pile of charts.\n\n"
        + (
            _TECH_SPEC_COMPACT + "\n\n"
            if state.artifact_type == "html-app" and getattr(state, "prd", "")
            else ""
        )
        + _TECH_SPEC_STACK
    )
    user = _brief_and_notes(state) + (
        f"\n\n## Artifact type\n{state.artifact_type}\n\n"
        "Write the technical specification now."
    )
    return system, user
