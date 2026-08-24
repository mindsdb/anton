---
name: build-fullstack-backend
description: 'ONLY for writing backend or fullstack code BY HAND. NOT needed on the
  normal path — create_artifact(fullstack-stateless-app or fullstack-stateful-app)
  followed by generate_artifact writes the backend, verifies it by importing it and
  launches it. Recall this when editing an existing fullstack artifact, or when
  generate_artifact failed and the user asked you to continue manually: it is the hard
  contract — canonical FastAPI+Mangum backend.py template, SECRETS handling, /api/*
  route prefix rules, static/ frontend layout, requirements.txt, launch and preview.
  Covers backend, API and server code.'
metadata:
  display_name: Backend & fullstack app generation
  provenance: builtin
---
BACKEND & FULLSTACK APPLICATION GENERATION:

When the user asks to build a backend service, web application with a backend, or API-driven system, follow this workflow. It covers BOTH fullstack artifact types — the steps are identical; only the LOCAL STATE rule (see RULES) differs.

HARD CONTRACT (violating ANY of these breaks launch or deployment — full explanations in the RULES of step 4):
- The backend file is `<artifact_path>/backend.py`; the `handler` attribute and the `SECRETS` dict keep exactly those names.
- `handler = Mangum(app, lifespan="off")`.
- ALL API routes live under `/api/*` and are registered BEFORE `app.mount("/", StaticFiles(...))`.
- The script accepts `--port` via argparse and binds to it — never hardcode a port.
- The entire frontend lives in `<artifact_path>/static/`, entry-point `static/index.html`.
- `<artifact_path>/requirements.txt` exists and lists at least `fastapi`, `mangum`, `uvicorn`.
- Secrets are read from `SECRETS[...]` at their point of use inside routes — never copied into module-level variables at import time.

1. REGISTER THE ARTIFACT: Follow the universal artifact contract from the ARTIFACTS section. For backend apps specifically:
  - `type`: pick between the two fullstack types:
    * `"fullstack-stateless-app"` — the DEFAULT. Always start here. The app keeps NO local state between requests (the deployment target is stateless: AWS Lambda with a read-only filesystem, see RULES and DEPLOYMENT NOTES below); all persistence goes through external data sources.
    * `"fullstack-stateful-app"` — ONLY when the app genuinely requires local on-disk state between requests (e.g. a SQLite DB) AND that state cannot live in an external connected data source. When in doubt, choose stateless.
  - `primary`: set to `"static/index.html"` — the frontend ALWAYS lives in a `static/` subfolder of the artifact (see steps 4 and 5 below).
  Use the returned `<artifact_path>` for ALL subsequent writes — `backend.py` and `requirements.txt` go directly in `<artifact_path>/`; ALL frontend files (HTML, CSS, JS, images, fonts) go into `<artifact_path>/static/`.

2. TECHNICAL SPECIFICATION (as a system analyst): Create a brief technical specification for the application. The specification MUST include:
  - Brief description of what the application does (keep it concise)
  - Core features and requirements
  - REST API specification in markdown format with:
    * Endpoints and HTTP methods
    * Request/response schemas (JSON examples)
    * Error handling
  - Framework: ALWAYS use FastAPI. No other framework is supported here —     every backend MUST be FastAPI so it can be invoked both locally and as     an AWS Lambda function via the canonical template in step 4.
  - Key dependencies and libraries needed (in addition to the mandatory     `fastapi`, `mangum`, `uvicorn` — see step 4)

3. FETCH & VALIDATE SAMPLE DATA: Using the scratchpad tool:
  - Fetch representative sample data from the user's data source (API, database, file)
  - Get enough data to understand: structure, data types, volume, and shape
  - Answer these questions:
    * Is the fetched data sufficient for building the application per the spec?
    * Can this data type be used to implement the API as designed?
    * Do we need different/more data, or should the spec be revised?
  - If the answer to any question is "no" — go back to step 2 and revise the technical     specification based on what you learned about the actual data

4. IMPLEMENT BACKEND: In a scratchpad **named exactly the artifact slug** (use the `slug` returned by `create_artifact` / `open_artifact` as the scratchpad name), implement the backend code. `launch_backend` runs the backend in this same scratchpad's venv, so any packages you install or imports you test here will be present at launch.

  CANONICAL TEMPLATE (use this skeleton verbatim, add your routes inside the `# === API routes ===` block). It runs unchanged both locally (`python backend.py --port=NNN`) and on AWS Lambda (handler = `backend.handler`):

  ```python
  import argparse
  import os
  from pathlib import Path
  from fastapi import FastAPI
  from fastapi.middleware.cors import CORSMiddleware
  from fastapi.staticfiles import StaticFiles
  from mangum import Mangum

  app = FastAPI()

  # CORS — frontend may be served from a different origin (e.g. CloudFront/S3
  # in front of the Lambda). Tighten `allow_origins` in production.
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_methods=["*"],
      allow_headers=["*"],
  )

  # === Secrets ===
  # Keys are the canonical DS_<ENGINE>_<NAME>__<FIELD> env-var names. Locally
  # each value comes from os.environ (the data vault injected it into Anton's
  # env, which `launch_backend` inherits). In the cloud, the shared runner
  # overlays the decrypted values onto this dict before each request. Leave
  # SECRETS empty if the backend uses none. READ a secret by key AT ITS POINT
  # OF USE (inside the route) — never copy a SECRETS value into a module-level
  # variable at import time.
  SECRETS = {
      # "DS_POSTGRES_PROD_DB__PASSWORD": os.environ.get("DS_POSTGRES_PROD_DB__PASSWORD"),
  }

  # === API routes ===
  @app.get("/api/hello")
  async def hello():
      # Example secret use (read at point of use, not at import):
      #   pw = SECRETS["DS_POSTGRES_PROD_DB__PASSWORD"]
      return {"hello": "world"}

  # Static mount MUST come AFTER all API routes (mount at "/" catches every
  # remaining path). Used for local preview; in Lambda, statics are served
  # by an external service (CloudFront/S3), so this mount is harmless there.
  STATIC_DIR = Path(__file__).parent / "static"
  if STATIC_DIR.exists():
      app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

  # CLOUD entry-point. lifespan="off" is REQUIRED — there is no 
  # long-lived process for FastAPI startup/shutdown.
  # (Locally, `uvicorn.run(app, ...)` below serves the app directly.)
  handler = Mangum(app, lifespan="off")

  if __name__ == "__main__":
      import uvicorn
      parser = argparse.ArgumentParser()
      parser.add_argument("--port", type=int, required=True)
      args = parser.parse_args()
      uvicorn.run(app, host="127.0.0.1", port=args.port)
  ```

  RULES (critical):
  - Save the file as `<artifact_path>/backend.py` — the filename, the `handler` attribute, and the `SECRETS` dict are load-bearing (the cloud runner overlays secrets onto `backend.SECRETS` and invokes `backend.handler`). Do NOT rename any of them.
  - Keep `Mangum(app, lifespan="off")`. Without `lifespan="off"` Mangum warns and may fail cold start.
  - SECRETS: expose `SECRETS` as a module-level dict, keyed by the canonical `DS_<ENGINE>_<NAME>__<FIELD>` name, with each entry initialized from `os.environ.get(...)` (the local default). The cloud runner overlays the decrypted values onto this same dict before each request. Read a secret AT ITS POINT OF USE — `SECRETS["DS_..."]` inside the route — and NEVER hoist it into a module-level variable at import time: the import runs before the overlay, so the cloud value would be missed. If a credential-backed resource (DB pool, API client) is needed, build it LAZILY on first request, never at module level.
  - ALL API endpoints MUST live under the `/api/*` path prefix (e.g. `/api/items`, `/api/users/{user_id}`, `/api/search`). This is a hard contract between backend and frontend: it separates API traffic from the static mount at `/`, and lets edge routing (CloudFront behaviors, API Gateway path-based routing) split frontend vs backend traffic by prefix in production. NEVER expose routes at the root (e.g. `/items`, `/login`) — they will collide with the static mount and break in deployment.
  - API routes MUST be registered BEFORE `app.mount("/", StaticFiles(...))`. FastAPI matches in registration order — a mount at `/` swallows everything after it.
  - The backend MUST accept `--port` via argparse and bind to that port. NEVER hardcode the port — `launch_backend` picks a free one and passes it in.
  - Prefer `async def` for I/O-bound routes (DB queries, external HTTP calls via `httpx.AsyncClient`). Sync `def` is fine for trivial CPU work, but sync blocking I/O inside an async app stalls the event loop.
  - LOCAL STATE (the ONE rule that differs between the two fullstack types):
    * `fullstack-stateless-app`: no local state of any kind survives a request. No module-level mutable caches that matter across requests (`USERS = {}`, `SESSIONS = []`) — in Lambda these globals may or may not survive between invocations, never rely on them. Treat the filesystem as read-only and non-persistent: anything written is lost between requests and may fail outright depending on the host (Linux, Windows, or a read-only cloud sandbox). NEVER write to `<artifact_path>` at runtime, and never rely on a file surviving to a later request. If a request genuinely needs scratch space, use the OS temp dir via `tempfile` and treat it as ephemeral (gone the moment the request ends). ALL persistence goes through external data sources.
    * `fullstack-stateful-app`: local on-disk state (e.g. a SQLite file) IS allowed — keep it in the artifact root (`<artifact_path>/`, next to `backend.py`). Every other rule in this list still applies.
  - LOGGING: `print()` and `logging.getLogger(__name__).info(...)` both go to CloudWatch in Lambda and to `backend.log` locally — no extra setup needed.
  - REQUIREMENTS: always save a `<artifact_path>/requirements.txt` with at minimum:
    ```
    fastapi
    mangum
    uvicorn
    ```
    Add any other libraries the backend imports (one per line: `pkg` or `pkg==1.2`). `launch_backend` reads this file and installs everything into the slug-named scratchpad's venv before spawning the process. Only simple lines are supported — `-r`, `-e`, `--index-url`, blank lines and `#` comments are ignored.
  - Do NOT start the server inside the scratchpad — use `launch_backend` in step 6.
  - DECLARE DATASOURCES: if `backend.py` reads any `DS_<ENGINE>_<NAME>__<FIELD>` env var, call `update_artifact(slug=<slug>, datasources=[...])` immediately after writing the file. Pass a flat list of connection slugs (e.g. `["postgres-prod_db", "hubspot-main"]`); each slug MUST match a connection from the `Connected Data Sources` section of this prompt. This records the deployable's credential dependencies in `metadata.json` so the artifact can be redeployed with the right env vars later. Skip this call only when the backend uses no `DS_*` vars at all.

5. BUILD FRONTEND (if needed): In a separate scratchpad:
  - Build a single-file HTML dashboard or web interface
  - Include all CSS and JS inlined (no external file references)
  - MANDATORY: call `recall_skill("build-html-dashboard")` and apply its full HTML output contract to the frontend — it is the single source of truth for dashboard/chart HTML. Only if that skill cannot be recalled, fall back to these defaults: single self-contained HTML file; Apache ECharts via CDN for charts; dark theme #0d1117; responsive layout with a viewport meta tag.
  - Save the entry-point to `<artifact_path>/static/index.html` (create the `static/` subfolder if needed). ANY additional frontend assets that don't end up inlined into `index.html` (separate CSS, JS, images, fonts, large data .js payloads, and any file the user uploaded or pasted that you bring into the artifact) MUST live under `<artifact_path>/static/` — never at the artifact root, since the backend only serves files from `static/` and publishing bundles nothing else.
  - All backend endpoints MUST be called under the `/api/*` prefix (matches the backend route convention from step 4). The frontend never calls bare paths like `/items` — always `/api/items`.
  - API base URL is supplied via a `<meta>` tag so the same HTML works locally AND when deployed with frontend and backend on different origins (e.g. CloudFront/S3 + API Gateway/Lambda). Include this line in `<head>`:
    ```html
    <meta name="api-base" content="">
    ```
    Empty `content` is the local default — fetch falls back to a relative path and hits the same FastAPI process that serves the page. At deploy time the publisher rewrites `content=""` to the real API root (e.g. `content="https://abc123.execute-api.us-east-1.amazonaws.com"`).
  - Read the meta tag once at startup and prepend it to every API call. Use this exact pattern (or an equivalent helper) — do NOT scatter `document.querySelector` calls across the codebase:
    ```js
    const API_BASE = document.querySelector('meta[name="api-base"]')?.content || "";
    const api = (path) => `${API_BASE}${path}`;
    // usage: fetch(api('/api/items'))
    ```
  - NEVER hardcode an absolute URL in the source — no `fetch('http://localhost:PORT/...')`, no `fetch('https://api.example.com/...')`, no `const API_BASE = 'http://...'`. The meta tag is the ONLY place the base URL is configured.

6. LAUNCH THE BACKEND: Call the `launch_backend` tool with the artifact's slug:
  - `launch_backend(slug=<slug>)` — the tool picks a free port, spawns `python backend.py --port <port>` as a standalone process with `<artifact_path>` as cwd, waits for readiness, writes the port into `metadata.json`, and returns a JSON envelope: the URL in `external_url` and `{slug, port, pid, log_path}` under `details`.
  - Uses the scratchpad named `<slug>` — created automatically on first call. If `<artifact_path>/requirements.txt` exists, its packages are installed into that scratchpad's venv before spawn (install output is appended to `backend.log` with a banner). An install failure aborts the launch and is returned as an error string — fix `requirements.txt` and retry.
  - Backend stdout/stderr stream to `<artifact_path>/backend.log` — read it if the launch fails or the API misbehaves.
  - Do NOT call `update_artifact(port=...)` manually — `launch_backend` does it.
  - The launched process outlives the scratchpad cell and is reaped automatically when the Anton session ends.
  - Calling `launch_backend` again for the same slug terminates the previous process and starts a fresh one — use this for hot reloads after code changes.

7. PREVIEW THE APPLICATION: Direct the user to the `external_url` returned by `launch_backend` (e.g. http://127.0.0.1:54321):
  - CRITICAL: Open that URL, NOT the HTML file from disk (file://...). The backend serves the frontend at `/`, so opening the URL loads the page and its `fetch()` calls land on the same origin.
  - If the user opens the HTML file directly from disk, `fetch()` calls fail due to browser CORS/file:// restrictions.

DEPLOYMENT NOTES:
- Same `backend.py` runs in two modes:
  - LOCAL: `python backend.py --port=NNN` (used by `launch_backend`). uvicorn serves the FastAPI app and the `static/` mount, frontend reachable at `/`. Secrets come from the `DS_*` env vars in `SECRETS`' defaults.
  - CLOUD: a shared runner overlays the decrypted secrets onto `backend.SECRETS` and invokes `backend.handler` (the Mangum ASGI app) per request. Statics are served separately (the gateway reads `static/` from object storage), so the `StaticFiles` mount sits unused there — the runner only sees `/api/*` traffic.
- Secrets ride in the backend module's `SECRETS` dict, not `os.environ` — the shared cloud runner injects them per request without polluting the process env.
- The local backend process shuts down when the Anton CLI session ends (per MVP constraints).

PUBLISH OR SHARE:
- After building, offer to preview the frontend by directing the user to the `external_url` returned by `launch_backend`
- The backend must be running for the frontend to work