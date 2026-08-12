from anton.core.tools.progress import ToolProgress
from anton.core.tools.tool_handlers import (
    handle_ask_user,
    handle_create_artifact,
    handle_generate_artifact,
    handle_generate_prd,
    handle_launch_backend,
    handle_list_artifacts,
    handle_memorize,
    handle_open_artifact,
    handle_read_image,
    handle_recall,
    handle_scratchpad,
    handle_select_path,
    handle_update_artifact_metadata,
)

from dataclasses import dataclass, replace
from typing import Callable, Optional


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict
    # async (session, tc_input) -> str | list[dict]. May instead be an async
    # generator yielding ToolProgress markers followed by one final
    # str | list[dict] — see anton/core/tools/progress.py and
    # ToolRegistry.dispatch_tool_stream (anton/core/tools/registry.py).
    handler: Callable
    prompt: Optional[str] = (
        None  # Optional prompt relevant to the tool to be injected into the system prompt.
    )
    # If set, the tool is deferred: not registered up front, but
    # unlocked when the model recalls the skill with this label. Lets hosts
    # tag their own tools as on-demand without a separate config map.
    unlock_skill: Optional[str] = None


SCRATCHPAD_TOOL = ToolDef(
    name="scratchpad",
    description=(
        "Run Python code in a persistent scratchpad. Use this whenever you need to "
        "count characters, do math, parse data, transform text, or any task that "
        "benefits from precise computation rather than guessing. Variables, imports, "
        "and data persist across cells — like a notebook you drive programmatically.\n\n"
        "Actions:\n"
        "- exec: Run code in the scratchpad (creates it if needed)\n"
        "- view: See all cells and their outputs\n"
        "- reset: Restart the process, clearing all state (installed packages survive)\n"
        "- remove: Kill the scratchpad and delete its environment\n"
        "- dump: Show a clean notebook-style summary of cells (code + truncated output)\n"
        "- install: Install Python packages into the scratchpad's environment. "
        "Packages persist across resets.\n\n"
        "IMPORTANT: Cells are kept alive automatically while the worker is running — "
        "deliberate sleeps and blocking calls (e.g. a throttled batch loop with "
        "time.sleep between sends) are safe in a single cell and are the preferred "
        "shape for batch work. A cell is killed only when its total time budget runs "
        "out or the worker itself dies or wedges; a kill loses the cell's state. "
        "You MUST provide estimated_execution_time_seconds for every exec call — it "
        "sizes the total budget (roughly 2x the estimate; without one the default "
        "budget is small). Call progress(message) to narrate long phases — it is "
        "user-visible status, not a survival requirement.\n\n"
        "Use print() to produce output. Host Python packages are available by default. "
        "Include a 'packages' array on exec calls for any libraries your code needs — "
        "they'll be auto-installed before the cell runs (already-installed ones are skipped).\n"
        "get_llm() returns a pre-configured LLM client (sync) — call "
        "llm.complete(system=..., messages=[...]) for AI-powered computation.\n"
        "llm.generate_object(MyModel, system=..., messages=[...]) extracts structured "
        "data into Pydantic models. Supports single models and list[Model].\n"
        "agentic_loop(system=..., user_message=..., tools=[...], handle_tool=fn) "
        "runs a tool-call loop where the LLM reasons and calls your tools iteratively. "
        "handle_tool(name, inputs) -> str is a plain sync function.\n"
        "web_search(query) routes a natural-language query (e.g. 'latest SpaceX IPO "
        "news') through the configured LLM's native web search and returns the "
        "model's narrative answer with source links, as a string.\n"
        "sample(var) inspects any variable with type-aware formatting — DataFrames get "
        "shape/dtypes/head, dicts get keys/values, lists get length/items. "
        "Defaults to 'preview' mode (compact); use sample(var, mode='full') for complete dump.\n"
        "All .anton/.env secrets are available as environment variables (os.environ)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["exec", "view", "reset", "remove", "dump", "install"],
            },
            "name": {"type": "string", "description": "Scratchpad name"},
            "code": {
                "type": "string",
                "description": "Python code (exec only). Use print() for output.",
            },
            "packages": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Package names needed by this cell (exec or install). "
                "Listed after code so you know exactly what to include. "
                "Already-installed packages are skipped automatically.",
            },
            "one_line_description": {
                "type": "string",
                "description": "Brief description of what this cell does (e.g. 'Scrape listing prices'). Required for exec.",
            },
            "estimated_execution_time_seconds": {
                "type": "integer",
                "description": "Estimated execution time in seconds. Drives the total time budget (roughly 2x estimate).",
            },
            "confirm_new_scratchpad": {
                "type": "boolean",
                "description": "Set true only to deliberately create a SECOND scratchpad while one is already in use this task. Normally reuse one scratchpad name for the whole task — each name is a separate isolated environment, so a new one loses all existing state. Leave unset/false unless you truly need isolation.",
            },
        },
        "required": ["action", "name"],
    },
    handler=handle_scratchpad,
)


MEMORIZE_TOOL = ToolDef(
    name="memorize",
    description=(
        "Encode a rule or lesson into long-term memory for future sessions. "
        "Use this when you learn something important, discover a useful pattern, "
        "or the user asks you to remember something.\n\n"
        "Entry kinds:\n"
        "- always: Something to always do ('Use httpx instead of requests')\n"
        "- never: Something to never do ('Never use time.sleep() in scratchpad')\n"
        "- when: Conditional rule ('If paginated API → use async + progress()')\n"
        "- lesson: Factual knowledge ('CoinGecko rate-limits at 50/min')\n"
        "- profile: Fact about the user ('Name: Jorge', 'Prefers dark mode')"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The memory to encode",
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["always", "never", "when", "lesson", "profile"],
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["global", "project"],
                        },
                        "topic": {
                            "type": "string",
                            "description": "Topic slug for lessons (e.g. 'api-coingecko')",
                        },
                    },
                    "required": ["text", "kind", "scope"],
                },
            },
        },
        "required": ["entries"],
    },
    handler=handle_memorize,
)


CREATE_ARTIFACT_TOOL = ToolDef(
    name="create_artifact",
    description=(
        "Claim a folder for a user-facing output (HTML dashboard, document, "
        "dataset, image, fullstack app, etc.). Call this BEFORE writing the "
        "files — the tool returns the absolute folder path you should write "
        "into. Each artifact gets its own subfolder under `<workspace>/artifacts/`, "
        "with a `metadata.json` + `README.md` written automatically.\n\n"
        "AFTER REGISTERING a web artifact (html-app, fullstack-stateless-app, "
        "or fullstack-stateful-app): call `generate_prd(slug, user_request, "
        "agent_understanding, ...)` first to draft and confirm a PRD, then "
        "call `generate_artifact(slug, context)` and let it write every "
        "file using that PRD as the requirements source — that is the "
        "normal path and it verifies its own output. For `document`, "
        "`dataset`, `image` and `mixed` there is no PRD step and no "
        "generator: write the files yourself into the returned path.\n\n"
        "If you do end up building one of the three generator-backed types BY "
        "HAND (editing an existing artifact, or `generate_artifact` failed and "
        "the user asked you to continue), the output contract lives in a skill: "
        "call `recall_skill(\"build-html-dashboard\")` for an html-app or "
        "`recall_skill(\"build-fullstack-backend\")` for a fullstack app before "
        "writing anything. Hand-written code that skips it breaks "
        "rendering/launch.\n\n"
        "Pick `type` from the closed enum:\n"
        "- html-app: a single self-contained HTML page (charts, dashboards, demos)\n"
        "- document: a doc, report, or markdown file the user reads\n"
        "- dataset: data files (CSV, JSON, parquet) the user downloads or feeds elsewhere\n"
        "- image: a generated image (PNG, SVG, etc.)\n"
        "- mixed: multi-modal output that doesn't fit the above\n"
        "- fullstack-stateless-app: fullstack web app (backend + frontend) that keeps "
        "no local state between requests; all persistence goes to external data sources. "
        "Reading from or writing to an external database (Postgres, MySQL, etc.) is "
        "stateless — the external DB is a data source, not local state. PREFER this type: "
        "use it for anything that just queries/serves external data.\n"
        "- fullstack-stateful-app: fullstack web app (backend + frontend) that keeps "
        "local state between requests inside the artifact itself (e.g. an on-disk SQLite "
        "DB, a file the backend writes and re-reads). Choose this ONLY when the app must "
        "persist its own state locally between requests; otherwise use "
        "fullstack-stateless-app.\n\n"
        "Pass `primary` (optional) when you already know the entry-point "
        "filename you'll write — e.g. `\"dashboard.html\"` for an html-app, "
        "`\"static/index.html\"` for a fullstack app, `\"report.pdf\"` for a "
        "document. The renderer uses it to decide what to open by default. "
        "Skip when you don't know yet — the renderer falls back to a "
        "heuristic, and you can set it later via `update_artifact`.\n\n"
        "To MODIFY an existing artifact instead of creating a new one, call "
        "`list_artifacts` first to find it, then `open_artifact(slug)` to get "
        "the path."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Human-readable artifact name. The folder slug is derived from this.",
            },
            "description": {
                "type": "string",
                "description": "Short description of what the artifact is. Shown in the UI and in the README.",
            },
            "type": {
                "type": "string",
                "enum": [
                    "html-app",
                    "document",
                    "dataset",
                    "image",
                    "mixed",
                    "fullstack-stateless-app",
                    "fullstack-stateful-app",
                ],
            },
            "primary": {
                "type": "string",
                "description": "Relative path of the entry-point file you'll write (e.g. \"dashboard.html\"). Optional — skip if you don't know yet.",
            },
        },
        "required": ["name", "description", "type"],
    },
    handler=handle_create_artifact,
)


UPDATE_ARTIFACT_METADATA_TOOL = ToolDef(
    name="update_artifact",
    description=(
        "Update mutable fields on an existing artifact. Pass only the fields you want to change.\n\n"
        "- `primary`: relative path of the entry-point file (e.g. \"index.html\"). "
        "Pass empty string to clear (renderer reverts to heuristic: "
        "`index.html` → newest `.html` → newest non-housekeeping file).\n"
        "- `port`: port the backend process is listening on (fullstack apps only). "
        "Normally written automatically by `launch_backend` — set manually only "
        "if you started the server some other way.\n"
        "- `datasources`: list of vault-connection slugs the artifact's backend "
        "reads from (e.g. `[\"postgres-prod_db\", \"hubspot-main\"]`). REQUIRED "
        "for fullstack apps whose `backend.py` references any "
        "`DS_<ENGINE>_<NAME>__<FIELD>` env var — declare it right after writing "
        "`backend.py` so metadata.json "
        "captures which connections the deployable depends on. Slugs must match "
        "existing vault connections (see `Connected Data Sources` in the system "
        "prompt). Pass `[]` to clear."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "Folder slug of the artifact to update.",
            },
            "primary": {
                "type": "string",
                "description": "Relative path of the entry-point file. Empty string to clear.",
            },
            "port": {
                "type": "integer",
                "description": "Port number the backend process is listening on.",
            },
            "datasources": {
                "type": "array",
                "description": (
                    "Vault-connection slugs the backend reads from. Replaces "
                    "the existing list — pass the full set every time. Use "
                    "`[]` to clear."
                ),
                "items": {
                    "type": "string",
                    "description": "Connection slug, e.g. \"postgres-prod_db\".",
                },
            },
        },
        "required": ["slug"],
    },
    handler=handle_update_artifact_metadata,
)


LIST_ARTIFACTS_TOOL = ToolDef(
    name="list_artifacts",
    description=(
        "List every artifact in the current workspace (newest first). "
        "Use this to find an existing artifact you want to modify — paired "
        "with `open_artifact(slug)` for the actual edit. Each entry includes "
        "the slug, human name, type, description, file count, and last-update "
        "timestamp. Returns an empty list when no artifacts exist yet."
    ),
    input_schema={
        "type": "object",
        "properties": {},
    },
    handler=handle_list_artifacts,
)


OPEN_ARTIFACT_TOOL = ToolDef(
    name="open_artifact",
    description=(
        "Load an existing artifact by slug. Returns the folder path plus the "
        "list of files so you can decide what to edit. Combine with the "
        "scratchpad to read existing files (`open(path).read()`) or write "
        "updates back into the folder. Provenance is updated automatically — "
        "every turn that modifies a file in the folder is appended to the "
        "artifact's metadata.json."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "Folder slug (returned by `list_artifacts` or the previous `create_artifact`).",
            },
        },
        "required": ["slug"],
    },
    handler=handle_open_artifact,
)


LAUNCH_BACKEND_TOOL = ToolDef(
    name="launch_backend",
    description=(
        "Start an artifact's backend script as a standalone subprocess. "
        "Picks a free TCP port, runs the script with `--port <port>` "
        "(plus any `extra_args`), waits until the server is reachable, "
        "records the port in the artifact's `metadata.json`, and returns "
        "`{slug, port, pid, url, log_path}` as JSON.\n\n"
        "You normally do NOT call this: `generate_artifact` launches the backend "
        "itself and records the port. Use this tool for a hand-built backend, or "
        "to restart one after editing its code.\n\n"
        "The backend MUST follow the contract in the `build-fullstack-backend` "
        "skill (template, `--port`, `/api/*` prefix, SECRETS) — a hand-written "
        "backend that skips it will not start.\n\n"
        "The spawned process inherits Anton's environment, including the "
        "`DS_<ENGINE>_<NAME>__<FIELD>` variables of connected data sources.\n\n"
        "Runs in a scratchpad named exactly `<slug>` (created on first call). "
        "If `<artifact_folder>/requirements.txt` exists, its package lines are "
        "installed into that scratchpad's venv before spawn — install output "
        "appended to `backend.log`, install failures abort the launch and are "
        "returned as an error string. Only simple lines are supported "
        "(`pkg` / `pkg==1.2`); blank lines, `#` comments, and `-`-prefixed "
        "flags (`-r`, `-e`, `--index-url`) are ignored.\n\n"
        "Idempotent: a second call with the same slug terminates the "
        "previously-launched backend before starting a new one.\n\n"
        "Requirements on the backend script:\n"
        "- MUST accept `--port` via argparse (or equivalent) and bind to it.\n"
        "- MUST be reachable at `health_path` (default `/`) within "
        "`health_timeout` seconds.\n"
        "- stdout/stderr stream to `<artifact>/backend.log`."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "Folder slug of the artifact whose backend to launch.",
            },
            "path": {
                "type": "string",
                "description": "Backend script path relative to the artifact folder. Default: \"backend.py\".",
            },
            "extra_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional CLI arguments appended after `--port <port>`.",
            },
            "health_path": {
                "type": "string",
                "description": "URL path for the readiness probe. Default: \"/\". Any HTTP response (including 4xx) counts as ready.",
            },
            "health_timeout": {
                "type": "number",
                "description": "Seconds to wait for readiness before failing. Default: 10.",
            },
        },
        "required": ["slug"],
    },
    handler=handle_launch_backend,
)


GENERATE_PRD_TOOL = ToolDef(
    name="generate_prd",
    description=(
        "Draft and get user confirmation on a PRD (Product Requirements "
        "Document) for an already-registered web artifact (html-app, "
        "fullstack-stateless-app, fullstack-stateful-app), BEFORE any code "
        "is written. Runs a bounded internal process: determines the "
        "artifact type, gathers/verifies any data needed (may call "
        "scratchpad, web_search, web_fetch, and ask the user clarifying "
        "questions internally), drafts a short brief, and shows it to the "
        "user for accept/cancel/revise. On acceptance, writes the full "
        "`prd.md` into the artifact folder.\n\n"
        "Inputs:\n"
        "- `slug`: the artifact slug from a prior `create_artifact` call.\n"
        "- `user_request`: the user's request, as close to their original "
        "wording as possible.\n"
        "- `agent_understanding`: how you understand the task, based on "
        "the whole conversation so far.\n"
        "- `known_data` (optional): anything already known about the data "
        "needed — descriptions, or scratchpad references (pad name, cell) "
        "if you already fetched something.\n"
        "- `user_preferences` (optional): relevant known preferences "
        "(style, preferred APIs, etc.).\n\n"
        "Returns one of three shapes:\n"
        '- `{"status": "prd_written", "prd_path", "artifact_type", '
        '"brief_summary", "qa_log"}` — the user confirmed; proceed to '
        "build the artifact using `prd_path` as the requirements source.\n"
        '- `{"status": "prd_written_unconfirmed", ...}` — a draft was '
        "written but NOT confirmed (the per-turn question budget ran "
        "out); do NOT proceed to building the artifact, show "
        "`brief_summary` and get explicit confirmation first.\n"
        '- `{"status": "cancelled", "reason", "qa_log"}` — the user '
        "declined; do NOT write a PRD yourself and do NOT proceed.\n\n"
        "On a technical failure, do NOT build the PRD or the artifact "
        "yourself — report the failure to the user."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "Slug of an already-registered artifact.",
            },
            "user_request": {
                "type": "string",
                "description": "The user's request, as close to their original wording as possible.",
            },
            "agent_understanding": {
                "type": "string",
                "description": "How you understand the task, based on the whole conversation.",
            },
            "known_data": {
                "type": "string",
                "description": (
                    "What's already known about needed data — descriptions "
                    "or scratchpad references (pad name, cell)."
                ),
            },
            "user_preferences": {
                "type": "string",
                "description": "Relevant known user preferences (style, preferred APIs, etc.).",
            },
        },
        "required": ["slug", "user_request", "agent_understanding"],
    },
    handler=handle_generate_prd,
)



GENERATE_ARTIFACT_TOOL = ToolDef(
    name="generate_artifact",
    description=(
        "Populate an already-registered artifact's folder via a dedicated "
        "sub-generator. Use INSTEAD OF writing files yourself in the "
        "scratchpad. Reads `type` from the artifact's metadata (must be "
        "`html-app`, `fullstack-stateless-app`, or `fullstack-stateful-app`).\n\n"
        "Inputs:\n"
        "- `slug`: the artifact slug from a prior `create_artifact` call.\n"
        "- `context`: a markdown brief with these four sections:\n"
        "  ## User request — the user's literal ask\n"
        "  ## Conversation context — relevant decisions/history from this chat\n"
        "  ## Functional Requirements Specification — what the system does from "
        "the user's point of view: what the user sees on screen, how they "
        "interact with it, and what result they get. MUST be technology-agnostic — "
        "do NOT mention frameworks, libraries, endpoints, HTTP methods, env vars, "
        "database engines, file paths, CSS colours, fonts, or any implementation "
        "detail. Describe behaviour and user-visible outcomes, not how to build them. "
        "For simple tasks a short plain-language description is enough.\n"
        "  ## Data — free-form, list everything known at call time about the "
        "data sources needed for the task and their properties: what each "
        "source is and where it conceptually lives (e.g. \"PostgreSQL "
        "`integration` table\", \"CoinGecko `/coins/markets` endpoint\"), the "
        "schema/columns with types, row counts, and any stable contextual "
        "facts that help frame the data. Include only what is needed and only "
        "what is already known. IF you have already interacted with the data "
        "sources, you MUST also name the scratchpad(s), the specific cells, and "
        "what exactly was done in them — which query/extraction ran, which "
        "variables were produced, and what the result showed — because the "
        "generator can open those scratchpads and will pull or rebuild the data "
        "itself. You may include a `### Sample` subsection with 2–5 real rows "
        "if you have actually observed them. DO NOT include env var names, "
        "connection strings, credentials, API endpoint paths the generator "
        "must implement, backend file layout, or any other implementation "
        "detail — those belong in the generator's own planning step, not in "
        "this brief.\n\n"
        "For fullstack apps the tool launches the backend itself (health-checked "
        "on `/api/health`) and records the port in metadata; you do NOT call "
        "`launch_backend` yourself. On success it returns "
        "`{slug, path, files_written, internal_files, summary, trace}` — "
        "`files_written` are the artifact's own files (report those to the "
        "user), `internal_files` are generation inputs like `spec.md` and "
        "`openapi.json` (do NOT present them as deliverables), and `trace` "
        "lists the generation steps and their outcomes; on failure a single error string "
        "naming the node that stopped the run. If generation fails, report the "
        "failure to the user and ask how to proceed — NEVER fall back to "
        "building the artifact yourself via scratchpad or any other means."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "Slug of an already-registered artifact.",
            },
            "context": {
                "type": "string",
                "description": (
                    "Markdown brief with sections `## User request`, "
                    "`## Conversation context`, "
                    "`## Functional Requirements Specification`, `## Data`. "
                    "The FRS section MUST describe behaviour from the user's "
                    "point of view (what they see, how they interact, what "
                    "result they get) and MUST NOT mention technologies, "
                    "frameworks, endpoints, env vars, file paths, colours, "
                    "or any implementation detail. "
                    "The `## Data` section is free-form: list everything known "
                    "about the needed data sources and their properties "
                    "(source, schema, types, row counts), and — if you already "
                    "interacted with the sources — name the scratchpad(s), the "
                    "cells, and what was done in them (the generator can open "
                    "those scratchpads to pull/rebuild the data). NO env var "
                    "names, connection details, API endpoint paths, or backend "
                    "layout. Add a `### Sample` of 2–5 real rows only if you "
                    "actually observed them (never fabricate). "
                    "Passed verbatim to the sub-generator."
                ),
            },
        },
        "required": ["slug", "context"],
    },
    handler=handle_generate_artifact,
    prompt=(
        "ARTIFACT GENERATION:\n"
        "`generate_artifact` produces every file for an already-registered "
        "artifact by running a strict internal generation state machine "
        "(data-sufficiency check → technical spec → REST API spec → backend & "
        "frontend generation with verification → launch & health check). Use it "
        "INSTEAD of writing artifact files yourself in the scratchpad.\n"
        "  1. Call `create_artifact` to register the slug and pick the type "
        "(one of html-app, fullstack-stateless-app, fullstack-stateful-app).\n"
        "  2. Call `generate_artifact(slug=<slug>, context=<markdown brief>)`.\n"
        "     - `context` MUST be a markdown document with these four sections:\n"
        "         ## User request\n"
        "         ## Conversation context\n"
        "         ## Functional Requirements Specification\n"
        "         ## Data\n"
        "       For `## Functional Requirements Specification`: describe ONLY "
        "user-facing behaviour:\n"
        "         • what the user sees on screen (content, structure, states),\n"
        "         • how the user interacts with it (clicks, inputs, navigation),\n"
        "         • what result the user gets in response (output, feedback, "
        "error states).\n"
        "       DO NOT mention any of the following in this section:\n"
        "         • technologies, frameworks, or libraries (FastAPI, ECharts, "
        "psycopg2, React, etc.),\n"
        "         • system architecture, file layout, or module boundaries,\n"
        "         • API endpoints, HTTP methods, request/response shapes, "
        "or status codes,\n"
        "         • database engines, table/column names, SQL, ORM details,\n"
        "         • environment variables, secrets, config keys,\n"
        "         • CSS colours, fonts, exact pixel sizes, or other styling "
        "internals (general phrases like \"dark theme\" or \"responsive\" are fine),\n"
        "         • any other implementation detail or technical constraint.\n"
        "       Use plain language a non-technical user would understand. Names "
        "of real-world entities the user knows about (e.g. \"companies\", "
        "\"integrations\") are fine; internal column names and engine slugs "
        "belong in `## Data`, not here. For simple tasks a short plain "
        "description is enough.\n"
        "       For `## Data`: free-form — list everything known about the data "
        "sources needed and their properties (the source and where it "
        "conceptually lives, schema/columns with types, row counts, stable "
        "facts). IF you already interacted with the sources, you MUST name the "
        "scratchpad(s), the specific cells, and what was done in them (which "
        "query/extraction ran, which variables were produced, what the result "
        "showed) — the generator can open those scratchpads and will pull or "
        "rebuild the data itself. Add a `### Sample` of 2–5 real rows only if "
        "you actually observed them; never fabricate one. DO NOT mention in "
        "this section:\n"
        "         • env var names (DS_POSTGRES_*, API_KEY, etc.) or any "
        "credentials/connection strings,\n"
        "         • API endpoint paths the generator must implement "
        "(`GET /api/...`),\n"
        "         • backend file layout, modules, or implementation details.\n"
        "       Those are the generator's job to design — not yours to dictate.\n"
        "For fullstack apps the tool launches the backend itself and records the "
        "port — you do NOT need to call `launch_backend` afterwards."
    ),
)



RECALL_TOOL = ToolDef(
    name="recall",
    description=(
        "Search your episodic memory — an archive of past conversations. "
        "ONLY use this when the user explicitly asks about a previous conversation "
        "or session (e.g. 'what did we talk about last time?', 'remember when we...', "
        "'have we discussed X before?'). Do NOT use this for questions about code, "
        "files, or data in the workspace — use the scratchpad to explore those directly.\n\n"
        "Returns timestamped episodes matching the query (newest first). "
        "A single call is enough — do not call multiple times with different queries."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search term to find in past conversations.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum episodes to return (default 20).",
            },
            "days_back": {
                "type": "integer",
                "description": "Only search episodes from the last N days.",
            },
        },
        "required": ["query"],
    },
    handler=handle_recall,
)


READ_IMAGE_TOOL = ToolDef(
    name="read_image",
    description=(
        "Read an image file from disk so you can see its contents. Use this "
        "whenever the user references a path to an image file (PNG, JPG, "
        "JPEG, GIF, WEBP, BMP) and you need to actually view the picture to "
        "answer. Pass `file_path` as an absolute path or a path relative to "
        "the current working directory. The image will appear in your next "
        "turn as a vision input — do not call this tool again for the same "
        "path within one turn."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the image file.",
            },
        },
        "required": ["file_path"],
    },
    handler=handle_read_image,
)


SELECT_PATH_TOOL = ToolDef(
    name="select_path",
    description=(
        "Show the user an inline picker to choose a file or folder, and get back "
        "the absolute path. Two modes, chosen by what you pass:\n\n"
        "• BROWSE — the location is unknown or the user only referred to it vaguely "
        "(e.g. 'a folder somewhere', 'my downloads', 'the project I mentioned'). Call "
        "with just a `prompt` (and `kind`); the user navigates a picker to locate it. "
        "Use this INSTEAD of asking the user to type or paste a path. Optionally set "
        "`start_dir` to seed the starting folder.\n"
        "• PICK — you already found several matches and need the user to disambiguate. "
        "Pass an explicit `candidates` list, OR a glob `pattern` (optionally under "
        "`base_dir`) to find matches within the project. Exactly one match resolves "
        "immediately with no prompt; zero matches tells you to refine.\n\n"
        'On selection the tool returns {"status":"resolved","path":"<absolute path>"} '
        "— use that path directly and keep going. Other statuses: 'cancelled' (user "
        "dismissed), 'no_matches', 'invalid', and 'picker_unavailable' (this host "
        "cannot render a picker). On 'picker_unavailable' follow the `message` in the "
        "result: in PICK mode it carries the `candidates` it found, so ask which of "
        "those the user meant; in BROWSE mode the file is somewhere you cannot reach, "
        "so ask the user to attach it to the conversation. Never re-ask in plain text "
        "after a resolved selection."
    ),
    # Injected into the system prompt: bias the model toward the picker over a
    # type-the-path request, which is the whole point of the tool.
    prompt=(
        "When the user refers to a file or folder without giving a path you can "
        "confidently resolve, call the `select_path` tool to let them pick it — do "
        "NOT ask them to paste or type a path, and do not guess. Browse mode (no "
        "candidates/pattern) is the right choice when you don't know where it is."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "One short line telling the user what to choose, "
                "e.g. 'Pick the folder to check' or 'Which \"report.csv\" did you mean?'.",
            },
            "kind": {
                "type": "string",
                "enum": ["file", "folder", "any"],
                "description": "What the user should choose. Default 'any'.",
            },
            "start_dir": {
                "type": "string",
                "description": "BROWSE mode only: directory to open the picker at "
                "(absolute, or relative to the project root). Defaults to the project root.",
            },
            "candidates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "PICK mode: explicit candidate paths (absolute, or "
                "relative to the project root) you have already identified.",
            },
            "pattern": {
                "type": "string",
                "description": "PICK mode: glob to find candidates within the project "
                "(e.g. '**/config.json'). Used when `candidates` is omitted.",
            },
            "base_dir": {
                "type": "string",
                "description": "PICK mode: directory to resolve `pattern` against, "
                "relative to the project root. Defaults to the project root.",
            },
        },
        "required": ["prompt"],
    },
    handler=handle_select_path,
)


# Variant registered when no elicitor on this host supports `kind="path"`
# (cowork-server injects none, so this is every cowork session today).
#
# BROWSE mode is physically impossible there — handle_select_path returns
# picker_unavailable before it ever reaches the elicitor — yet the full
# definition above advertises browse as the right move for an unknown location
# AND injects a system-prompt rule forbidding the alternatives. That
# combination is what routed a user who said "I have my sales data" into a dead
# end with no legitimate exit: picker can't render, asking for a path is
# forbidden here and by the harness file-access policy, guessing is forbidden.
# The model resolved it by inventing the data (ENG-1357).
#
# PICK mode still earns its place without an elicitor: exactly one match
# auto-resolves with no user interaction, and ≥2 returns the candidate list so
# the agent can ask in plain text — legitimate, because those paths are ones it
# already found inside the project, not a path the user must type from memory.
SELECT_PATH_TOOL_PICK_ONLY = replace(
    SELECT_PATH_TOOL,
    description=(
        "Disambiguate between file/folder paths you have ALREADY located inside "
        "the project. Pass an explicit `candidates` list, OR a glob `pattern` "
        "(optionally under `base_dir`).\n\n"
        "Exactly one match resolves immediately and you get the path back. Zero "
        "matches tells you to refine. Two or more comes back as "
        "'picker_unavailable' WITH the candidates it found — ask the user which "
        "of those they meant.\n\n"
        "This host cannot render an interactive file browser, so there is no "
        "BROWSE mode: you cannot ask the user to navigate to a file whose "
        "location you do not know. If the file is not in the project, ask the "
        "user to attach it to the conversation.\n\n"
        'Returns {"status":"resolved","path":"<absolute path>"} on success. '
        "Other statuses: 'no_matches', 'invalid', 'picker_unavailable'. Never "
        "re-ask in plain text after a resolved selection."
    ),
    prompt=(
        "When the user refers to a file or folder without giving a path you can "
        "confidently resolve, first look for it inside the project — then call "
        "`select_path` with `candidates` or a `pattern` to have the user pick "
        "between the matches. Do not guess which one they meant. If the file is "
        "not in the project at all, ask the user to attach it to the "
        "conversation; this host has no file browser, so do not ask them to "
        "navigate to it and do not ask them to type or paste a path."
    ),
    # `replace()` copies input_schema by reference, so the schema must be
    # overridden too — otherwise the model is told two different things by the
    # two channels it reads: prose saying "there is no BROWSE mode" and a
    # function-calling schema still offering `start_dir`, whose own description
    # reads "BROWSE mode only". Dropping it is the point of the variant.
    #
    # Rebuilt rather than mutated, for the same reason the ToolDef itself is
    # copied: `SELECT_PATH_TOOL.input_schema` is a module-level dict shared by
    # every session in the process. The inner property dicts are shared but
    # never written, so a one-level rebuild is enough.
    input_schema={
        **SELECT_PATH_TOOL.input_schema,
        "properties": {
            key: value
            for key, value in SELECT_PATH_TOOL.input_schema["properties"].items()
            if key != "start_dir"
        },
    },
)


ASK_USER_TOOL = ToolDef(
    name="ask_user",
    description=(
        "Ask the user to choose between concrete options, and get their answer "
        "back as the tool result within this same turn. Use this INSTEAD of "
        "writing the question as text and ending your turn, whenever the "
        "answers form a short closed set (which database, which table, which "
        "of these three approaches).\n\n"
        "Give 2-10 options with unique `value`s. `value` is what comes back to "
        "you; `label` is what the user sees. Set `select` to 'many' when more "
        "than one answer makes sense.\n\n"
        'Returns {"status":"answered","values":["<value>"]} — or "text" when the '
        'user typed their own answer instead of picking, possibly alongside '
        '"values". Other statuses: "cancelled" (the user declined to answer), '
        '"timeout", "error". On cancelled/timeout/error do NOT call this tool '
        "again for the same question: either proceed on an assumption you state "
        "out loud, or ask in plain text and end your turn.\n\n"
        "Ask one question at a time."
    ),
    # This is where the carve-out from CONVERSATION DISCIPLINE's "never ask and
    # act in the same turn" rule lives. It belongs here rather than in the
    # discipline text because that text is injected unconditionally, while
    # `ask_user` is registered only when an elicitor advertises "choice" —
    # headless runs, the telegram adapter and goal mode have no such tool, and
    # a prompt commanding a tool that is not in the tool list is worse than no
    # prompt. `ToolDef.prompt` is emitted only for registered tools, which is
    # exactly the condition needed.
    prompt=(
        "HOW to ask depends on the shape of the answer. When the answers form a "
        "short closed set (which database, which table, which of these three "
        "approaches), call the `ask_user` tool instead of writing the options as "
        "text and stopping: the answer comes back as the tool result, so you keep "
        "working in the SAME turn. The conversation-discipline rule about stopping "
        "after you ask applies to questions you write as TEXT — an `ask_user` "
        "answer arrives inside the current turn, so continue with it immediately. "
        "Open-ended questions still go in plain text with the turn ended. Ask one "
        "question at a time either way. If `ask_user` comes back cancelled, "
        "timeout or error, do not re-ask: proceed on an assumption you state out "
        "loud, or ask in plain text and end your turn."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "One short line asking what to choose, e.g. "
                "'Which database should I read from?'.",
            },
            "options": {
                "type": "array",
                "minItems": 2,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "string",
                            "description": "What is returned to you on selection. Unique.",
                        },
                        "label": {
                            "type": "string",
                            "description": "What the user sees. Defaults to `value`.",
                        },
                        "detail": {
                            "type": "string",
                            "description": "Optional second line of context.",
                        },
                    },
                    "required": ["value"],
                },
                "description": "The choices, 2-10 of them.",
            },
            "select": {
                "type": "string",
                "enum": ["one", "many"],
                "description": "Whether the user picks one option or several. Default 'one'.",
            },
            "allow_custom": {
                "type": "boolean",
                "description": "Whether a free-form answer is useful here. Default true.",
            },
        },
        "required": ["question", "options"],
    },
    handler=handle_ask_user,
)
