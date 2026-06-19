from anton.core.tools.tool_handlers import (
    handle_create_artifact,
    handle_generate_artifact,
    handle_launch_backend,
    handle_list_artifacts,
    handle_memorize,
    handle_open_artifact,
    handle_read_image,
    handle_recall,
    handle_scratchpad,
    handle_update_artifact_metadata,
)

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict
    handler: Callable  # async (session, tc_input) -> str
    prompt: Optional[str] = (
        None  # Optional prompt relevant to the tool to be injected into the system prompt.
    )


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
        "All .anton/.env secrets are available as environment variables (os.environ).\n\n"
        "IMPORTANT: Cells have an inactivity timeout of 30 seconds — if a cell produces "
        "no output and no progress() calls for 30s, it is killed and all state is lost. "
        "For long-running code (API calls, data extraction, heavy computation), call "
        "progress(message) periodically to signal work is ongoing and reset the timer. "
        "The total timeout scales from your estimated_execution_time_seconds "
        "(roughly 2x the estimate). You MUST provide estimated_execution_time_seconds "
        "for every exec call. For very long operations, provide a realistic estimate "
        "and use progress() to keep the cell alive."
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
                "description": "Estimated execution time in seconds. Drives the total timeout (roughly 2x estimate). Use progress() for long cells.",
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
        "  ## Data — describe ONLY the user-facing data: the source (what it is, "
        "where it lives conceptually — e.g. \"PostgreSQL `integration` table\", "
        "\"CoinGecko `/coins/markets` endpoint\"), the schema/columns with their "
        "types, row counts, and any stable contextual facts that help frame the "
        "data. Include only what is needed and only what is already known. "
        "Include a `### Sample` subsection with a small sample of real rows "
        "(2–5 rows, or a representative excerpt for non-tabular data) — but "
        "ONLY if you have actually observed the sample earlier (queried the "
        "DB, fetched the API, loaded the file). Do NOT fabricate a sample, "
        "and do NOT include the subsection at all when no sample exists yet. "
        "Overlap with `data_refs` is intentional — the inline sample makes the "
        "brief self-contained and human-readable; `data_refs` separately gives "
        "the generator a raw pickle. DO NOT include env var names, connection "
        "strings, credentials, API endpoint paths the generator must implement, "
        "backend file layout, or any other implementation detail — those "
        "belong in the generator's own planning step, not in this brief.\n"
        "- `data_refs`: scratchpad variables that hold the actual data "
        "(DataFrame, list, dict, fetched JSON, etc.) as "
        "`[{\"scratchpad\": \"<name>\", \"variable\": \"<py_identifier>\"}]`. "
        "Each variable is dill-pickled into a sidecar file the generator can "
        "load at write time, AND a short type-aware summary (shape, dtypes, "
        "head(5) for DataFrames; key list for dicts; first items for lists) is "
        "embedded into the generator's prompt automatically. REQUIRED whenever "
        "you have inspected the user's data in a scratchpad (queried the DB, "
        "fetched an API sample, loaded a file): pass the variable holding the "
        "sample/dataframe so the generator sees the real shape instead of just "
        "your textual description. Only pass `[]` when no concrete data has "
        "been observed yet.\n\n"
        "Returns `{slug, path, files_written, rounds_used, summary}` on success "
        "or a plain error string on failure. The tool DOES NOT call "
        "`launch_backend` — for `fullstack-stateful-app`, you still call it "
        "yourself after this returns."
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
                    "The `## Data` section MUST contain only user-data info "
                    "(source, schema, types, row counts) — NO env var names, "
                    "connection details, API endpoint paths, or backend layout. "
                    "Include a `### Sample` subsection with 2–5 real rows ONLY "
                    "if you have actually observed the sample earlier; omit the "
                    "subsection entirely otherwise (never fabricate it). "
                    "Passed verbatim to the sub-generator."
                ),
            },
            "data_refs": {
                "type": "array",
                "description": (
                    "Scratchpad variables holding observed data (samples, "
                    "dataframes, fetched JSON, etc.) for the generator to "
                    "consume. Each item: {scratchpad: <pad name>, variable: "
                    "<python identifier>}. REQUIRED whenever you have inspected "
                    "the user's data in a scratchpad — pass the variable so the "
                    "generator sees real shape, not just your description. Pass "
                    "`[]` only when no concrete data has been observed."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "scratchpad": {"type": "string"},
                        "variable": {"type": "string"},
                    },
                    "required": ["scratchpad", "variable"],
                },
            },
        },
        "required": ["slug", "context"],
    },
    handler=handle_generate_artifact,
    prompt=(
        "ARTIFACT GENERATION GATE (experimental):\n"
        "You have access to `generate_artifact`, which produces every file "
        "for an already-registered artifact in a single tool call (it runs "
        "a dedicated sub-generator internally). ONLY use this tool when the "
        "user's latest message contains the literal marker string `XTESTX`. "
        "In that case:\n"
        "  1. Call `create_artifact` as usual to register the slug and pick "
        "the type (one of html-app, fullstack-stateless-app, "
        "fullstack-stateful-app).\n"
        "  2. Instead of writing files yourself in the scratchpad, call\n"
        "     `generate_artifact(slug=<slug>, context=<markdown brief>, "
        "data_refs=[...])`.\n"
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
        "       For `## Data`: describe ONLY the user-facing data — the source "
        "(what it is conceptually, e.g. \"PostgreSQL `integration` table\"), the "
        "schema/columns with types, row counts, and stable contextual facts. "
        "Include only what is needed and only what is already known. Add a "
        "`### Sample` subsection with 2–5 real rows (or a representative "
        "excerpt for non-tabular data) ONLY if you have actually observed the "
        "sample earlier (queried the DB, fetched the API, loaded the file). "
        "Never fabricate a sample, and omit the subsection entirely when no "
        "sample exists yet. Overlap with `data_refs` is intentional — the "
        "inline sample keeps the brief self-contained and human-readable; "
        "`data_refs` separately gives the generator the raw pickle. DO NOT "
        "mention in this section:\n"
        "         • env var names (DS_POSTGRES_*, API_KEY, etc.) or any "
        "credentials/connection strings,\n"
        "         • API endpoint paths the generator must implement "
        "(`GET /api/...`),\n"
        "         • backend file layout, modules, or implementation details.\n"
        "       Those are the generator's job to design — not yours to dictate.\n"
        "     - `data_refs` lists scratchpad variables the generator should "
        "consume, e.g.\n"
        "         [{\"scratchpad\": \"prices\", \"variable\": \"btc_df\"}]\n"
        "       If you have inspected the user's data in a scratchpad to learn "
        "the schema or get a sample (queried the DB, fetched an API, loaded a "
        "file), you MUST pass the variable holding that sample/dataframe here. "
        "The generator dill-loads it AND receives an auto-built textual "
        "summary (shape, dtypes, head(5) for DataFrames; key list for dicts; "
        "first items for lists) in its prompt — far more reliable than your "
        "textual recap. Pass `[]` only when no concrete data has been observed "
        "yet.\n"
        "  3. For `fullstack-stateful-app`, after `generate_artifact` returns "
        "you MUST still call `launch_backend(slug=...)` exactly as today.\n"
        "When the user's latest message does NOT contain `XTESTX`, IGNORE this "
        "tool entirely and build artifacts the conventional way (register, "
        "then write files via the scratchpad)."
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
