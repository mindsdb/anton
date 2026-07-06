from anton.core.tools.tool_handlers import (
    handle_create_artifact,
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
        "Pick `type` from the closed enum:\n"
        "- html-app: a single self-contained HTML page (charts, dashboards, demos)\n"
        "- document: a doc, report, or markdown file the user reads\n"
        "- dataset: data files (CSV, JSON, parquet) the user downloads or feeds elsewhere\n"
        "- image: a generated image (PNG, SVG, etc.)\n"
        "- mixed: multi-modal output that doesn't fit the above\n"
        "- fullstack-stateless-app: fullstack web app (backend + frontend) that keeps "
        "no local state between requests; all persistence goes to external data sources. "
        "DEFAULT for fullstack apps\n"
        "- fullstack-stateful-app: fullstack web app (backend + frontend) that keeps "
        "local state between requests (e.g. an on-disk SQLite DB). Use ONLY when that "
        "state truly cannot live in an external data source; prefer stateless when in doubt\n\n"
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
        "dismissed), 'no_matches', 'invalid'. Never re-ask in plain text after a "
        "resolved selection."
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
