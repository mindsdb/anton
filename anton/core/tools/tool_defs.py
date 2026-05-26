from anton.core.tools.tool_handlers import (
    handle_create_artifact,
    handle_list_artifacts,
    handle_memorize,
    handle_open_artifact,
    handle_read_image,
    handle_recall,
    handle_scratchpad,
    handle_set_artifact_primary,
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
        "- fullstack-stateless-app: HTML + JS + CSS that runs without a server\n"
        "- fullstack-stateful-app: needs a backend process to serve\n\n"
        "Pass `primary` (optional) when you already know the entry-point "
        "filename you'll write — e.g. `\"dashboard.html\"` for an html-app, "
        "`\"index.html\"` for a fullstack app, `\"report.pdf\"` for a "
        "document. The renderer uses it to decide what to open by default. "
        "Skip when you don't know yet — the renderer falls back to a "
        "heuristic, and you can set it later via `set_artifact_primary`.\n\n"
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


SET_ARTIFACT_PRIMARY_TOOL = ToolDef(
    name="set_artifact_primary",
    description=(
        "Update the primary-file pointer on an existing artifact. Call this "
        "when you created the artifact without a `primary` and now know what "
        "it should be, or when the entry-point file's name changed. Pass an "
        "empty string or omit `primary` to clear (the renderer reverts to "
        "its heuristic — `index.html` → newest `.html` → newest non-"
        "housekeeping file)."
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
                "description": "Relative path of the new entry-point file. Empty string to clear.",
            },
        },
        "required": ["slug"],
    },
    handler=handle_set_artifact_primary,
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
