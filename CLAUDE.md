# CLAUDE.md — anton

Python AI agent. Read this before touching any code in this repo.

---

## Running & Testing

```bash
uv sync                              # install deps
uv run anton                         # run the agent
uv run pytest                        # run all tests
uv run pytest tests/test_scratchpad.py  # single file
uv run pytest -k "scratchpad"        # keyword filter
uv run pytest -v                     # verbose
uv sync --extra clipboard            # optional clipboard support
```

Tests live in `tests/`. Async tests use `pytest-asyncio` (`asyncio_mode = auto`).
Tests marked `@pytest.mark.stub_only` require a stub server and are skipped unless `--live` is passed.

---

## Code Style

### Naming

- Use `_` prefix only for **class private attributes** — do not use it for module-level
  variables, local variables, function arguments, or anything that does not need
  to signal class-private intent.
- Do not use `_` as a throwaway variable name unless you are genuinely discarding
  a loop value with no alternative.
- Classes: `PascalCase`. Functions and variables: `snake_case`. Constants: `UPPER_SNAKE_CASE`.

### Docstrings

- Keep them short and direct. One sentence for simple functions; a short paragraph for
  complex classes. No need to restate what the name already says.
- Use plain prose — no Sphinx-style `:param:` / `:returns:` annotations.
- Omit docstrings entirely for trivial private methods where the name is self-explanatory.

### Comments

- No section divider comments in source files (e.g. `# ── Helpers ──`).
- No comments in tests at all — test names and assertion messages should be
  self-explanatory. If a test needs a comment to be understood, rename or restructure it.
- Inline comments only for genuinely non-obvious logic. Do not narrate what the code does.

### Types

- Use type hints throughout. `from __future__ import annotations` at the top of every file.
- Use `TYPE_CHECKING` guards for imports that are only needed for annotations.
- Prefer `X | None` over `Optional[X]`.

### General

- Python 3.11+. No `asyncio` at the `ChatSession` tool dispatch layer — it is synchronous.
- All LLM access goes through `LLMClient` in `anton/core/llm/client.py`. Never import
  `anthropic` or `openai` SDKs directly outside `anton/core/llm/`.
- Credential values must never appear in logs, prints, or LLM calls. The LLM sees
  env var names only (e.g. `DS_POSTGRES_PROD__PASSWORD`).
- Use `Cortex` as the sole memory entry point from `ChatSession`. Do not access
  `Hippocampus` or `EpisodicMemory` directly from outside `anton/core/memory/`.
- Obtain scratchpads via `ScratchpadManager.get_or_create(name)` — never instantiate
  `LocalScratchpadRuntime` directly from outside the backends package.

---

## Project Structure

```
anton/
  core/
    session.py          # ChatSession — conversation orchestrator
    llm/
      client.py         # LLMClient — provider abstraction
      provider.py       # stream event types, provider protocol
      anthropic.py      # Anthropic implementation
      openai.py         # OpenAI implementation
      prompt_builder.py # system prompt construction
    memory/
      cortex.py         # Cortex — executive memory coordinator
      hippocampus.py    # Hippocampus — semantic storage (Markdown)
      episodes.py       # EpisodicMemory — conversation log
      consolidator.py   # offline lesson extraction
      cerebellum.py     # skill-level memory
    backends/
      local.py          # LocalScratchpadRuntime
      manager.py        # ScratchpadManager
      scratchpad_boot.py# REPL subprocess script
      base.py           # Cell, ScratchpadRuntimeFactory protocols
    tools/
      registry.py       # ToolRegistry
      tool_defs.py      # SCRATCHPAD_TOOL, MEMORIZE_TOOL, RECALL_TOOL
      tool_handlers.py  # tool implementations
    datasources/
      datasource_registry.py
      data_vault.py     # DataVault — credential storage
  chat.py               # CLI chat loop, StreamDisplay wiring
  chat_ui.py            # StreamDisplay, terminal UI
  cli.py                # Typer app entry point
  minds_client.py       # MindsDB HTTP wrapper
tests/
  test_scratchpad.py
  test_chat.py
  ...                   # colocated by feature
```
