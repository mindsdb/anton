---
title: Scratchpad runtime
description: How scratchpads execute code — the runtime ABC, the local venv backend, the remote backend, observer hooks, credential injection, and the in-pad LLM bridge.
---

# Scratchpad runtime

Scratchpads are Anton's working memory: persistent, named Python environments
the LLM drives like a notebook. The code lives in `anton/core/backends/`:

| File | Role |
|---|---|
| `base.py` | `ScratchpadRuntime` ABC + the `Cell` dataclass + `ScratchpadRuntimeFactory` protocol. Shared view/dump rendering, output truncation, and cell compaction live here so all backends benefit |
| `local.py` | `LocalScratchpadRuntime` — per-name venv + subprocess (the CLI default) |
| `remote.py` | `RemoteScratchpadRuntime` — same ABC, delegates execution over REST + SSE |
| `manager.py` | `ScratchpadManager` — lifecycle of named runtimes |
| `scratchpad_boot.py` | The boot script that runs *inside* the subprocess: REPL loop, `get_llm()`, `progress()`, `sample()`, `agentic_loop()`, `web_search()` |
| `wire.py` | Protocol markers (cell delimiter, progress marker, result delimiters) |

## The `Cell` and the runtime contract

A `Cell` is one execution unit: `code, stdout, stderr, error, description,
estimated_time, logs`. Every backend implements `start / reset / close / cancel /
cleanup / install_packages / execute_streaming`. `execute()` is a base-class
convenience that drains `execute_streaming()` (progress strings, then a final
`Cell`). `close()` preserves persistent resources (the venv); `cleanup()`
destroys them — that's the difference between session end and the `remove`
action.

## `ScratchpadManager`

Owned by the session; maps names to runtimes. `get_or_create(name)` builds a
runtime via the injected `ScratchpadRuntimeFactory` and calls `start()`.
`remove(name)` calls `cleanup()` (venv deleted); `close_all()` runs at session
end; `cancel_all_running()` handles Ctrl-C. The manager also probes the host's
installed packages (`probe_packages()`) so the tool description can advertise
notable libraries.

The manager and runtimes are **completely hook-agnostic**: the cerebellum's
pre/post-execute observers are fired by the tool dispatcher
(`handle_scratchpad` in `anton/core/tools/tool_handlers.py`), never by the
runtime. See [Cerebellum & ACC](/developer/cerebellum-and-acc) for why.

## Local runtime: isolated venv per scratchpad

`LocalScratchpadRuntime` gives each named scratchpad its own persistent venv
under `<workspace>/.anton/scratchpad-venvs/<name>/` (falling back to
`~/.anton/scratchpad-venvs/` when no workspace is bound):

- **Venv creation** prefers `uv venv --system-site-packages --seed`, falls back
  to stdlib `venv`. Creation is verified (`python -c "print('ok')"`) and retried
  up to 3 times; broken venvs are nuked and recreated.
- **Recycling**: an existing venv is reused if its Python binary works and its
  recorded Python version matches; `requirements.txt` inside the venv dir
  restores the installed-package set across sessions. Installed packages
  survive `reset` (only process state is cleared).
- **Execution**: `start()` launches `scratchpad_boot.py` as a subprocess
  (stdin/stdout pipes, own process group). Cells are written to stdin with a
  delimiter; results come back as JSON between result markers, with
  `progress()` lines streamed in between.
- **Timeouts**: `compute_timeouts(estimated_seconds)` derives a total timeout
  (roughly 2x the LLM's `estimated_execution_time_seconds`) and an inactivity
  timeout (default 30s, extended by `progress()` calls). On timeout the whole
  process tree is killed (`os.killpg` with SIGKILL on POSIX) and the cell comes
  back with a `Cell timed out` / `Cell killed` error — which is exactly what
  the ACC's `scratchpad_killed` event watches for.
- **cwd pinning**: the subprocess cwd is pinned to the workspace root when one
  is bound, so relative paths in scratchpad code operate on the project.
- **Windows**: a per-venv outbound firewall rule is added so the scratchpad can
  reach the internet.

## Remote backend

`RemoteScratchpadRuntime` implements the same ABC over HTTP: `start/reset/
cancel/install/cleanup` are POSTs and `execute_streaming` consumes an SSE
stream (`progress` events, then a `cell` event). Selection happens in
`get_runtime_factory(settings)` (`anton/chat_session.py`): when
`settings.backend == "remote"`, the remote factory is built with
`endpoint_url=settings.minds_url` and `api_key=settings.minds_api_key`;
otherwise the local factory is used.

`backend` comes from `ANTON_BACKEND` (default `local`), and a validator in
`anton/config/settings.py` rejects `remote` unless both the Minds URL and the
Minds API key are configured. The `/remote` chat command provisions a remote
scratchpad and persists `ANTON_BACKEND=remote`. A `RemoteLightsailScratchpadRuntime`
variant first resolves a Cloudflare worker URL to a direct instance endpoint
via a `/resolve` call.

## Credential injection: `DS_*` env vars

At chat-loop startup (`anton/chat.py`), every connection in the local data
vault is injected into the parent process environment as namespaced `DS_*`
variables (e.g. `DS_POSTGRES_PROD_DB__HOST`) **before any session is created** —
scratchpad subprocesses inherit `os.environ`, so credentials are available to
cells without ever appearing in code strings or the LLM transcript. Workspace
`.anton/.env` secrets are applied to the process the same way. See
[Security](/configure/security) and [Adding a data source](/developer/adding-a-datasource).

The subprocess env also carries the coding model/provider configuration
(`ANTON_SCRATCHPAD_MODEL`, `ANTON_SCRATCHPAD_PROVIDER`, plus the appropriate
SDK key/base-URL variables) so the in-pad LLM bridge works without extra setup.

## The in-pad LLM bridge: `_ScratchpadLLM`

`scratchpad_boot.py` defines `_ScratchpadLLM`, exposed to user code as
`get_llm()`. It is a **synchronous** wrapper over the coding provider, running
inside the subprocess:

- `complete(system=..., messages=[...])` — one-shot completion, with an
  automatic heartbeat that emits progress every 10s so long LLM calls don't
  trip the inactivity timeout.
- `complete_async(...)` — same, awaitable, for `asyncio.gather` fan-out.
- `generate_object(Model, system=..., messages=[...])` — forced-tool-call
  structured output. It shares `build_structured_tool` /
  `unwrap_structured_response` from `anton/core/llm/structured.py` with the
  main process's `LLMClient.generate_object`; only the provider invocation
  differs (sync subprocess vs async planning). See [LLM dispatch](/developer/llm-dispatch).

The boot script also provides `agentic_loop(...)` (a sync tool-call loop),
`web_search(query)` (routes through the configured LLM's native web search),
`progress(message)` (inactivity-timer reset), and `sample(var)` (type-aware
variable inspection).

## Dispatch flow for one `exec` call

```
handle_scratchpad(session, tc_input)             tool_handlers.py
  → prepare_scratchpad_exec(...)                 validate code/args
  → ACC observe: scratchpad_call
  → _fire_pre_execute(session, prelim_cell)      cerebellum counter
  → pad.execute(code, ...)                       pure runtime execution
  → _fire_post_execute(session, cell)            cerebellum buffers errors
  → ACC observe: scratchpad_result | scratchpad_killed
  → format_cell_result(cell)                     back to the LLM
```

Other actions: `view` (all cells + outputs), `dump` (clean notebook-style
markdown), `reset` (restart process, keep venv packages), `remove` (cleanup,
delete venv), `install` (pip/uv install into the venv).
