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
| `sandbox.py` | `CellSandbox`, the per-session Landlock policy, and the launcher call each spawn goes through when a policy is set |
| `landlock_launcher.py` | The launcher: locks itself with Landlock, then execs the target in place (standard library only) |

## The `Cell` and the runtime contract

A `Cell` is one execution unit: `code, stdout, stderr, error, description,
estimated_time, logs`. Every backend implements `start / reset / close / cancel /
cleanup / install_packages / execute_streaming`. `execute()` is a base-class
convenience that drains `execute_streaming()` (progress strings, then a final
`Cell`). `close()` preserves persistent resources (the venv); `cleanup()`
destroys them — that's the difference between session end and the `remove`
action.

When the task running a cell is cancelled, the local backend kills the cell's
process tree, records the `Cell` in `pad.cells` and re-raises the cancel
instead of yielding the `Cell`, so the turn ends. A cell past its own time
budget still yields its error `Cell`.

The CLI builds its local runtimes with `cancel_ends_turn=False`: Ctrl+C
during a cell kills that cell and the turn continues. During the first-run
demo's cell it reports the error and starts chat; Ctrl+C during the earlier
wait or dependency installation still exits the CLI.

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
- **Worker threads**: `start()`, `reset()`, `cleanup()` and `install_packages()`
  run the venv check, build and removal on a worker thread, so the host's event
  loop keeps serving other turns. Pads with the same name in one workspace
  share one venv directory, and creating or deleting it takes a lock per
  directory, so concurrent first starts build it once.
  Cancellation waits for the worker to finish before teardown clears its
  runtime paths. Reset checks and removes a broken venv under that same lock.
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
- **Pad names**: the name becomes a folder the host deletes and recreates, so
  it must be one plain folder name on every platform (`validate_pad_name`):
  not empty, no NUL, no `:` (a Windows drive or NTFS stream), not an
  absolute path, no `..`, no `/` or `\` (one pad's venv would sit inside
  another's), not only dots, and no trailing dot or space (Windows drops
  them, so `x.` would name pad `x`'s folder). The venv path must also resolve
  inside the venvs folder before any create or delete. `/share` import maps
  a pad name an older export holds to one a pad may have.
- **Host writes**: on Linux and macOS, the files the host writes into a venv
  (`_parent_venv.pth`, `requirements.txt`, `.python_version`), the venv build
  and removal, and `_agent_pads.json` refuse a symlink at or inside the venv
  or session folder in every mode (`anton/utils/nofollow.py`). A file
  written in place must be a regular file with no other hard link, checked
  before it is truncated, and `_agent_pads.json` is written to a fresh
  `.tmp` file (whatever held that name is unlinked first) and renamed into
  place. Windows has no `dir_fd` or `O_NOFOLLOW`, so there these writes act
  on the joined path and follow links; only the local user could have
  planted one.
- **Under a sandbox policy** the host also refuses a link anywhere below the
  workspace root, a linked `.anton` included, and creates and deletes the
  conversation's snapshot folder (`scratchpad-sessions/<conversation>`)
  the same way, so a reset never deletes through a linked folder.
  `_agent_pads.json` is read the same way and must be a regular file with
  no other hard link, so a linked folder or file never shows another
  person's pad names. A host
  that calls `Workspace.initialize(refuse_links=True)` gets `.anton`,
  `anton.md` and `.env` created the same way, and `read_anton_md` and
  `load_env` then raise `OSError` when either file is a symlink, a hard link
  or not a regular file. Without a policy a linked `.anton` or conversation
  folder is followed, and those reads follow links.

## Sandbox policy (Landlock)

A host that runs several people's agent code in one process tree passes a
`CellSandbox` as `ChatSessionConfig.cell_sandbox`. It names the person's
project folder, a per-person `HOME` and `TMPDIR` (each read, write and
execute, so a file moves between them and the project; a folder can't move
into `TMPDIR`, the only one that may hold sockets), an optional shared
folder (read and write, no renames in or out), and read-only folders and
files such as linked skills and attachments. The model refuses a grant that equals or contains
`/`, `/home`, `/opt`, `/opt/cowork`, `/tmp`, `/var/tmp`, `/dev`, `/etc`,
`/proc`, `/sys`, the server's own home, a root the host names in
`forbidden_roots`, or a partition it names in `other_partitions`. A writable
grant (project, `HOME`, `TMPDIR`, shared folder) inside one of
`other_partitions` is refused too; a read-only grant there, such as a skill
its owner shares, is allowed.

With a policy set, every spawn that runs agent code starts through the
launcher: the cell (the installed `scratchpad_boot.py`, run with `-P`), the
venv health check, package installs (pip only; uv is never used), and the
artifact verifier's compile and import checks, each with a timeout.
`CellSandbox.launch()` returns the launcher's argv, its fixed two-variable
environment, the policy descriptor and the write end of a refusal pipe.
The descriptor carries the rules and the target's environment as JSON, so no
secret appears in an argv another process can read, and nothing from the
target's environment loads into the launcher before the lock. `launch()`
refuses a writable grant that holds the launcher's interpreter, the launcher
module or the standard library, which run before the lock.

The launcher (`python -I -S landlock_launcher.py`) reads and closes the
descriptor, sets `no_new_privs`, requires Landlock ABI 3 (Linux 6.2) or later,
handles every filesystem right of the running ABI, and on ABI 6 or later also
scopes signals and abstract unix sockets. It adds these system rules, which
follow links (`/bin` and `/lib` are links on Debian):

- read and execute: `/usr`, `/lib`, `/lib64`, `/bin`, `/sbin`, the host
  interpreter's prefixes and site-packages, and anton's own root (in an
  editable install, the whole checkout);
- read: `/proc`, `/sys`, `/etc/ssl` and `/etc/ca-certificates`, and the
  files `/etc/passwd`, `/etc/group`, `/etc/nsswitch.conf`, `/etc/hosts`,
  `/etc/resolv.conf`, `/etc/host.conf`, `/etc/gai.conf`, `/etc/services`,
  `/etc/protocols`, `/etc/mime.types`, `/etc/localtime`, `/etc/ld.so.cache`,
  `/etc/debian_version`, `/etc/os-release` and `/etc/lsb-release`;
- read: the host's `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `PIP_CERT` and
  `CURL_CA_BUNDLE` files and its `SSL_CERT_DIR` folder. Each is held to the
  same roots as a grant and must be a regular file (a folder for
  `SSL_CERT_DIR`); anything else gets no rule and is logged once per
  process;
- devices: `/dev/null`, `/dev/zero` and `/dev/full` (read and write),
  `/dev/urandom` and `/dev/random` (read), and `/dev/shm` (create, read,
  write, truncate and remove files, without listing).

It then adds the policy's grants. A grant that is a symlink, or sits below
one, is treated like a missing path: a required one (project, `HOME`,
`TMPDIR`) refuses the launch, and an optional one (shared folder, read-only
folders and files) is skipped, which `launch()` logs. It restricts itself,
closes every descriptor above 2 except the refusal pipe, which is
close-on-exec, and execs the target. The process id, process group and exit
code are the target's, so cancel and the stdout protocol work as before.

The child environment is an allowlist (`CellSandbox.child_env`): model and
coding-provider keys, `ANTON_SCRATCHPAD_*`, `DS_*`, the pip, proxy and CA
variables, `PATH`, `PYTHONPATH` and locale, plus the per-person `HOME`,
`TMPDIR`, `USER`, `LOGNAME`, `MPLCONFIGDIR`, `XDG_*` and `PIP_CACHE_DIR`.
`COWORK_*`, `DATABASE_URI`, `MINDS_API_KEY`, `ANTON_MINDS_API_KEY` and `LD_*`
never pass. The cell, the health check and installs get only the pad's own
`DS_*`, never ones inherited from the server. A workspace `.env` key passes
where the server doesn't already set it, unless it could repoint the
interpreter, the loader, the folders or the certificate trust. A key the
server sets keeps the allowlist's verdict, so naming it in the `.env` never
passes the server's value. An entry `execve` would reject (an empty key, a
key with `=` or NUL, a value with NUL) is dropped and logged, and if one
still reaches the launcher it refuses instead of failing after the lock.

Every refusal is one line starting with `anton-sandbox: refused:`, written to
the refusal pipe and to stderr before any target code runs, with exit code
87. anton reads refusals only from the pipe, which a target the launcher
starts never holds: a target that prints the marker or exits 87 is not a
refusal. An `execve` that fails because the target's own permissions forbid
it (`access(2)`, which Landlock does not mediate) exits 126 and reads as a
broken venv, which is rebuilt. anton raises a refusal as
`SandboxUnavailable`, which the agent sees by name, and keeps the venv and
the saved cell state: a refusal never counts as a broken venv or toward the
auto-resume reset. After a refused restart the cell error does not offer a
reset, and `reset()` runs the health check before it discards the saved
namespace, so a refused reset keeps it too.

Under a policy, app backends are refused (`launch_artifact_backend` returns
`BACKENDS_OFF_MESSAGE` before it touches the pool, and `generate_artifact`
refuses a fullstack artifact before discovery asks anything, or before the
spec when discovery settles on one). The checks that start a browser or
LibreOffice (`lint_html`, `lint_url`, `check_xlsx_via_office`,
`check_opens_via_office`, `verify_frontend_live`, `verify_app_live`) take a
required `cell_sandbox` and return None under a policy before they spawn.
The in-process lints still run.

`ScratchpadManager` refuses at construction a runtime factory that does not
name a `cell_sandbox` parameter (a bare `**kwargs` is refused), and refuses
to start a runtime whose `cell_sandbox` property reports another policy, so
a policy is never dropped. A pool whose `cell_sandbox` holds anything other
than a `CellSandbox` or None raises `TypeError` where anton reads it
(`cell_sandbox_of`). With no policy (the desktop, the CLI and the cloud
pod), every spawn runs unlocked with the server's environment, and
`tests/test_sandbox_off_capture.py` pins each call's argv, environment, cwd
and keyword arguments.

The real-Landlock tests carry the `landlock` mark. `tests/conftest.py` skips
them where the kernel has no Landlock ABI 3, fails them instead on Linux
under GitHub Actions or with `LANDLOCK_REQUIRED=1`, and prints the ABI in
pytest's report header.

## Remote backend

`RemoteScratchpadRuntime` implements the same ABC over HTTP: `start/reset/
cancel/install/cleanup` are POSTs and `execute_streaming` consumes an SSE
stream (`progress` events, then a `cell` event). Selection happens in
`get_runtime_factory(settings)` (`anton/chat_session.py`): when
`settings.backend == "remote"`, the remote factory is built with
`endpoint_url=settings.minds_url` and `api_key=settings.minds_api_key`;
otherwise the local factory is used.

Each execution sends the turn's `trace_context` alongside the unchanged cell
source. The scratchpad service restores it while executing the request, so
the local runtime can forward it to the cell's model calls. A request outside
a turn sends `null`. This requires the scratchpad service's trace-aware
receiver and an Anton runtime that forwards cell traces; older receivers
ignore the extra field and continue executing without trace attribution.

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

Each model call from these helpers builds its own provider client and closes it before the call returns. A sync call runs on an event loop that `asyncio.run` closes when the call ends, so a client kept for the next call would hand it a connection tied to a closed loop.

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
