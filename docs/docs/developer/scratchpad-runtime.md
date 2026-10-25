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
Every URL in a `PIP_*` value (an index, a links page, a proxy) loses its
userinfo and query string and keeps its scheme, host, port and path, so a
mirror's password or token never reaches agent code. The health check and
installs get the same URLs as the cell: they run the venv's own Python,
which a cell can rewrite. So an index or proxy that needs credentials can't
be used under a policy. A `PIP_*` value with a URL that can't be parsed is
dropped. Each variable that loses credentials is logged once per process.

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
LibreOffice in this process (`lint_html`, `lint_url`, `check_xlsx_via_office`,
`check_opens_via_office`, `verify_app_live`) take a required `cell_sandbox`
and return None under a policy before they spawn. The in-process lints still
run.

When the host sets `ANTON_LIVE_APP_SLOTS`, a fullstack app runs in an app
slot instead (`anton/core/artifacts/live_app_slots.py`). A slot is a separate
container that holds one app at a time, restarts empty before it takes
another, and serves the app at its own address (`ANTON_LIVE_APP_ADDRESS`,
default `http://app{n}.view.localhost:8080/`), which only the person who
launched it opens. `launch_artifact_backend` still starts nothing in this
process; `launch_backend`, `generate_artifact`'s run step, the check after
each cell and a change to an app's declared connections send the launch to
a slot, and its result carries the address, never a port or a pid. A
launch sends `backend.py`, `requirements.txt`, `state_manifest.json` and the
files under `static/`, read with no-follow opens and only from a folder the
person's own cells may read (`CellSandbox.allows_read`), and the `DS_*` of
each declared connection that the operator approved
(`ANTON_LIVE_APP_CONNECTIONS_FILE`) and the person saved pointing at the
approved host. An API connection's URLs are rewritten to the stack's relay,
which enforces what the approval allows. Every other declared connection is
named in the result, never with a value. The first launch, and any relaunch
after `backend.py`, `requirements.txt` or the connections change, waits for
the write gate's card (`scratchpad.ask_approval_card`), which names each
connection with what it may do; a change to `static/` alone relaunches
without one, and a cell that leaves every launch file's bytes as they were
relaunches nothing. A launch takes the slot that still runs the app, else
the lowest free one, so a relaunch after a stop may get a new address, else
the launcher's own oldest app, which the card names; it never replaces
another person's app, and with every slot held by others it names who holds
what. The host hands the session a `LiveAppHost`
(`ChatSessionConfig.live_app_host`): who launches, the slots' token, and the
record of each launch, including a launch that stopped an app and then
failed. With live apps on, the generator installs nothing for the backend,
since a slot installs nothing, and its verifier fails a fullstack page that
loads a script from anywhere but `https://cdn.jsdelivr.net`, the one script
origin the app's content policy allows.

Pages are still loaded in a browser under a policy when the host sets
`ANTON_RENDER_CHECK_URL`: the render-check service
(`anton/core/artifacts/render_check.py`) runs Chromium in its own container
and loads the page the way the app preview does, and only the page's text
leaves this process. The server reads the page only where the person's own
cells may read (`CellSandbox.allows_read`), and through
`nofollow.read_bytes_nofollow`, so a link agent code swaps in after that
check is refused, never followed. The variable is not on the cell
environment's allowlist. `generate_artifact`'s browser step
(`verify_frontend_live`) and the check after every scratchpad cell that
changes an artifact (`tool_handlers._check_pages`) both use it. A cell
changes an artifact when it writes, replaces, renames, copies in or deletes
any of its content files: each file is compared by its mtime and size
(`tool_handlers.ArtifactFiles`). The cell check loads the pages the cell
edited. When an artifact other than a fullstack app changed but none of
its pages did (the cell rewrote only a data file the page loads, say), it
loads the artifact's main page: its primary, else its `index.html`, else
its first page by name. It sends no page of a fullstack app:
that page calls its backend, which the service does not run, so the cell's
result names the page as not checked. Without the variable,
`verify_frontend_live` reports the page as unverified and the cell's result
says the page was not checked in a browser. Either way, a sandboxed turn
with a page that did not pass its last check ends with a line anton writes
itself, naming the artifact (`ChatSession._unverified_page_lines`). Each
page keeps its own latest result, so a later pass of one page never clears
another page's failure.

A check the service could not vouch for is unverified, and its line says
why (a resource it could not load, the deadline). Under that line the
agent sees what the check still found, such as an exception, or a blank
view, an overflow or a loading message left over a drawn view, measured
before the deadline, which the line marks as possibly still drawing
(`render_check.unverified_findings`). The cell check
lists those findings once per page per turn: once an unverified check in
the turn listed findings for a page, in a cell's result or in
`generate_artifact`'s `checks_skipped`, a later unverified check of that
page gets only the line that says it was not checked
(`ChatSession._unverified_findings_listed`). A check that listed nothing,
because the service was busy or the cell's time ran out, leaves the next
check free to list what it finds. So the agent fixes them once and then
reports the page as unverified, instead of editing a page that may work
on every cell.

Each check also compares the page with how it was when the turn began.
Before the turn's first cell or `generate_artifact` run, anton keeps a copy
of every artifact's main page but a fullstack app's
(`page_copies.TurnPages.capture`): newest first, at most 64 MiB a turn,
read the way the check reads a page. A check of a page with a copy sends it
as `previous_html`, unless less than a whole check's time
(`render_check.TIMEOUT_SECONDS`) is left of the cell's budget, and the
service's `comparison` names the views the turn's edits broke, left without
a graphic, removed, or left fewer than half the marks they drew (svg shapes,
map markers, table rows; a canvas's aren't counted). A view that broke fails the page, so its repair cells
apply. The others are not failures, and the agent hears about each set once
a turn. When a page the turn's edits made worse fails with no repair cells
left, anton puts the copy back. A `generate_artifact` run on an existing
artifact that fails, stops over budget, crashes or is abandoned by the
turn puts the main page back as it was when the run started: as the turn
found it, when the run was the first to change it. A build that fails only
its browser check comes back to repair with the same comparison
(`changes_from_previous`). A run that never wrote (cancelled, needs
confirmation) puts nothing back. On a sandboxed
deployment the turn ends with a line for each page put back as the turn
found it and for each view broken, blanked, removed or shrunk
(`TurnPages.end_lines`). The comparison never costs a page its own verdict:
a copy too large to send goes without it, and when a service that predates
the comparison answers such a request 400, anton asks once more with the
page alone.

The service also presses each tab and toggle it finds, and a pass vouches
only for the views it measured. The cell check's line for a passed page
names them: the page as it opens and the view behind each tab or toggle the
service pressed, at the sizes where each showed
(`render_check.view_coverage`). When the service found tabs or toggles it
did not press (its cap on presses, its time ran out, the page navigated
away), a second line names them and tells the agent to call the views
behind them unchecked. `generate_artifact` puts the same beside a pass as
`browser_checked` and `browser_unchecked`, and its instruction lets the
agent call only the views `browser_checked` names verified or working. An
older service that names no views, and the local browser, which presses
nothing, vouch for the page as it opens only.

The service also sends a screenshot of the page, which reaches the agent as
an image beside the check's text. History keeps only the newest one
(`render_check.drop_older_screenshots`). When the model refuses a request
that may have failed for carrying images at all
(`provider.is_image_refusal`: a content rejection, or a 400 or 422 that
names an image, which is what a model with no image input gets), the turn
replaces the images anton added itself with a note, keeps the text beside
them, and sends the request once more. Render screenshots
(`ChatSession._drop_render_screenshots`) and the images other tools
returned, such as what `read_image` loaded
(`ChatSession._drop_refused_tool_images`), are each dropped at most once
per turn, so one turn resends at most twice. An image the person attached
is left to the content repair. A refusal of one image, such as an image
over the size limit, does not count. Only when that request goes through
without the dropped images does the note say the model refused them
(`ChatSession._confirm_render_screenshots_refused`,
`ChatSession._confirm_tool_images_refused`). When one refused request
carries both kinds, the tool images go first; the screenshots go only if
the request is refused again, and the request that then goes through
confirms both. Otherwise each kind follows its own refusal. After refused
screenshots, later checks in that session ask the service for no
screenshot. After a refused tool image, `read_image` answers in text
instead of loading another, and checks keep asking for screenshots: one
corrupt file draws a refusal from a model that takes images too. A model
with no image input then refuses the next screenshot as well, and that
screenshot gets its own drop and resend. If the request is refused again
with nothing left to drop, the dropped images were not the cause, and
nothing changes for later checks or calls. A session seeded with a
confirmed note, as cowork-server builds one for every turn, starts the same
way.

When `generate_artifact`'s browser step fails a page, the generator's next
attempt gets the findings and, beside them, the screenshot of the failed
page (`orchestrator._kickoff_with_screenshot`). A model that refuses that
request gets it once more without the screenshot (`engine._run_loop`). When
the refused request was the attempt's first round, which runs on the
planning model, the main agent's own, the session records the refusal and
later checks send no screenshot either. A later round runs on the coding
model, so its refusal drops only that attempt's screenshot.

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
