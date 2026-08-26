# Scratchpad DS_* env scoping (ENG-392)

## Problem

`LocalScratchpadRuntime.start()` (`anton/core/backends/local.py:591-733`) copies
the entire parent process `os.environ` into every scratchpad subprocess
(`env = _utf8_env(os.environ)`, line 602). On the desktop-sidecar / local
single-tenant path, `AntonHarness._build_chat_session`
(`cowork-server/cowork/harnesses/anton_harness/harness.py`) injects every one
of the user's connected datasources into that same process's `os.environ` via
`data_vault.inject_env()`. Combined, a scratchpad started for one conversation
inherits `DS_*` credentials for every connected datasource, including
connections disabled for, or unrelated to, that specific conversation — the
exposure ENG-392 describes.

## Scope

Fix the desktop-sidecar / local in-process path. No `cowork-server` changes.

The real hosted multi-tenant deployment never runs this code path at all:
`harness.py:418-436` raises `RuntimeError` out of `_build_chat_session` whenever
`tenancy_mode == "org"`, and `deployment/cowork-server/values.yaml` sets
`COWORK_TENANCY_MODE=org` + `COWORK_TURN_BACKEND=remote` for every environment
the chart deploys (dev/staging/prod). Org-mode turns instead go
`cowork-server → Redis → scratchpad-controller → k8s exec python -m
anton.cloud_turn` inside a per-conversation gVisor pod. That pod's
`build_cloud_chat_session()` (`anton/cloud_turn/session.py:518-621`) builds
`ChatSessionConfig(data_vault=None, ...)` — connectors are off on the cloud
path entirely today, so nothing injects `DS_*` there to leak. The pod's
`LocalScratchpadRuntime.start()` still does the same unfiltered `os.environ`
copy, but this task does not change that: nothing confirmed exploitable there
today, and out of scope per decision below.

Explicitly out of scope, carried as follow-ups (see bottom):

- `scrub_credentials()`'s own read of `os.environ` / `_DS_SECRET_VARS`
  (`anton/utils/datasources.py:138-175`) — same hazard class, different
  mechanism (needs values to redact, not just names to allow through).
- The cloud pod's env copy — harmless only because `data_vault=None` there
  today.
- The `pip install` subprocess env (`local.py` ~line 1276) — doesn't need
  `DS_*` credentials.
- Any `cowork-server` code change.

## Mechanism

`DataVault.env_for(engine, name) -> dict[str, str] | None`
(`anton/core/datasources/data_vault.py:308-339`) already exists and already
does exactly what's needed: builds the `DS_*` mapping for one connection
without touching `os.environ`. Nothing currently calls it for the scratchpad
boundary.

- `LocalScratchpadRuntime` gains an optional constructor param
  `scratchpad_ds_env: dict[str, str] | None = None`.
- In `start()`, immediately after `env = _utf8_env(os.environ)`: if
  `scratchpad_ds_env is not None`, strip every `DS_`-prefixed key out of
  `env`, then `env.update(scratchpad_ds_env)`. `None` (default) leaves
  today's behavior byte-for-byte unchanged.
- `ScratchpadManager` gains a `data_vault: DataVault | None = None`
  constructor param. In `get_or_create(name)`, only when creating a *new*
  pad (not a cached one) and `self._data_vault is not None`: build
  `scratchpad_ds_env` by iterating `self._data_vault.list_connections()`,
  merging `self._data_vault.env_for(conn["engine"], conn["name"]) or {}`
  for each, and pass the merged dict to the runtime factory.
- `ChatSession.__init__` (`core/session.py`) passes
  `data_vault=config.data_vault` into `ScratchpadManager(...)`.
  `ChatSessionConfig.data_vault` already exists — no new config field.
- `ScratchpadRuntimeFactory` protocol (`core/backends/base.py`) and
  `local_scratchpad_runtime_factory` gain the same `scratchpad_ds_env` param,
  threaded straight through.
- Carried over from the closed `anton#200` PR: in `start()`, when
  `self._coding_model` is falsy, `env.pop("ANTON_SCRATCHPAD_MODEL", None)`
  instead of leaving an inherited value in place; same for
  `self._coding_provider` / `ANTON_SCRATCHPAD_PROVIDER`.

No hand-maintained "system essentials" allowlist (PATH, HOME, ...) like the
closed PRs used — that's what made them silently incompatible with `start()`
as it exists today (it would have dropped `ANTON_OPENAI_BASE_URL`,
`ANTON_MINDS_API_KEY`, etc., breaking the scratchpad's own coding-model
wiring). Denylisting only the `DS_` prefix is safe because that prefix is
already the codebase's dedicated, exclusive credential namespace —
`clear_ds_env()` relies on the identical invariant.

## Why fresh-per-pad-creation, not computed once at ChatSession init

A CLI `ChatSession` can be long-lived across an interactive session; a user
can `/connect` a new datasource mid-session, then start a scratchpad that
didn't exist yet. Computing the `DS_*` overlay once at `ChatSession.__init__`
would snapshot the vault too early and miss it. Computing it inside
`ScratchpadManager.get_or_create()`, at the moment a *new* named pad is
actually created, matches the timing `os.environ.copy()` already uses today
(read at spawn time) — a freshly created pad sees current vault state, and a
pad that's already running keeps whatever it started with, exactly like
today.

## Data flow

1. CLI or desktop-sidecar cowork-server builds/loads `data_vault` (full local
   vault, or cowork-server's temp filtered vault when `disabled_connections`
   is set) exactly as today — unchanged.
2. `ChatSession.__init__` hands `data_vault` to `ScratchpadManager`.
3. First cell execution for a given scratchpad name → `get_or_create(name)`
   finds no cached pad → reads `data_vault.list_connections()` + `env_for()`
   per connection → builds `scratchpad_ds_env` → constructs
   `LocalScratchpadRuntime(..., scratchpad_ds_env=scratchpad_ds_env)` →
   `.start()`.
4. Subprocess env = full parent copy, minus any inherited `DS_*`, plus
   exactly this vault's current connections' `DS_*` pairs.

## Error handling

`env_for()` returns `None` for a connection that fails to load — the merge
loop treats that as "contributes nothing," matching how existing
`inject_env()` callers already tolerate a missing connection.

## Testing

In `anton`, mostly unit-level (inspect the dict `start()` builds, no
subprocess needed), plus one subprocess-based integration test proving
isolation end to end — mirrors the closed PR's test shape:

- `scratchpad_ds_env=None` → full env copy unchanged, including any `DS_*`
  present (legacy / no-vault case).
- `scratchpad_ds_env={}` (empty vault, or all connections disabled) → every
  inherited `DS_*` stripped, nothing added.
- `scratchpad_ds_env={"DS_X__Y": "v"}` → only that key present in the `DS_`
  namespace; any other inherited `DS_*` gone.
- `ANTON_SCRATCHPAD_MODEL` / `ANTON_SCRATCHPAD_PROVIDER` stripped when the
  coding model/provider isn't configured, even if inherited.
- `ScratchpadManager.get_or_create()` derives the right dict from a fake
  `DataVault` (connections list × `env_for()`), skips a connection whose
  `env_for()` returns `None`, and does not recompute for an already-cached
  pad.

## Follow-ups (not this task — to note on the ticket once this lands)

- `scrub_credentials()` reads secret values from `os.environ` /
  `_DS_SECRET_VARS` process globals at redaction time — same hazard class,
  different mechanism, unaddressed here.
- The cloud pod's `LocalScratchpadRuntime.start()` still does a full,
  unfiltered env copy; harmless today only because `data_vault=None` there.
  Worth a dedicated look whenever cloud connectors are wired up. Separately
  unconfirmed: whether k8s `enableServiceLinks` auto-injects
  `*_SERVICE_HOST`/`*_SERVICE_PORT` topology vars into that pod, and whether
  `scratchpad_extra_env` is truly never set in prod.
- `cowork-server`'s `harness.py` calls `inject_env()` twice back to back
  today (once via `restore_namespaced_env`, once again in a raw loop before
  `picked_files_by_project`) — looks redundant. Not touched here.
- Recommend closing `anton#200` and `cowork-server#79` for real, pointing at
  this design instead of resurrecting their diffs.
