# Scratchpad DS_* Env Scoping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop a scratchpad subprocess from inheriting `DS_*` datasource credentials it has no business seeing, by building its env from the current data vault instead of trusting whatever is sitting in the parent process's `os.environ`.

**Architecture:** `LocalScratchpadRuntime.start()` gains an explicit `scratchpad_ds_env: dict[str, str] | None` — when set, it strips every inherited `DS_`-prefixed key from the copied parent env and overlays exactly this dict. `ScratchpadManager` computes that dict fresh, at the moment it creates a new pad, from a `data_vault` it's now handed. `ChatSession` passes its already-existing `config.data_vault` through. `None` anywhere in this chain (no vault) means today's behavior, unchanged.

**Tech Stack:** Python 3.11+, `uv`, `pytest` (`pytest-asyncio`, tests are `async def` with no explicit marker per this repo's existing convention).

**Spec:** `docs/superpowers/specs/2026-08-26-scratchpad-ds-env-scoping-design.md`

## Global Constraints

- Single repo: `anton` only. No `cowork-server` changes.
- `scratchpad_ds_env=None` / `data_vault=None` anywhere in the chain must leave behavior byte-for-byte identical to today — this is the CLI/legacy path and must not regress.
- Only the `DS_` prefix is ever stripped/overlaid. Every other env var (system vars, `ANTON_*`/`OPENAI_*`/`ANTHROPIC_*` coding-model vars) passes through exactly as it does today.
- No new `ChatSessionConfig` field — `data_vault` already exists there.
- New optional kwargs on `ScratchpadRuntimeFactory` follow the existing `_probe_factory_kwarg` compatibility pattern (see `manager.py:41-43,46-55`) so factories that don't know about them (e.g. `remote.py`'s) keep working unmodified.
- Match existing code style in each file: `from __future__ import annotations`, type all signatures, comments explain *why* not *what*, ≤2 lines per comment.
- Test convention in this repo for scratchpad backends is real subprocess execution (`await pad.start()` / `await pad.execute(...)`), not mocks — follow `tests/test_scratchpad.py::TestScratchpadEnvironment` exactly.
- **Commit granularity:** one commit per logically-independent change, not one per task — small enough to revert individually without pulling in unrelated changes.
- **Commit messages:** one short line stating the change. No ticket numbers, issue IDs, or ticket names.
- **Comments/docstrings you write:** never reference a ticket number or issue ID. (Surrounding existing code in these files already does this in places — that's pre-existing convention; don't extend it in anything new you add.)

---

### Task 1: `LocalScratchpadRuntime` — accept and apply an explicit DS_* overlay

**Files:**
- Modify: `anton/core/backends/local.py:196-230` (`__init__`), `anton/core/backends/local.py:591-607` (`start`)
- Test: `tests/test_scratchpad.py` (extend `class TestScratchpadEnvironment`, starts at line 497)

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `LocalScratchpadRuntime.__init__(..., scratchpad_ds_env: dict[str, str] | None = None)` and the instance attribute `self._scratchpad_ds_env`, which Task 2 reads indirectly by passing this kwarg through the factory.

This task lands as two small commits: the DS_* overlay itself, then the separate (smaller) model/provider leak fix.

#### Part A — the DS_* overlay

- [ ] **Step 1: Write the failing tests**

Open `tests/test_scratchpad.py`. Add these three tests at the end of `class TestScratchpadEnvironment` (after `test_api_key_bridged`, before the blank line that precedes `class TestScratchpadVenv:`):

```python
    async def test_scratchpad_ds_env_none_preserves_inherited_ds_vars(self, monkeypatch):
        """scratchpad_ds_env=None (default) keeps today's behaviour: inherited
        DS_* vars pass through untouched."""
        monkeypatch.setenv("DS_POSTGRES_PROD__PASSWORD", "inherited-secret")
        pad = make_scratchpad(name="ds-env-none")
        await pad.start()
        try:
            cell = await pad.execute(
                "import os; print(os.environ.get('DS_POSTGRES_PROD__PASSWORD', 'NOT_FOUND'))"
            )
            assert cell.stdout.strip() == "inherited-secret"
        finally:
            await pad.close()

    async def test_scratchpad_ds_env_empty_strips_all_inherited_ds_vars(self, monkeypatch):
        """An explicit empty dict (empty vault / all connections disabled)
        strips every inherited DS_* var — none are added back."""
        monkeypatch.setenv("DS_POSTGRES_PROD__PASSWORD", "should-not-leak")
        pad = make_scratchpad(name="ds-env-empty", scratchpad_ds_env={})
        await pad.start()
        try:
            cell = await pad.execute(
                "import os; print(os.environ.get('DS_POSTGRES_PROD__PASSWORD', 'NOT_FOUND'))"
            )
            assert cell.stdout.strip() == "NOT_FOUND"
        finally:
            await pad.close()

    async def test_scratchpad_ds_env_only_exposes_explicit_keys(self, monkeypatch):
        """Only the DS_* pairs in scratchpad_ds_env are visible — a DS_* var
        inherited for a different connection is stripped even though a
        DIFFERENT DS_* var was explicitly allowed."""
        monkeypatch.setenv("DS_SLACK_MAIN__BOT_TOKEN", "wrong-conversation-token")
        pad = make_scratchpad(
            name="ds-env-scoped",
            scratchpad_ds_env={"DS_POSTGRES_PROD__PASSWORD": "right-conversation-secret"},
        )
        await pad.start()
        try:
            cell = await pad.execute(
                "import os\n"
                "print(os.environ.get('DS_POSTGRES_PROD__PASSWORD', 'NOT_FOUND'))\n"
                "print(os.environ.get('DS_SLACK_MAIN__BOT_TOKEN', 'NOT_FOUND'))"
            )
            lines = cell.stdout.strip().splitlines()
            assert lines[0] == "right-conversation-secret"
            assert lines[1] == "NOT_FOUND"
        finally:
            await pad.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_scratchpad.py -k "test_scratchpad_ds_env" -v`

Expected: `test_scratchpad_ds_env_none_preserves_inherited_ds_vars` PASSES already (describes today's behavior). The other two FAIL with `TypeError: __init__() got an unexpected keyword argument 'scratchpad_ds_env'`.

- [ ] **Step 3: Add `scratchpad_ds_env` to `__init__`**

In `anton/core/backends/local.py`, the `__init__` signature at line 196 currently reads (through line 230):

```python
    def __init__(
        self,
        name: str,
        *,
        coding_provider: str,
        coding_model: str,
        coding_api_key: str,
        coding_base_url: str,
        cells: list[Cell] | None = None,
        workspace_path: Path | None = None,
        session_id: str | None = None,
        _venvs_base: Path | None = None,
    ) -> None:
```

Add the new parameter after `session_id`:

```python
    def __init__(
        self,
        name: str,
        *,
        coding_provider: str,
        coding_model: str,
        coding_api_key: str,
        coding_base_url: str,
        cells: list[Cell] | None = None,
        workspace_path: Path | None = None,
        session_id: str | None = None,
        scratchpad_ds_env: dict[str, str] | None = None,
        _venvs_base: Path | None = None,
    ) -> None:
```

Then at line 229 (`self._session_id: str | None = session_id`), add the new attribute right after it:

```python
        self._session_id: str | None = session_id
        # Explicit DS_* values for this pad's subprocess, computed by
        # ScratchpadManager from the current data vault. None (default)
        # leaves DS_* handling exactly as before — the parent env's own
        # copy, whatever it holds.
        self._scratchpad_ds_env: dict[str, str] | None = scratchpad_ds_env
        self._proc: asyncio.subprocess.Process | None = None
```

- [ ] **Step 4: Apply the overlay in `start()`**

In `anton/core/backends/local.py`, lines 602-606 currently read:

```python
        env = _utf8_env(os.environ)
        if self._coding_model:
            env["ANTON_SCRATCHPAD_MODEL"] = self._coding_model
        if self._coding_provider:
            env["ANTON_SCRATCHPAD_PROVIDER"] = self._coding_provider
```

Replace with:

```python
        env = _utf8_env(os.environ)
        if self._scratchpad_ds_env is not None:
            # Never trust inherited DS_* values — they may belong to a
            # different conversation/connection set than this vault's
            # current state. Strip whatever the parent process is
            # holding, then overlay exactly what this pad should see.
            for key in [k for k in env if k.startswith("DS_")]:
                del env[key]
            env.update(self._scratchpad_ds_env)
        if self._coding_model:
            env["ANTON_SCRATCHPAD_MODEL"] = self._coding_model
        if self._coding_provider:
            env["ANTON_SCRATCHPAD_PROVIDER"] = self._coding_provider
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_scratchpad.py -k "test_scratchpad_ds_env" -v`

Expected: all 3 PASS.

- [ ] **Step 6: Run the full scratchpad test file**

Run: `uv run pytest tests/test_scratchpad.py -v`

Expected: PASS (no regressions in the ~180 pre-existing tests in this file).

- [ ] **Step 7: Commit**

```bash
git add anton/core/backends/local.py tests/test_scratchpad.py
git commit -m "Scope scratchpad subprocess env to an explicit DS_* overlay"
```

#### Part B — stop leaking an unconfigured model/provider

- [ ] **Step 8: Write the failing tests**

Add these two tests right after the three from Part A (still inside `class TestScratchpadEnvironment`):

```python
    async def test_scratchpad_model_not_leaked_when_unconfigured(self, monkeypatch):
        """An inherited ANTON_SCRATCHPAD_MODEL must not reach a pad that has
        no coding model configured."""
        monkeypatch.setenv("ANTON_SCRATCHPAD_MODEL", "inherited-model")
        pad = make_scratchpad(name="ds-env-no-model")  # coding_model="" by default
        await pad.start()
        try:
            cell = await pad.execute(
                "import os; print(os.environ.get('ANTON_SCRATCHPAD_MODEL', 'NOT_FOUND'))"
            )
            assert cell.stdout.strip() == "NOT_FOUND"
        finally:
            await pad.close()

    async def test_scratchpad_provider_not_leaked_when_unconfigured(self, monkeypatch):
        """An inherited ANTON_SCRATCHPAD_PROVIDER must not reach a pad that
        has no coding provider configured."""
        monkeypatch.setenv("ANTON_SCRATCHPAD_PROVIDER", "inherited-provider")
        pad = make_scratchpad(name="ds-env-no-provider", coding_provider="")
        await pad.start()
        try:
            cell = await pad.execute(
                "import os; print(os.environ.get('ANTON_SCRATCHPAD_PROVIDER', 'NOT_FOUND'))"
            )
            assert cell.stdout.strip() == "NOT_FOUND"
        finally:
            await pad.close()
```

- [ ] **Step 9: Run the tests to verify they fail**

Run: `uv run pytest tests/test_scratchpad.py -k "not_leaked_when_unconfigured" -v`

Expected: both FAIL with an assertion mismatch (`"inherited-model" != "NOT_FOUND"` / `"inherited-provider" != "NOT_FOUND"`) — today nothing pops an inherited `ANTON_SCRATCHPAD_MODEL`/`PROVIDER` when the pad has no coding model/provider configured, so the monkeypatched value leaks straight through.

- [ ] **Step 10: Pop the inherited var when unconfigured**

In `anton/core/backends/local.py`, the block Step 4 just left reads:

```python
        if self._coding_model:
            env["ANTON_SCRATCHPAD_MODEL"] = self._coding_model
        if self._coding_provider:
            env["ANTON_SCRATCHPAD_PROVIDER"] = self._coding_provider
```

Replace with:

```python
        if self._coding_model:
            env["ANTON_SCRATCHPAD_MODEL"] = self._coding_model
        else:
            env.pop("ANTON_SCRATCHPAD_MODEL", None)
        if self._coding_provider:
            env["ANTON_SCRATCHPAD_PROVIDER"] = self._coding_provider
        else:
            env.pop("ANTON_SCRATCHPAD_PROVIDER", None)
```

- [ ] **Step 11: Run the tests to verify they pass**

Run: `uv run pytest tests/test_scratchpad.py -k "not_leaked_when_unconfigured" -v`

Expected: both PASS.

- [ ] **Step 12: Run the full scratchpad test file again**

Run: `uv run pytest tests/test_scratchpad.py -v`

Expected: PASS.

- [ ] **Step 13: Commit**

```bash
git add anton/core/backends/local.py tests/test_scratchpad.py
git commit -m "Stop an unconfigured scratchpad from inheriting a model/provider"
```

---

### Task 2: `ScratchpadManager` derives the DS_* overlay from a data vault

**Files:**
- Modify: `anton/core/datasources/data_vault.py:170-176` (add `env_for` to the `DataVault` protocol)
- Modify: `anton/core/backends/base.py:251-266` (`ScratchpadRuntimeFactory.__call__`)
- Modify: `anton/core/backends/local.py:1334-1354` (`local_scratchpad_runtime_factory`)
- Modify: `anton/core/backends/manager.py` (`ScratchpadManager.__init__` and `get_or_create`)
- Test: `tests/test_scratchpad.py` (extend `class TestScratchpadManager`, starts at line 267)

**Interfaces:**
- Consumes: `LocalScratchpadRuntime(..., scratchpad_ds_env=...)` from Task 1; `DataVault.list_connections() -> list[dict[str, str]]` and `DataVault.env_for(engine: str, name: str, *, flat: bool = False) -> dict[str, str] | None` (already implemented by `LocalDataVault`, `anton/core/datasources/data_vault.py:287-306,308-339`).
- Produces: `ScratchpadManager.__init__(..., data_vault: DataVault | None = None)`. Task 3 passes `config.data_vault` here.

This task lands as three small commits: the protocol declaration, the factory plumbing, then the manager behavior + its tests.

#### Part A — declare `env_for` on the `DataVault` protocol

`ScratchpadManager` (Part C, below) is about to call `data_vault.env_for(...)` on a value typed as `DataVault` (the protocol), not `LocalDataVault` directly. `env_for` is already implemented on `LocalDataVault` (`anton/core/datasources/data_vault.py:308-339`) but isn't declared on the protocol itself, which would make it invisible to type checkers.

- [ ] **Step 1: Add `env_for` to the protocol**

In `anton/core/datasources/data_vault.py`, lines 170-176 currently read:

```python
    def list_connections(self) -> list[dict[str, str]]:
        """Return [{engine, name, created_at}] for all stored connections."""
        ...

    def inject_env(self, engine: str, name: str, *, flat: bool = False) -> list[str] | None:
        """Load credentials and set DS_* environment variables."""
        ...
```

Insert a new method between them, so the protocol reads:

```python
    def list_connections(self) -> list[dict[str, str]]:
        """Return [{engine, name, created_at}] for all stored connections."""
        ...

    def env_for(self, engine: str, name: str, *, flat: bool = False) -> dict[str, str] | None:
        """Build the DS_* env mapping for a connection WITHOUT mutating os.environ.

        Returns the {var: value} mapping, or None if the connection isn't
        found. Use this when the env should reach only a specific
        subprocess; use `inject_env` when the variables must be visible in
        the current process.
        """
        ...

    def inject_env(self, engine: str, name: str, *, flat: bool = False) -> list[str] | None:
        """Load credentials and set DS_* environment variables."""
        ...
```

- [ ] **Step 2: Sanity check — nothing broke**

Run: `uv run pytest tests/test_data_vault.py -v`

Expected: PASS. This step adds no new behavior (`LocalDataVault.env_for` already existed and is untouched); it only makes an existing method part of the declared protocol, so there's no new test to write here — just confirm the file still imports and the existing suite is green.

- [ ] **Step 3: Commit**

```bash
git add anton/core/datasources/data_vault.py
git commit -m "Declare env_for on the DataVault protocol"
```

#### Part B — thread `scratchpad_ds_env` through the runtime factory

- [ ] **Step 4: Update the factory protocol**

In `anton/core/backends/base.py`, the `ScratchpadRuntimeFactory.__call__` signature at lines 251-266 currently ends:

```python
        workspace_path: Path | None,
        # Conversation/session identifier, when the host supplies one. Scopes the
        # namespace snapshot so two conversations in one workspace can reuse the same
        # pad name without reading each other's state (ENG-1124). Optional so hosts
        # and test doubles that predate it keep working.
        session_id: str | None = None,
    ) -> ScratchpadRuntime: ...
```

Add the new parameter after `session_id`:

```python
        workspace_path: Path | None,
        # Conversation/session identifier, when the host supplies one. Scopes the
        # namespace snapshot so two conversations in one workspace can reuse the same
        # pad name without reading each other's state (ENG-1124). Optional so hosts
        # and test doubles that predate it keep working.
        session_id: str | None = None,
        # Explicit DS_* env values for this pad, when the host has a data
        # vault to scope them from. Optional so hosts and test doubles
        # that predate it keep working.
        scratchpad_ds_env: dict[str, str] | None = None,
    ) -> ScratchpadRuntime: ...
```

(The `(ENG-1124)` reference above is pre-existing text you are not touching — leave it as is; just add the new parameter below it.)

- [ ] **Step 5: Update `local_scratchpad_runtime_factory`**

In `anton/core/backends/local.py`, `local_scratchpad_runtime_factory` at lines 1334-1354 currently reads:

```python
def local_scratchpad_runtime_factory(
    *,
    name: str,
    coding_provider: str,
    coding_model: str,
    coding_api_key: str,
    coding_base_url: str,
    cells: list[Cell] | None,
    workspace_path: Path | None,
    session_id: str | None = None,
) -> ScratchpadRuntime:
    return LocalScratchpadRuntime(
        name=name,
        coding_provider=coding_provider,
        coding_model=coding_model,
        coding_api_key=coding_api_key,
        coding_base_url=coding_base_url,
        cells=cells,
        workspace_path=workspace_path,
        session_id=session_id,
    )
```

Replace with:

```python
def local_scratchpad_runtime_factory(
    *,
    name: str,
    coding_provider: str,
    coding_model: str,
    coding_api_key: str,
    coding_base_url: str,
    cells: list[Cell] | None,
    workspace_path: Path | None,
    session_id: str | None = None,
    scratchpad_ds_env: dict[str, str] | None = None,
) -> ScratchpadRuntime:
    return LocalScratchpadRuntime(
        name=name,
        coding_provider=coding_provider,
        coding_model=coding_model,
        coding_api_key=coding_api_key,
        coding_base_url=coding_base_url,
        cells=cells,
        workspace_path=workspace_path,
        session_id=session_id,
        scratchpad_ds_env=scratchpad_ds_env,
    )
```

- [ ] **Step 6: Sanity check — nothing broke**

Run: `uv run pytest tests/test_scratchpad.py -v`

Expected: PASS. Nothing calls the factory with `scratchpad_ds_env` yet (that's Part C), so this is pure plumbing — confirm the signature change alone doesn't break any existing caller.

- [ ] **Step 7: Commit**

```bash
git add anton/core/backends/base.py anton/core/backends/local.py
git commit -m "Thread scratchpad_ds_env through the runtime factory"
```

#### Part C — `ScratchpadManager` derives and passes the overlay

- [ ] **Step 8: Write the failing tests**

Open `tests/test_scratchpad.py`. Add a fake vault class right after the `make_manager` helper (after line 35, before `class TestScratchpadBasicExecution:`):

```python
class _FakeVault:
    """Duck-typed DataVault stand-in — ScratchpadManager only calls
    list_connections() and env_for()."""

    def __init__(self, connections: dict[tuple[str, str], dict[str, str] | None]) -> None:
        self._connections = connections

    def list_connections(self) -> list[dict[str, str]]:
        return [{"engine": e, "name": n} for e, n in self._connections]

    def env_for(self, engine: str, name: str, *, flat: bool = False) -> dict[str, str] | None:
        return self._connections[(engine, name)]
```

Then add these three tests at the end of `class TestScratchpadManager` (after `test_close_all_does_not_restart_processes`, before the blank line that precedes `class TestScratchpadRenderNotebook:`):

```python
    async def test_get_or_create_derives_ds_env_from_vault(self):
        """A new pad gets exactly this vault's current DS_* values."""
        vault = _FakeVault({
            ("postgres", "prod"): {"DS_POSTGRES_PROD__HOST": "db.example.com"},
            ("slack", "main"): {"DS_SLACK_MAIN__BOT_TOKEN": "xoxb-123"},
        })
        mgr = make_manager(data_vault=vault)
        try:
            pad = await mgr.get_or_create("alpha")
            assert pad._scratchpad_ds_env == {
                "DS_POSTGRES_PROD__HOST": "db.example.com",
                "DS_SLACK_MAIN__BOT_TOKEN": "xoxb-123",
            }
        finally:
            await mgr.close_all()

    async def test_get_or_create_skips_connection_whose_env_for_fails(self):
        """A connection whose env_for() returns None contributes nothing,
        not a crash."""
        vault = _FakeVault({
            ("postgres", "gone"): None,
            ("postgres", "prod"): {"DS_POSTGRES_PROD__HOST": "db.example.com"},
        })
        mgr = make_manager(data_vault=vault)
        try:
            pad = await mgr.get_or_create("alpha")
            assert pad._scratchpad_ds_env == {"DS_POSTGRES_PROD__HOST": "db.example.com"}
        finally:
            await mgr.close_all()

    async def test_get_or_create_without_vault_passes_none(self):
        """No data_vault (default) -> scratchpad_ds_env stays None, so
        LocalScratchpadRuntime keeps its legacy full-copy behaviour."""
        mgr = make_manager()  # no data_vault
        try:
            pad = await mgr.get_or_create("alpha")
            assert pad._scratchpad_ds_env is None
        finally:
            await mgr.close_all()
```

- [ ] **Step 9: Run the tests to verify they fail**

Run: `uv run pytest tests/test_scratchpad.py -k "derives_ds_env or skips_connection_whose_env_for_fails or without_vault_passes_none" -v`

Expected: `test_get_or_create_without_vault_passes_none` PASSES already (nothing changed yet, `scratchpad_ds_env` is already `None` by default from Task 1). The other two FAIL with `TypeError: __init__() got an unexpected keyword argument 'data_vault'`.

- [ ] **Step 10: Add `data_vault` to `ScratchpadManager` and derive the overlay in `get_or_create`**

In `anton/core/backends/manager.py`, add the import at the top (after the existing `from anton.core.backends.base import ...` line):

```python
from anton.core.datasources.data_vault import DataVault
```

The `__init__` signature currently reads:

```python
    def __init__(
        self,
        runtime_factory: ScratchpadRuntimeFactory,
        coding_provider: str,
        coding_model: str,
        coding_api_key: str,
        coding_base_url: str,
        cells: list[Cell] | None = None,
        workspace_path: Path | None = None,
        session_id: str | None = None,
    ) -> None:
```

Add `data_vault` after `session_id`:

```python
    def __init__(
        self,
        runtime_factory: ScratchpadRuntimeFactory,
        coding_provider: str,
        coding_model: str,
        coding_api_key: str,
        coding_base_url: str,
        cells: list[Cell] | None = None,
        workspace_path: Path | None = None,
        session_id: str | None = None,
        data_vault: DataVault | None = None,
    ) -> None:
```

Right after `self._session_id = session_id` (line 35), store it:

```python
        self._session_id = session_id
        self._data_vault = data_vault
```

Right after the existing `self._factory_takes_session_id = self._probe_factory_kwarg(...)` block (lines 41-43), add the second probe:

```python
        self._factory_takes_session_id = self._probe_factory_kwarg(
            runtime_factory, "session_id"
        )
        self._factory_takes_scratchpad_ds_env = self._probe_factory_kwarg(
            runtime_factory, "scratchpad_ds_env"
        )
```

Add a new private method anywhere in the class (e.g. right before `get_or_create`):

```python
    def _scratchpad_ds_env(self) -> dict[str, str] | None:
        """DS_* env values from the current vault state, or None without a vault.

        Built fresh on every call, not cached, so a pad created after a
        mid-session /connect sees the newly added connection — the same
        timing LocalScratchpadRuntime's own os.environ-at-spawn-time
        behaviour already has.
        """
        if self._data_vault is None:
            return None
        env: dict[str, str] = {}
        for conn in self._data_vault.list_connections():
            env.update(self._data_vault.env_for(conn["engine"], conn["name"]) or {})
        return env
```

Finally, `get_or_create` (lines 170-185) currently reads:

```python
    async def get_or_create(self, name: str) -> ScratchpadRuntime:
        """Return existing pad or create + start a new one."""
        if name not in self._pads:
            pad = self._runtime_factory(
                name=name,
                cells=self._cells,
                coding_provider=self._coding_provider,
                coding_model=self._coding_model,
                coding_api_key=self._coding_api_key,
                coding_base_url=self._coding_base_url,
                workspace_path=self._workspace_path,
                **({"session_id": self._session_id} if self._factory_takes_session_id else {}),
            )
            await pad.start()
            self._pads[name] = pad
        return self._pads[name]
```

Add the second conditional kwarg the same way:

```python
    async def get_or_create(self, name: str) -> ScratchpadRuntime:
        """Return existing pad or create + start a new one."""
        if name not in self._pads:
            pad = self._runtime_factory(
                name=name,
                cells=self._cells,
                coding_provider=self._coding_provider,
                coding_model=self._coding_model,
                coding_api_key=self._coding_api_key,
                coding_base_url=self._coding_base_url,
                workspace_path=self._workspace_path,
                **({"session_id": self._session_id} if self._factory_takes_session_id else {}),
                **(
                    {"scratchpad_ds_env": self._scratchpad_ds_env()}
                    if self._factory_takes_scratchpad_ds_env
                    else {}
                ),
            )
            await pad.start()
            self._pads[name] = pad
        return self._pads[name]
```

- [ ] **Step 11: Run the tests to verify they pass**

Run: `uv run pytest tests/test_scratchpad.py -k "derives_ds_env or skips_connection_whose_env_for_fails or without_vault_passes_none" -v`

Expected: all 3 PASS.

- [ ] **Step 12: Run the full scratchpad test file, and the data vault tests**

Run: `uv run pytest tests/test_scratchpad.py tests/test_data_vault.py -v`

Expected: PASS, no regressions.

- [ ] **Step 13: Commit**

```bash
git add anton/core/backends/manager.py tests/test_scratchpad.py
git commit -m "Derive the scratchpad DS_* overlay from the active vault"
```

---

### Task 3: Wire `ChatSessionConfig.data_vault` into `ScratchpadManager`

**Files:**
- Modify: `anton/core/session.py:1161-1170`

**Interfaces:**
- Consumes: `ScratchpadManager(..., data_vault: DataVault | None = None)` from Task 2.
- Produces: nothing further downstream — this is the last task, the point where CLI and desktop-sidecar cowork-server callers (both of which already populate `ChatSessionConfig.data_vault`, unchanged) start getting the real fix.

- [ ] **Step 1: Wire it through**

In `anton/core/session.py`, the `ScratchpadManager(...)` construction at lines 1161-1170 currently reads:

```python
        self._scratchpads = ScratchpadManager(
            runtime_factory=config.runtime_factory,
            coding_provider=coding_conn.provider,
            coding_model=config.llm_client.coding_model,
            coding_api_key=coding_conn.api_key or "",
            coding_base_url=coding_conn.base_url or "",
            cells=config.cells,
            workspace_path=config.workspace.base if config.workspace else None,
            session_id=config.session_id,
        )
```

Add `data_vault`:

```python
        self._scratchpads = ScratchpadManager(
            runtime_factory=config.runtime_factory,
            coding_provider=coding_conn.provider,
            coding_model=config.llm_client.coding_model,
            coding_api_key=coding_conn.api_key or "",
            coding_base_url=coding_conn.base_url or "",
            cells=config.cells,
            workspace_path=config.workspace.base if config.workspace else None,
            session_id=config.session_id,
            data_vault=config.data_vault,
        )
```

No new test here: constructing a full `ChatSession` needs a live LLM client and many other dependencies (`tests/test_session_acc_init.py`'s own docstring notes this, and the `session_id=config.session_id` line right above — added earlier for a related fix — has no dedicated wiring test either, for the same reason). Task 2's `ScratchpadManager`-level tests already cover the behavior this line switches on; this step is verified by the full suite in Step 3 plus the manual check in Step 2.

- [ ] **Step 2: Manual sanity check — the vault-to-subprocess-env chain works end to end**

Run this from the `anton` worktree root (no fixtures, no test harness — a plain script against a throwaway vault dir):

```bash
uv run python -c "
import asyncio, tempfile
from pathlib import Path
from anton.core.datasources.data_vault import LocalDataVault
from anton.core.backends.local import LocalScratchpadRuntime

async def main():
    with tempfile.TemporaryDirectory() as d:
        vault = LocalDataVault(Path(d))
        vault.save('postgres', 'prod', {'host': 'db.example.com', 'password': 'sanity-check-secret'})
        ds_env = {}
        for conn in vault.list_connections():
            ds_env.update(vault.env_for(conn['engine'], conn['name']) or {})
        print('computed ds_env:', ds_env)

        pad = LocalScratchpadRuntime(
            name='sanity', coding_provider='anthropic', coding_model='',
            coding_api_key='', coding_base_url='', scratchpad_ds_env=ds_env,
        )
        await pad.start()
        try:
            cell = await pad.execute('import os; print([k for k in os.environ if k.startswith(\"DS_\")])')
            print('subprocess saw:', cell.stdout.strip())
        finally:
            await pad.close()

asyncio.run(main())
"
```

Expected: `computed ds_env` shows a `DS_POSTGRES_PROD__PASSWORD`-style key, and `subprocess saw:` lists that same key — confirming the whole chain (`env_for` → `ScratchpadManager`'s derivation logic, exercised here inline → `LocalScratchpadRuntime.start()`'s overlay) actually produces the intended subprocess env, not just the individually-unit-tested pieces.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest`

Expected: PASS. Report any pre-existing failures unrelated to this change separately rather than attributing them to this task (per this workspace's testing convention).

- [ ] **Step 4: Commit**

```bash
git add anton/core/session.py
git commit -m "Pass the active data vault into the scratchpad manager"
```

---

## Self-Review Notes

- **Spec coverage:** every mechanism section of the design doc maps to a task — `env_for` reuse (Task 2, Part A), the `LocalScratchpadRuntime` strip-and-overlay (Task 1, Part A), the model/provider leak fix (Task 1, Part B), fresh-per-pad-creation derivation (Task 2, Part C — `_scratchpad_ds_env` docstring), `ChatSessionConfig.data_vault` reuse with no new field (Task 3). Explicitly-out-of-scope items (`scrub_credentials`, the cloud pod, the `pip install` env, `cowork-server`) have no task, by design.
- **Placeholder scan:** no TBDs; every step has literal code or an exact runnable command.
- **Type consistency:** `scratchpad_ds_env: dict[str, str] | None` is spelled identically in `LocalScratchpadRuntime.__init__`, `ScratchpadRuntimeFactory.__call__`, and `local_scratchpad_runtime_factory` across Tasks 1–2. `DataVault | None` is spelled identically in `ScratchpadManager.__init__` (Task 2) and matches the type already used in `ChatSessionConfig.data_vault` (Task 3, pre-existing).
- **No ticket references introduced:** every new comment/docstring in every code block above was checked and contains no ticket number. Two pre-existing comments quoted verbatim as surrounding context (the `(ENG-1124)` line in `base.py`, Task 2 Part B) are existing code you are shown but not asked to change — leave them exactly as they are.
