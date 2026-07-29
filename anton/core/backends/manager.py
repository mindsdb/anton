"""ScratchpadManager — lifecycle manager for named scratchpad runtimes."""

from __future__ import annotations

import inspect
from pathlib import Path

from anton.core.backends.base import Cell, ScratchpadRuntime, ScratchpadRuntimeFactory


class ScratchpadManager:
    """Manages named scratchpad runtime instances."""

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
        self._pads: dict[str, ScratchpadRuntime] = {}
        self._runtime_factory = runtime_factory
        self._coding_provider = coding_provider
        self._coding_model = coding_model
        self._coding_api_key = coding_api_key
        self._coding_base_url = coding_base_url
        self._cells = cells
        self._workspace_path = workspace_path
        # Conversation id, forwarded to each runtime so its namespace snapshot is
        # scoped per conversation (ENG-1124).
        self._session_id = session_id
        # Only pass `session_id` to factories that accept it. A default on the Protocol
        # does not adapt an existing callable, so passing it unconditionally raises
        # `TypeError: unexpected keyword argument` for an out-of-tree factory written
        # against the previous signature. Same signature-probe pattern cowork-server
        # uses to stay compatible with older anton builds. Resolved once, not per call.
        self._factory_takes_session_id = self._probe_factory_kwarg(
            runtime_factory, "session_id"
        )
        self._available_packages: list[str] = self.probe_packages()

    @staticmethod
    def _probe_factory_kwarg(factory, name: str) -> bool:
        """Whether `factory` accepts keyword `name`. Assumes yes if unintrospectable."""
        try:
            params = inspect.signature(factory).parameters
        except (TypeError, ValueError):
            return True  # e.g. a C callable or a mock — let the call decide
        if name in params:
            return True
        return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    @property
    def pads(self) -> dict[str, ScratchpadRuntime]:
        """Read-only view of the active scratchpad runtimes."""
        return self._pads

    def _agent_pads_file(self):
        """Where this conversation's agent-chosen pad names are recorded, or None.

        Sits inside the namespace-snapshot directory so it is scoped per conversation
        and reclaimed by the same cleanup (cowork-server prunes that directory on
        conversation delete). Underscore-prefixed so the `*.pkl` snapshot glob skips it.

        Returns None without a `session_id`, matching `snapshot_dir`, which refuses to
        hand out a directory at all when the conversation scope is unknown. That is also
        the behaviour this record wants on its own terms: anything shared across unscoped
        conversations (bare CLI, tests) would pool their pad names, and the guard would
        start challenging on names from unrelated past sessions. Unscoped callers keep the
        previous in-memory-only behaviour.
        """
        if not self._session_id:
            return None
        try:
            from anton.core.backends.local import default_venvs_base, snapshot_dir

            base = snapshot_dir(default_venvs_base(self._workspace_path), self._session_id)
            return None if base is None else base / "_agent_pads.json"
        except Exception:
            return None

    def agent_pads(self) -> set[str]:
        """Pad names the AGENT has exec'd in this conversation, including earlier turns.

        The single-scratchpad guard needs this. It originally consulted only an
        in-memory set on `ChatSession` — but cowork-server builds a fresh `ChatSession`
        (and so a fresh manager) per user message, and the agent switches pad names
        precisely *at* turn boundaries, so that set was always empty exactly when the
        guard needed it. It fired 0 times across 676 cells of a real session that
        accumulated 22 pad names (ENG-1124 Fix 5).

        Deliberately NOT derived from `self._pads` or from the snapshot files: both
        include system-created pads (the artifact backend launcher's slug pad), which
        must never count against the agent. Only names the guard explicitly recorded.
        """
        path = self._agent_pads_file()
        if path is None:
            return set()
        try:
            import json

            names = json.loads(path.read_text(encoding="utf-8"))
            return {n for n in names if isinstance(n, str)} if isinstance(names, list) else set()
        except (OSError, ValueError):
            return set()

    def record_agent_pad(self, name: str) -> None:
        """Remember that the agent exec'd `name`, so a later turn's guard can see it.

        Best-effort: losing this only returns the guard to its previous behaviour.
        """
        path = self._agent_pads_file()
        if path is None or not name:
            return
        try:
            import json

            current = self.agent_pads()
            if name in current:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(sorted(current | {name})), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass

    @property
    def available_packages(self) -> list[str]:
        """Sorted list of installed package distribution names."""
        return self._available_packages

    @staticmethod
    def probe_packages() -> list[str]:
        """Return sorted list of installed package distribution names."""
        from importlib.metadata import distributions

        return sorted({d.metadata["Name"] for d in distributions()})

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

    async def remove(self, name: str) -> str:
        """Kill and fully delete a scratchpad (including its persistent venv)."""
        pad = self._pads.pop(name, None)
        if pad is None:
            return f"No scratchpad named '{name}'."
        await pad.cleanup()
        return f"Scratchpad '{name}' removed."

    def list_pads(self) -> list[str]:
        return list(self._pads.keys())

    async def cancel_all_running(self) -> None:
        """Cancel running executions in all scratchpads and restart them."""
        for pad in self._pads.values():
            await pad.cancel()

    async def close_all(self) -> None:
        """Cleanup all scratchpads on session end."""
        for pad in self._pads.values():
            await pad.close()
        self._pads.clear()

    async def venv_python(self, name: str = "main") -> str | None:
        """Return the Python interpreter path of the named scratchpad.

        Provisions the scratchpad on demand so callers don't have to
        synchronize with whatever cell the LLM happens to be running.
        Returns None when the runtime can't expose a local interpreter
        (e.g. remote backends).
        """
        pad = await self.get_or_create(name)
        return pad.venv_python()
