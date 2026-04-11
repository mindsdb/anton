"""ScratchpadManager — lifecycle manager for named scratchpad runtimes."""

from __future__ import annotations

from pathlib import Path

from anton.core.backends.base import ScratchpadRuntime
from anton.core.backends.local import LocalScratchpadRuntime


class ScratchpadManager:
    """Manages named scratchpad runtime instances."""

    def __init__(
        self,
        coding_provider: str = "anthropic",
        coding_model: str = "",
        coding_api_key: str = "",
        coding_base_url: str = "",
        workspace_path: Path | None = None,
    ) -> None:
        self._pads: dict[str, ScratchpadRuntime] = {}
        self._coding_provider = coding_provider
        self._coding_model = coding_model
        self._coding_api_key = coding_api_key
        self._coding_base_url = coding_base_url
        self._workspace_path = workspace_path
        self._available_packages: list[str] = self.probe_packages()

    @property
    def pads(self) -> dict[str, ScratchpadRuntime]:
        """Read-only view of the active scratchpad runtimes."""
        return self._pads

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
            pad = LocalScratchpadRuntime(
                name=name,
                coding_provider=self._coding_provider,
                coding_model=self._coding_model,
                coding_api_key=self._coding_api_key,
                coding_base_url=self._coding_base_url,
                workspace_path=self._workspace_path,
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
