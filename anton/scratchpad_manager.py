from pathlib import Path

from anton.backends.base import ScratchpadRuntime, ScratchpadRuntimeFactory


class ScratchpadManager:
    """Manages named scratchpad instances."""

    def __init__(
        self,
        backend: str = "local",
        coding_provider: str = "anthropic",
        coding_model: str = "",
        coding_api_key: str = "",
        workspace_path: Path = Path("~/.anton").expanduser(),
    ) -> None:
        self._backend: str = backend
        self._pads: dict[str, ScratchpadRuntime] = {}
        self._coding_provider: str = coding_provider
        self._coding_model: str = coding_model
        self._coding_api_key: str = coding_api_key
        self._workspace_path: Path = workspace_path

        self._available_packages: list[str] = self.probe_packages()

    @staticmethod
    def probe_packages() -> list[str]:
        """Return sorted list of installed package distribution names."""
        from importlib.metadata import distributions

        return sorted({d.metadata["Name"] for d in distributions()})

    async def get_or_create(self, name: str) -> ScratchpadRuntime:
        """Return existing pad or create + start a new one."""
        if name not in self._pads:
            pad = ScratchpadRuntimeFactory().create(
                name=name,
                backend=self._backend,
                coding_provider=self._coding_provider,
                coding_model=self._coding_model,
                coding_api_key=self._coding_api_key,
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
        await pad.close(cleanup=True)
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
