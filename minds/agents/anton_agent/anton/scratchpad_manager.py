from pathlib import Path

from minds.common.settings.app_settings import get_app_settings

from .backends.base import ScratchpadRuntime, ScratchpadRuntimeFactory

app_settings = get_app_settings()


class ScratchpadManager:
    """Manages named scratchpad instances."""

    def __init__(
        self,
        backend: str = "docker",
        coding_provider: str = app_settings.default_models.default_provider,
        coding_model: str = "",
        coding_api_key: str = "",
        workspace_path: Path = Path("~/.anton").expanduser(),
        extra_env: dict[str, str] | None = None,
        events: list[dict] = None,
    ) -> None:
        # Each conversation will only be associated with a single scratchpad.
        self._pad_name = f"anton_scratchpad_{extra_env['ANTON_MINDS_CONVERSATION_ID']}"
        self._pad: ScratchpadRuntime | None = None

        self._backend: str = backend
        self._coding_provider: str = coding_provider
        self._coding_model: str = coding_model
        self._coding_api_key: str = coding_api_key
        self._workspace_path: Path = workspace_path
        self._extra_env: dict[str, str] | None = extra_env
        self._events: list[dict] = events or []

        self._available_packages: list[str] = self.probe_packages()

    @staticmethod
    def probe_packages() -> list[str]:
        """Return sorted list of installed package distribution names."""
        from importlib.metadata import distributions

        return sorted({d.metadata["Name"] for d in distributions()})

    async def get_or_create(self, start: bool = True) -> ScratchpadRuntime:
        """Return existing pad or create + start a new one."""
        if self._pad is None:
            pad = ScratchpadRuntimeFactory().create(
                name=self._pad_name,
                backend=self._backend,
                coding_provider=self._coding_provider,
                coding_model=self._coding_model,
                coding_api_key=self._coding_api_key,
                workspace_path=self._workspace_path,
                extra_env=self._extra_env,
                events=self._events,
            )
            self._pad = pad

        if start:
            # Start should be an idempotent operation, so it can be called multiple times.
            await self._pad.start()

        return self._pad

    async def remove(self) -> str:
        """Kill and fully delete a scratchpad (including its persistent resources)."""
        pad = await self.get_or_create(start=False)
        await pad.close(cleanup=True)
        self._pad = None
        return "Scratchpad removed."

    async def close_all(self) -> None:
        """Cleanup all scratchpads on session end."""
        if self._pad is not None:
            await self._pad.close()
        self._pad = None
