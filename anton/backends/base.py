from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """
    Final cell result produced by a scratchpad runner.

    This mirrors the JSON payload returned by `anton/scratchpad_boot.py`.
    """

    stdout: str
    stderr: str
    logs: str
    error: str | None


ExecutionEvent = str | ExecutionResult


class ScratchpadRuntime(Protocol):
    """A stateful scratchpad execution session (local, docker, etc.)."""

    async def start(self) -> None:
        """Start the underlying runner process/session."""

    async def execute_streaming(
        self,
        code: str,
        *,
        estimated_seconds: int = 0,
    ) -> AsyncIterator[ExecutionEvent]:
        """Execute code, yielding progress messages and a final `ExecutionResult`."""

    async def install_packages(self, packages: list[str]) -> str:
        """Install packages into this runtime's environment and return installer output."""

    async def reset(self) -> None:
        """Restart the runner while keeping the environment (if supported)."""

    async def close(self) -> None:
        """Stop the runner; the environment may remain persisted."""

    async def remove(self) -> None:
        """Delete the runtime and its persisted environment (if any)."""


class ScratchpadBackend(Protocol):
    """Factory for creating scratchpad runtimes."""

    async def create_runtime(
        self,
        name: str,
        *,
        workspace_path: Path | None,
        env: Mapping[str, str],
        coding_provider: str,
        coding_model: str,
        coding_api_key: str,
    ) -> ScratchpadRuntime:
        """Create a runtime for a scratchpad name (not necessarily started)."""

    def probe_packages(self) -> list[str]:
        """Return installed package names available to the backend (best-effort)."""
