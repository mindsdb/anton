"""Scratchpad — persistent Python subprocess for stateful, notebook-like execution."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from anton.backends.base import ExecutionResult, ScratchpadBackend, ScratchpadRuntime

_CELL_TIMEOUT_DEFAULT = 120        # Default total timeout when no estimate given
_CELL_INACTIVITY_TIMEOUT = 30      # Max silence between output lines before killing
_INSTALL_TIMEOUT = 120
_MAX_OUTPUT = 10_000
_PROGRESS_MARKER = "__ANTON_PROGRESS__"


def _compute_timeouts(estimated_seconds: int) -> tuple[float, float]:
    """Compute (total_timeout, inactivity_timeout) from estimated execution time.

    - If estimate is 0: use defaults (120s total, 30s inactivity).
    - Otherwise: total = max(estimate * 2, estimate + 30) with no cap.
      Inactivity = min(max(estimate * 0.5, 30), 60).
    """
    if estimated_seconds <= 0:
        return float(_CELL_TIMEOUT_DEFAULT), float(_CELL_INACTIVITY_TIMEOUT)
    total = max(estimated_seconds * 2, estimated_seconds + 30)
    inactivity = min(max(estimated_seconds * 0.5, 30), 60)
    return float(total), float(inactivity)


@dataclass
class Cell:
    code: str
    stdout: str
    stderr: str
    error: str | None
    description: str = ""
    estimated_time: str = ""
    logs: str = ""


@dataclass
class Scratchpad:
    name: str
    cells: list[Cell] = field(default_factory=list)
    _coding_provider: str = field(default="anthropic", repr=False)
    _coding_model: str = field(default="", repr=False)
    _coding_api_key: str = field(default="", repr=False)
    _venvs_base: Path = field(
        default_factory=lambda: Path("~/.anton/scratchpad-venvs").expanduser(),
        repr=False,
    )
    _runtime: ScratchpadRuntime | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._runtime is None:
            from anton.backends.local import LocalScratchpadRuntime

            self._runtime = LocalScratchpadRuntime(
                name=self.name,
                venvs_base=self._venvs_base,
                env_overrides={},
                coding_provider=self._coding_provider,
                coding_model=self._coding_model,
                coding_api_key=self._coding_api_key,
            )

    def __getattr__(self, attr: str):
        runtime = object.__getattribute__(self, "_runtime")
        if runtime is not None and hasattr(runtime, attr):
            return getattr(runtime, attr)
        raise AttributeError(attr)

    async def start(self) -> None:
        if self._runtime is None:
            self.__post_init__()
        await self._runtime.start()  # type: ignore[union-attr]

    async def execute(
        self,
        code: str,
        *,
        description: str = "",
        estimated_time: str = "",
        estimated_seconds: int = 0,
    ) -> Cell:
        async for item in self.execute_streaming(
            code,
            description=description,
            estimated_time=estimated_time,
            estimated_seconds=estimated_seconds,
        ):
            if isinstance(item, Cell):
                return item
        return Cell(code=code, stdout="", stderr="", error="No result produced.")

    async def execute_streaming(
        self,
        code: str,
        *,
        description: str = "",
        estimated_time: str = "",
        estimated_seconds: int = 0,
    ) -> AsyncIterator[str | Cell]:
        if self._runtime is None:
            self.__post_init__()

        async for event in self._runtime.execute_streaming(  # type: ignore[union-attr]
            code,
            estimated_seconds=estimated_seconds,
        ):
            if isinstance(event, str):
                yield event
                continue

            if isinstance(event, ExecutionResult):
                cell = Cell(
                    code=code,
                    stdout=event.stdout,
                    stderr=event.stderr,
                    error=event.error,
                    description=description,
                    estimated_time=estimated_time,
                    logs=event.logs,
                )
                self.cells.append(cell)
                yield cell
                return

        cell = Cell(code=code, stdout="", stderr="", error="No result produced.", description=description, estimated_time=estimated_time)
        self.cells.append(cell)
        yield cell

    async def install_packages(self, packages: list[str]) -> str:
        if self._runtime is None:
            self.__post_init__()
        return await self._runtime.install_packages(packages)  # type: ignore[union-attr]

    async def reset(self) -> None:
        if self._runtime is None:
            self.__post_init__()
        self.cells.clear()
        await self._runtime.reset()  # type: ignore[union-attr]

    async def close(self) -> None:
        if self._runtime is None:
            self.__post_init__()
        await self._runtime.close()  # type: ignore[union-attr]

    async def remove(self) -> None:
        if self._runtime is None:
            self.__post_init__()
        await self._runtime.remove()  # type: ignore[union-attr]

    def view(self) -> str:
        """Format all cells with their outputs."""
        if not self.cells:
            return f"Scratchpad '{self.name}' is empty."

        parts: list[str] = []
        for i, cell in enumerate(self.cells):
            header = f"--- Cell {i + 1}"
            if cell.description:
                header += f": {cell.description}"
            header += " ---"
            parts.append(header)
            parts.append(cell.code)
            if cell.stdout:
                parts.append(f"[output]\n{cell.stdout}")
            if cell.logs:
                parts.append(f"[logs]\n{cell.logs}")
            if cell.stderr:
                parts.append(f"[stderr]\n{cell.stderr}")
            if cell.error:
                parts.append(f"[error]\n{cell.error}")
            if not cell.stdout and not cell.logs and not cell.stderr and not cell.error:
                parts.append("(no output)")
        return "\n".join(parts)

    @staticmethod
    def _truncate_output(text: str, max_lines: int = 20, max_chars: int = 2000) -> str:
        """Truncate output to *max_lines* / *max_chars*, whichever is shorter."""
        lines = text.split("\n")
        # Apply line limit
        if len(lines) > max_lines:
            kept = "\n".join(lines[:max_lines])
            remaining = len(lines) - max_lines
            return kept + f"\n... ({remaining} more lines)"
        # Apply char limit (don't cut mid-line)
        if len(text) > max_chars:
            total = 0
            kept_lines: list[str] = []
            for line in lines:
                if total + len(line) + 1 > max_chars and kept_lines:
                    break
                kept_lines.append(line)
                total += len(line) + 1
            return "\n".join(kept_lines) + "\n... (truncated)"
        return text

    def render_notebook(self) -> str:
        """Return a clean markdown notebook-style summary of all cells."""
        # Filter out empty/whitespace-only cells
        numbered: list[tuple[int, Cell]] = []
        idx = 0
        for cell in self.cells:
            idx += 1
            if not cell.code.strip():
                continue
            numbered.append((idx, cell))

        if not numbered:
            return f"Scratchpad '{self.name}' has no cells."

        parts: list[str] = [f"## Scratchpad: {self.name} ({len(numbered)} cells)"]

        for i, (num, cell) in enumerate(numbered):
            header = f"\n### Cell {num}"
            if cell.description:
                header += f" \u2014 {cell.description}"
            parts.append(header)
            parts.append(f"```python\n{cell.code}\n```\n")

            if cell.error:
                # Show only the last traceback line
                last_line = cell.error.strip().split("\n")[-1]
                parts.append(f"**Error:** `{last_line}`")
                # If there was partial output before the error, show it
                if cell.stdout:
                    truncated = self._truncate_output(cell.stdout.rstrip("\n"))
                    parts.append(f"**Partial output:**\n```\n{truncated}\n```\n")
            elif cell.stdout:
                truncated = self._truncate_output(cell.stdout.rstrip("\n"))
                parts.append(f"**Output:**\n```\n{truncated}\n```\n")

            if cell.logs:
                truncated_logs = self._truncate_output(cell.logs.rstrip("\n"), max_lines=10, max_chars=1000)
                parts.append(f"**Logs:**\n```\n{truncated_logs}\n```\n")

            if i < len(numbered) - 1:
                parts.append("---")

        return "\n".join(parts)


class ScratchpadManager:
    """Manages named scratchpad instances."""

    def __init__(
        self,
        coding_provider: str = "anthropic",
        coding_model: str = "",
        coding_api_key: str = "",
        workspace_path: Path | None = None,
        *,
        backend: ScratchpadBackend | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._pads: dict[str, Scratchpad] = {}
        self._coding_provider: str = coding_provider
        self._coding_model: str = coding_model
        self._coding_api_key: str = coding_api_key
        self._workspace_path: Path | None = workspace_path
        
        if backend is None:
            from anton.backends.local import LocalScratchpadBackend
            self._backend = LocalScratchpadBackend()
        else:
            self._backend = backend

        self._env: dict[str, str] = dict(env) if env else {}
        self._available_packages: list[str] = self._backend.probe_packages()

    @staticmethod
    def probe_packages() -> list[str]:
        """Return sorted list of installed package distribution names."""
        from importlib.metadata import distributions

        return sorted({d.metadata["Name"] for d in distributions()})

    async def get_or_create(self, name: str) -> Scratchpad:
        """Return existing pad or create + start a new one."""
        if name not in self._pads:
            runtime = await self._backend.create_runtime(
                name,
                workspace_path=self._workspace_path,
                env=self._env,
                coding_provider=self._coding_provider,
                coding_model=self._coding_model,
                coding_api_key=self._coding_api_key,
            )
            pad = Scratchpad(
                name=name,
                _coding_provider=self._coding_provider,
                _coding_model=self._coding_model,
                _coding_api_key=self._coding_api_key,
                _runtime=runtime,
            )
            await pad.start()
            self._pads[name] = pad
        return self._pads[name]

    async def remove(self, name: str) -> str:
        """Kill and fully delete a scratchpad (including its persistent venv)."""
        pad = self._pads.pop(name, None)
        if pad is None:
            return f"No scratchpad named '{name}'."
        await pad.remove()
        return f"Scratchpad '{name}' removed."

    def list_pads(self) -> list[str]:
        return list(self._pads.keys())

    async def close_all(self) -> None:
        """Cleanup all scratchpads on session end."""
        for pad in self._pads.values():
            await pad.close()
        self._pads.clear()
