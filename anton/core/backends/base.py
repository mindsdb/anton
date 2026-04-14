"""ScratchpadRuntime ABC — pluggable scratchpad backend contract.

Core interface shared by LocalScratchpadRuntime (CLI) and
our cloud ScratchpadRuntime (Enterprise).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class Cell:
    """A single scratchpad execution unit — code in, outputs out."""

    code: str
    stdout: str
    stderr: str
    error: str | None
    description: str = ""
    estimated_time: str = ""
    logs: str = ""


class ScratchpadRuntime(ABC):
    """Abstract base class for scratchpad execution backends.

    Concrete implementations provide a specific execution environment
    (local venv, Docker container, etc.). The shared display, compaction,
    and timeout logic lives here so all backends benefit automatically.
    """

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
    ) -> None:
        self.name = name
        self._coding_provider = coding_provider
        self._coding_model = coding_model
        self._coding_api_key = coding_api_key
        self._coding_base_url = coding_base_url
        self.cells: list[Cell] = cells if cells is not None else []
        self._workspace_path = workspace_path or Path("~/.anton").expanduser()
        self._installed_packages: set[str] = set()

    @abstractmethod
    async def start(self) -> None:
        """Launch the runtime environment."""

    @abstractmethod
    async def reset(self) -> None:
        """Kill the runtime, clear cells, and restart."""

    @abstractmethod
    async def close(self) -> None:
        """Shut down the runtime, preserving any persistent resources."""

    @abstractmethod
    async def cancel(self) -> None:
        """Cancel the currently running cell and restart the runtime."""

    @abstractmethod
    async def install_packages(self, packages: list[str]) -> str:
        """Install Python packages into the runtime environment."""

    @abstractmethod
    async def execute_streaming(
        self,
        code: str,
        *,
        description: str = "",
        estimated_time: str = "",
        estimated_seconds: int = 0,
    ):
        """Execute code and yield progress strings then a final Cell."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Release all backend-specific resources (venv dir, containers, etc.).

        Called by ScratchpadManager.remove() to fully destroy this runtime.
        Unlike close(), cleanup() removes persistent storage too.
        """

    async def execute(
        self,
        code: str,
        *,
        description: str = "",
        estimated_time: str = "",
        estimated_seconds: int = 0,
    ) -> Cell:
        """Drain execute_streaming() and return just the final Cell."""
        async for item in self.execute_streaming(
            code,
            description=description,
            estimated_time=estimated_time,
            estimated_seconds=estimated_seconds,
        ):
            if isinstance(item, Cell):
                return item
        return Cell(code=code, stdout="", stderr="", error="No result produced.")

    def view(self) -> str:
        """Format all cells with their outputs for LLM consumption."""
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

    def render_notebook(self) -> str:
        """Return a clean markdown notebook-style summary of all cells."""
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
                last_line = cell.error.strip().split("\n")[-1]
                parts.append(f"**Error:** `{last_line}`")
                if cell.stdout:
                    truncated = self._truncate_output(cell.stdout.rstrip("\n"))
                    parts.append(f"**Partial output:**\n```\n{truncated}\n```\n")
            elif cell.stdout:
                truncated = self._truncate_output(cell.stdout.rstrip("\n"))
                parts.append(f"**Output:**\n```\n{truncated}\n```\n")

            if cell.logs:
                truncated_logs = self._truncate_output(
                    cell.logs.rstrip("\n"), max_lines=10, max_chars=1000
                )
                parts.append(f"**Logs:**\n```\n{truncated_logs}\n```\n")

            if i < len(numbered) - 1:
                parts.append("---")

        return "\n".join(parts)

    @staticmethod
    def _truncate_output(text: str, max_lines: int = 20, max_chars: int = 2000) -> str:
        """Truncate output to max_lines / max_chars, whichever is shorter."""
        lines = text.split("\n")
        if len(lines) > max_lines:
            kept = "\n".join(lines[:max_lines])
            remaining = len(lines) - max_lines
            return kept + f"\n... ({remaining} more lines)"
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

    def _compact_cells(self) -> bool:
        """Collapse old cells into a summary cell to reduce context size.

        Keeps the most recent settings.cell_keep_recent cells intact. Returns
        True if compaction actually happened.
        """
        from anton.core.settings import CoreSettings

        keep = CoreSettings().cell_keep_recent
        if len(self.cells) <= keep + 1:
            return False

        to_compact = self.cells[:-keep]
        recent = self.cells[-keep:]

        summary_lines: list[str] = []
        for i, cell in enumerate(to_compact, 1):
            status = "error" if cell.error else "ok"
            desc = cell.description or f"Cell {i}"
            first_line = ""
            output = cell.stdout or cell.error or ""
            if output:
                first_line = output.strip().split("\n")[0][:120]
            summary_lines.append(f"  [{status}] {desc}: {first_line}")

        summary_text = f"# Compacted {len(to_compact)} earlier cells:\n" + "\n".join(
            summary_lines
        )
        summary_cell = Cell(
            code="# (compacted — see summary above)",
            stdout=summary_text,
            stderr="",
            error=None,
            description=f"Summary of cells 1–{len(to_compact)}",
        )
        self.cells = [summary_cell] + recent
        return True


class ScratchpadRuntimeFactory(Protocol):
    def __call__(
        self,
        *,
        name: str,
        cells: list[Cell] | None,
        coding_provider: str,
        coding_model: str,
        coding_api_key: str,
        coding_base_url: str,
        workspace_path: Path | None,
    ) -> ScratchpadRuntime: ...