from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
from dataclasses import dataclass, field


# Boot script
_BOOT_SCRIPT_PATH = Path(__file__).parent / "scratchpad_boot.py"

# Cell settings
_CELL_TIMEOUT_DEFAULT = 120        # Default total timeout when no estimate given
_CELL_INACTIVITY_TIMEOUT = 30      # Max silence between output lines before killing
_CELL_INACTIVITY_AFTER_PROGRESS = 60  # Grace window after a progress() call
_KEEP_RECENT = 5  # Number of recent cells to keep during compaction

# Installation settings
_INSTALL_TIMEOUT = 120

# Delimiters and markers
_CELL_DELIM = "__ANTON_CELL_END__"
_RESULT_START = "__ANTON_RESULT__"
_RESULT_END = "__ANTON_RESULT_END__"
_PROGRESS_MARKER = "__ANTON_PROGRESS__"


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
class ScratchpadRuntime(ABC):
    name: str
    cells: list[Cell] = field(default_factory=list)

    _boot_path: str = field(default=str(_BOOT_SCRIPT_PATH), repr=False)
    _coding_provider: str = field(default="anthropic", repr=False)
    _coding_model: str = field(default="", repr=False)
    _coding_api_key: str = field(default="", repr=False)
    _installed_packages: set[str] = field(default_factory=set, repr=False)

    @abstractmethod
    async def start(self) -> None:
        """
        Start the runtime.

        This method should launch the boot script in a runtime environment.
        """
        pass

    @abstractmethod
    async def reset(self) -> None:
        """
        Reset the runtime.

        This method should kill the runtime, clear the cells and restart the runtime.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the runtime.

        This method should kill the runtime and clean up any resources.
        """

    @abstractmethod
    async def cancel(self) -> None:
        """
        Kill the current execution and restart the runtime.

        This method should kill the runtime, record a cancelled cell and restart.
        """
        pass
        
    @abstractmethod
    async def install_packages(self, packages: list[str]) -> str:
        """
        Install Python packages into the runtime.

        This method should install the specified packages into the runtime environment.
        """
        pass

    @abstractmethod
    async def execute_streaming(
        self,
        code: str,
        *,
        description: str = "",
        estimated_time: str = "",
        estimated_seconds: int = 0,
    ):
        """
        Execute code in the runtime and yield progress strings and a final Cell.

        This method should execute the provided code in the runtime environment and yield progress strings and a final Cell.
        """
        pass

    async def execute(
        self,
        code: str,
        *,
        description: str = "",
        estimated_time: str = "",
        estimated_seconds: int = 0,
    ) -> Cell:
        """Send code to the runtime, read the JSON result, return a Cell.

        Backward-compatible wrapper around execute_streaming() that drains
        all events and returns just the final Cell.
        """
        async for item in self.execute_streaming(
            code,
            description=description,
            estimated_time=estimated_time,
            estimated_seconds=estimated_seconds,
        ):
            if isinstance(item, Cell):
                return item
        # Should not reach here, but just in case
        return Cell(code=code, stdout="", stderr="", error="No result produced.")

    def view(self) -> str:
        """
        Format all cells with their outputs.
        """
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
        """
        Truncate output to *max_lines* / *max_chars*, whichever is shorter.
        """
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
        """
        Return a clean markdown notebook-style summary of all cells.
        """
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

    def _compact_cells(self) -> bool:
        """
        Collapse old cells into a single summary cell to reduce context size.

        Keeps the most recent _KEEP_RECENT cells intact.  Older cells are
        replaced by one summary cell with a one-line-per-cell digest.

        Returns True if compaction actually happened.
        """
        if len(self.cells) <= _KEEP_RECENT + 1:
            return False

        to_compact = self.cells[: -_KEEP_RECENT]
        recent = self.cells[-_KEEP_RECENT:]

        summary_lines: list[str] = []
        for i, cell in enumerate(to_compact, 1):
            status = "error" if cell.error else "ok"
            desc = cell.description or f"Cell {i}"
            first_line = ""
            output = cell.stdout or cell.error or ""
            if output:
                first_line = output.strip().split("\n")[0][:120]
            summary_lines.append(f"  [{status}] {desc}: {first_line}")

        summary_text = (
            f"# Compacted {len(to_compact)} earlier cells:\n"
            + "\n".join(summary_lines)
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

    @staticmethod
    def _compute_timeouts(estimated_seconds: int) -> tuple[float, float]:
        """
        Compute (total_timeout, inactivity_timeout) from estimated execution time.

        - If estimate is 0: use defaults (120s total, 30s inactivity).
        - Otherwise: total = max(estimate * 2, estimate + 30) with no cap.
        Inactivity = max(estimate * 0.5, 30) — no hard cap, scales with estimate.
        """
        if estimated_seconds <= 0:
            return float(_CELL_TIMEOUT_DEFAULT), float(_CELL_INACTIVITY_TIMEOUT)
        total = max(estimated_seconds * 2, estimated_seconds + 30)
        inactivity = max(estimated_seconds * 0.5, 30)
        return float(total), float(inactivity)
