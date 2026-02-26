from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

from anton.backends.base import ExecutionEvent, ExecutionResult

CELL_DELIM = "__ANTON_CELL_END__"
RESULT_START = "__ANTON_RESULT__"
RESULT_END = "__ANTON_RESULT_END__"
PROGRESS_MARKER = "__ANTON_PROGRESS__"


def boot_script_path() -> Path:
    """Return a filesystem path to `anton/scratchpad_boot.py` when available."""
    try:
        from importlib.resources import files

        return files("anton").joinpath("scratchpad_boot.py")  # type: ignore[return-value]
    except Exception:
        # Fallback for environments where resources/files aren't available
        return Path(__file__).resolve().parents[1] / "scratchpad_boot.py"


def boot_script_source() -> str:
    """Load the source of `anton/scratchpad_boot.py` as text."""
    try:
        from importlib.resources import files

        return files("anton").joinpath("scratchpad_boot.py").read_text(encoding="utf-8")
    except Exception:
        return boot_script_path().read_text(encoding="utf-8")


def encode_cell(code: str) -> bytes:
    """Encode a cell for the runner stdin protocol."""
    return (code + "\n" + CELL_DELIM + "\n").encode()


async def read_execution_events(
    readline: Callable[[], Awaitable[bytes]],
    *,
    total_timeout: float,
    inactivity_timeout: float,
) -> AsyncIterator[ExecutionEvent]:
    """Read runner stdout and yield progress + a final ExecutionResult.

    The runner is expected to emit:
    - progress lines: `__ANTON_PROGRESS__ <message>`
    - result section delimited by:
      `__ANTON_RESULT__` ... JSON lines ... `__ANTON_RESULT_END__`
    """
    import time as _time

    lines: list[str] = []
    in_result = False
    start = _time.monotonic()

    while True:
        elapsed = _time.monotonic() - start
        remaining_total = total_timeout - elapsed
        if remaining_total <= 0:
            raise asyncio.TimeoutError(f"Cell timed out after {total_timeout:.0f}s total")

        line_timeout = min(inactivity_timeout, remaining_total)
        try:
            raw = await asyncio.wait_for(readline(), timeout=line_timeout)
        except asyncio.TimeoutError:
            elapsed_now = _time.monotonic() - start
            if elapsed_now >= total_timeout - 0.5:
                raise asyncio.TimeoutError(f"Cell timed out after {total_timeout:.0f}s total") from None
            raise asyncio.TimeoutError(
                f"Cell killed after {inactivity_timeout:.0f}s of inactivity "
                f"(no output or progress() calls)"
            ) from None

        if not raw:
            yield ExecutionResult(
                stdout="",
                stderr="",
                logs="",
                error="Process exited unexpectedly.",
            )
            return

        line = raw.decode().rstrip("\n")

        if line.startswith(PROGRESS_MARKER):
            message = line[len(PROGRESS_MARKER) :].strip()
            yield message
            continue

        if line == RESULT_START:
            in_result = True
            continue
        if line == RESULT_END:
            break
        if in_result:
            lines.append(line)

    data = json.loads("\n".join(lines)) if lines else {}
    yield ExecutionResult(
        stdout=data.get("stdout", "") if isinstance(data, dict) else "",
        stderr=data.get("stderr", "") if isinstance(data, dict) else "",
        logs=data.get("logs", "") if isinstance(data, dict) else "",
        error=data.get("error") if isinstance(data, dict) else "Invalid result payload.",
    )

