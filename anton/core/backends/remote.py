"""RemoteScratchpadRuntime — HTTP-based scratchpad backend.

Connects to the scratchpad web service running on a user's
Lightsail instance (sp_{hash}.4nton.ai). Implements the same
ScratchpadRuntime ABC as LocalScratchpadRuntime but delegates
all execution to the remote service via REST + SSE.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

import httpx2 as httpx

from anton.core.backends.base import Cell, ScratchpadRuntime, stopped_line
from anton.core.backends.wire import NOTHING_KEPT
from anton.core.llm.tracing import get_trace_context, trace_context_to_json


class RemoteScratchpadRuntime(ScratchpadRuntime):
    """Runs scratchpad cells on a remote Lightsail instance via HTTP."""

    def __init__(
        self,
        name: str,
        *,
        endpoint_url: str,
        api_key: str,
        coding_provider: str = "",
        coding_model: str = "",
        coding_api_key: str = "",
        coding_base_url: str = "",
        cells: list[Cell] | None = None,
        workspace_path: Path | None = None,
    ) -> None:
        super().__init__(
            name,
            coding_provider=coding_provider,
            coding_model=coding_model,
            coding_api_key=coding_api_key,
            coding_base_url=coding_base_url,
            cells=cells,
            workspace_path=workspace_path,
        )
        self._endpoint_url = endpoint_url
        self._api_key = api_key
        # The code of the cell the service runs now, which a cancel records.
        self._running_code: str | None = None

    @property
    def process_running(self) -> bool:
        """True: the service runs the pad's process and starts it as it
        needs, which the host can't see."""
        return True

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "anton-remote-scratchpad/1.0",
        }

    async def _post(self, path: str, body: dict | None = None) -> dict:
        """POST to the remote service and return parsed JSON."""
        url = f"{self._endpoint_url}{path}"
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(url, json=body or {}, headers=self._headers())
            if resp.status_code >= 400:
                raise RuntimeError(f"Remote scratchpad error ({resp.status_code}): {resp.text}")
            return resp.json()

    async def _sse(self, path: str, body: dict) -> AsyncIterator[dict]:
        """POST to an SSE endpoint and yield parsed events."""
        url = f"{self._endpoint_url}{path}"
        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream("POST", url, json=body, headers=self._headers()) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", errors="replace")
                    raise RuntimeError(f"Remote scratchpad error ({resp.status_code}): {text}")

                buffer = ""
                async for chunk in resp.aiter_bytes():
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        for line in event_str.split("\n"):
                            if line.startswith("data: "):
                                try:
                                    yield json.loads(line[6:])
                                except json.JSONDecodeError:
                                    pass

    # ------------------------------------------------------------------
    # ScratchpadRuntime implementation
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self._post("/scratchpad/start", {
            "name": self.name,
            "coding_provider": self._coding_provider,
            "coding_model": self._coding_model,
            "coding_api_key": self._coding_api_key,
            "coding_base_url": self._coding_base_url,
        })

    async def reset(self) -> None:
        await self._post("/scratchpad/reset", {"name": self.name})
        self.cells.clear()
        self.compacted_cells = ()

    async def close(self) -> None:
        try:
            await self._post("/scratchpad/close", {"name": self.name})
        except Exception:
            pass  # Best effort on close

    async def cancel(self) -> None:
        """Ask the service to stop the running cell and restart the pad, and
        record the cell it stopped, as the local runtime does: its process
        is gone, so it kept none of its names, and the next cell runs in a
        new process. The running cell is taken before the request, so the
        stream it ends doesn't record it again, and recorded even when the
        request fails."""
        running, self._running_code = self._running_code, None
        try:
            await self._post("/scratchpad/cancel", {"name": self.name})
        finally:
            self._record_cell(
                Cell(
                    code=running if running is not None else "# (cancelled by user)",
                    stdout="",
                    stderr="",
                    error="Cancelled by user.",
                    description="Cancelled",
                    stopped_at=NOTHING_KEPT,
                    ended_process=True,
                )
            )

    async def install_packages(self, packages: list[str]) -> str:
        result = await self._post("/scratchpad/install", {
            "name": self.name,
            "packages": packages,
        })
        return result.get("result", "")

    async def execute_streaming(
        self,
        code: str,
        *,
        description: str = "",
        estimated_time: str = "",
        estimated_seconds: int = 0,
    ):
        """Execute code via SSE — yields progress strings then a final Cell."""
        trace = get_trace_context()
        body = {
            "name": self.name,
            "code": code,
            "description": description,
            "estimated_time": estimated_time,
            "estimated_seconds": estimated_seconds,
            # The service restores this context around execution. Keep it out
            # of code so the cell's source and notebook remain unchanged.
            "trace_context": json.loads(trace_context_to_json(trace)) if trace is not None else None,
        }

        self._running_code = code
        stopped_by = "The scratchpad service's stream ended with no result."
        try:
            async for event in self._sse("/scratchpad/execute-stream", body):
                event_type = event.get("type", "")

                if event_type == "progress":
                    yield event.get("message", "")

                elif event_type == "cell":
                    cell_data = event.get("cell", {})
                    cell = Cell(
                        code=cell_data.get("code", code),
                        stdout=cell_data.get("stdout", ""),
                        stderr=cell_data.get("stderr", ""),
                        error=cell_data.get("error"),
                        description=cell_data.get("description", description),
                        estimated_time=cell_data.get("estimated_time", estimated_time),
                        logs=cell_data.get("logs", ""),
                        stopped_at=stopped_line(cell_data.get("stopped_at")),
                        # The service says when it ended the pad's process, as a
                        # timeout kill does, so the next cell runs in a new one.
                        ended_process=cell_data.get("ended_process") is True,
                    )
                    self._running_code = None
                    cell = self._record_cell(cell)
                    yield cell

                elif event_type == "error":
                    error_msg = event.get("error", "Unknown remote error")
                    cell = Cell(
                        code=code,
                        stdout="",
                        stderr="",
                        error=error_msg,
                        description=description,
                        # The service sent no cell, so what it bound is unknown:
                        # none of it counts as bound. Nor can the check tell
                        # whether the service started a new process for the next
                        # cell, which runs the startup hooks a cell wrote, so it
                        # counts as one that ended its process.
                        stopped_at=NOTHING_KEPT,
                        ended_process=True,
                    )
                    self._running_code = None
                    cell = self._record_cell(cell)
                    yield cell
        except (asyncio.CancelledError, GeneratorExit, KeyboardInterrupt):
            stopped_by = "Cancelled."
            raise
        except Exception as exc:
            stopped_by = f"The scratchpad service's stream failed ({type(exc).__name__})."
            raise
        finally:
            # A stream that ends with no cell for `code` stopped it: a host
            # that cancels the turn's task, a close at a progress yield (as a
            # turn closed at its own yield leaves it), or a stream that fails.
            # Record it as `cancel()` does, unless a cell or `cancel()` already
            # did (each clears `_running_code` first).
            running, self._running_code = self._running_code, None
            if running is not None:
                self._record_cell(
                    Cell(
                        code=running,
                        stdout="",
                        stderr="",
                        error=stopped_by,
                        description=description,
                        estimated_time=estimated_time,
                        stopped_at=NOTHING_KEPT,
                        ended_process=True,
                    )
                )

    async def cleanup(self) -> None:
        try:
            await self._post("/scratchpad/cleanup", {"name": self.name})
        except Exception:
            pass


def remote_scratchpad_runtime_factory(
    *,
    name: str,
    coding_provider: str,
    coding_model: str,
    coding_api_key: str,
    coding_base_url: str,
    cells: list[Cell] | None,
    workspace_path: Path | None,
    session_id: str | None = None,
    endpoint_url: str = "",
    api_key: str = "",
) -> ScratchpadRuntime:
    """Factory that creates a RemoteScratchpadRuntime.

    The endpoint_url and api_key are injected via functools.partial
    when building the factory for a specific user.
    """
    return RemoteScratchpadRuntime(
        name=name,
        endpoint_url=endpoint_url,
        api_key=api_key,
        coding_provider=coding_provider,
        coding_model=coding_model,
        coding_api_key=coding_api_key,
        coding_base_url=coding_base_url,
        cells=cells,
        workspace_path=workspace_path,
    )
