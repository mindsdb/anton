import asyncio
from dataclasses import dataclass
import docker
import io
import json
import socket
import tarfile

from .base import (
    Cell,
    ScratchpadRuntime,
    _BOOT_SCRIPT_PATH,
    _CELL_DELIM,
    _CELL_TIMEOUT_DEFAULT,
    _CELL_INACTIVITY_TIMEOUT,
    _CELL_INACTIVITY_AFTER_PROGRESS,
    _PROGRESS_MARKER,
    _RESULT_START,
    _RESULT_END,
)


@dataclass
class DockerScratchpadRuntime(ScratchpadRuntime):
    client: docker.DockerClient = docker.from_env()
    container_prefix: str = "anton-scratchpad-"
    sock: socket.socket | None = None
    
    def __post_init__(self) -> None:
        self.container_name = f"{self.container_prefix}{self.name}"

    async def start(self) -> None:
        print(f"Starting Docker scratchpad runtime for {self.name}")
        container = self.client.containers.run(
            "python:3.12-slim",
            name=self.container_name,
            detach=True,
            stdin_open=True,
            tty=True,
            command=["sleep", "infinity"],  # keep container alive while we copy files
        )

        self._copy_file_to_container(container)

        exec_id = self.client.api.exec_create(
            container.id,
            cmd=["python", "/scratchpad_boot.py"],
            stdin=True,
            tty=True,
        )

        self.sock = self.client.api.exec_start(exec_id, socket=True)
        # Ensure recv/send can be integrated with asyncio without blocking the loop.
        # docker-py returns a wrapper; the underlying socket is at ._sock.
        self.sock._sock.setblocking(False)

    def _copy_file_to_container(
        self,
        container: docker.models.containers.Container,
    ) -> None:
        data = io.BytesIO()

        with tarfile.open(fileobj=data, mode="w") as tar:
            tar.add(_BOOT_SCRIPT_PATH, arcname="scratchpad_boot.py")

        data.seek(0)
        container.put_archive("/", data.read())

    async def reset(self) -> None:
        self.client.containers.get(self.container_name).restart()

    async def close(self, cleanup: bool = True) -> None:
        self.client.containers.get(self.container_name).stop()
        if cleanup:
            self.client.containers.get(self.container_name).remove()

    async def cancel(self) -> None:
        self.client.containers.get(self.container_name).kill()

    async def install_packages(self, packages: list[str]) -> str:
        self.client.containers.get(self.container_name).exec_run(
            ["pip", "install", *packages]
        )
        return f"Packages {', '.join(packages)} installed."

    async def execute_streaming(
        self,
        code: str,
        *,
        description: str = "",
        estimated_time: str = "",
        estimated_seconds: int = 0,
    ):
        """Async generator that sends code and yields progress strings and a final Cell."""
        if self.sock is None:
            yield Cell(
                code=code,
                stdout="",
                stderr="",
                error="Docker scratchpad is not running. Use reset to restart.",
                description=description,
                estimated_time=estimated_time,
            )
            return

        payload = code + "\n" + _CELL_DELIM + "\n"

        loop = asyncio.get_running_loop()
        await loop.sock_sendall(self.sock._sock, payload.encode())

        total_timeout, inactivity_timeout = self._compute_timeouts(estimated_seconds)

        try:
            result_data: dict | None = None
            async for item in self._read_result(
                total_timeout=total_timeout,
                inactivity_timeout=inactivity_timeout,
            ):
                if isinstance(item, str):
                    yield item
                else:
                    result_data = item
        except asyncio.TimeoutError as exc:
            # Restart container so the scratchpad is usable again.
            try:
                container = self.client.containers.get(self.container_name)
                await asyncio.to_thread(container.restart)
            except Exception:
                pass
            cell = Cell(
                code=code,
                stdout="",
                stderr="",
                error=f"{exc}. Runtime restarted — state lost. Use reset if it doesn't recover.",
                description=description,
                estimated_time=estimated_time,
            )
            self.cells.append(cell)
            yield cell
            return

        if result_data is None:
            result_data = {"stdout": "", "stderr": "", "error": "Process exited unexpectedly."}

        for pkg in result_data.get("auto_installed") or []:
            self._installed_packages.add(str(pkg).lower())

        cell = Cell(
            code=code,
            stdout=result_data.get("stdout", ""),
            stderr=result_data.get("stderr", ""),
            error=result_data.get("error"),
            description=description,
            estimated_time=estimated_time,
            logs=result_data.get("logs", ""),
        )
        self.cells.append(cell)
        yield cell

    async def _read_result(
        self,
        *,
        total_timeout: float = _CELL_TIMEOUT_DEFAULT,
        inactivity_timeout: float = _CELL_INACTIVITY_TIMEOUT,
    ):
        """Async generator that reads lines from stdout until result delimiters.

        Yields:
            str — progress messages (lines starting with _PROGRESS_MARKER)
            dict — the final JSON result (always the last item)

        Raises asyncio.TimeoutError with a descriptive message.

        After a progress() call is received, the inactivity window is extended
        to _CELL_INACTIVITY_AFTER_PROGRESS (60s) so that long-running work
        that signals liveness isn't killed prematurely.
        """
        import time as _time
        import re as _re

        if self.sock is None:
            yield {"stdout": "", "stderr": "", "error": "Docker scratchpad is not running."}
            return

        lines: list[str] = []
        in_result = False
        start = _time.monotonic()
        current_inactivity = inactivity_timeout

        loop = asyncio.get_running_loop()
        buffer = ""

        def _strip_ansi(s: str) -> str:
            # Best-effort: tty mode can include control sequences.
            return _re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s)

        def _extract_json_candidate(s: str) -> str:
            # Docker streams (or tty echo) can prepend garbage; locate JSON start.
            s = s.strip()
            for ch in ("{", "["):
                idx = s.find(ch)
                if idx != -1:
                    return s[idx:]
            return s

        while True:
            elapsed = _time.monotonic() - start
            remaining_total = total_timeout - elapsed
            if remaining_total <= 0:
                raise asyncio.TimeoutError(f"Cell timed out after {total_timeout:.0f}s total")

            chunk_timeout = min(current_inactivity, remaining_total)
            try:
                raw = await asyncio.wait_for(
                    loop.sock_recv(self.sock._sock, 65536),
                    timeout=chunk_timeout,
                )
            except asyncio.TimeoutError:
                elapsed_now = _time.monotonic() - start
                if elapsed_now >= total_timeout - 0.5:
                    raise asyncio.TimeoutError(
                        f"Cell timed out after {total_timeout:.0f}s total"
                    ) from None
                raise asyncio.TimeoutError(
                    f"Cell killed after {current_inactivity:.0f}s of inactivity "
                    f"(no output or progress() calls)"
                ) from None

            if not raw:
                yield {"stdout": "", "stderr": "", "error": "Process exited unexpectedly."}
                return

            buffer += raw.decode(errors="replace")

            while True:
                if "\n" not in buffer:
                    break
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                cleaned = _strip_ansi(line)

                if cleaned.startswith(_PROGRESS_MARKER):
                    current_inactivity = max(current_inactivity, _CELL_INACTIVITY_AFTER_PROGRESS)
                    message = cleaned[len(_PROGRESS_MARKER):].strip()
                    yield message
                    continue

                # Markers may have leading framing/echo bytes, so treat as substring.
                if _RESULT_START in cleaned:
                    in_result = True
                    continue
                if _RESULT_END in cleaned:
                    if not in_result:
                        # Ignore stray marker before a result block begins.
                        continue
                    if not lines:
                        yield {
                            "stdout": "",
                            "stderr": "",
                            "error": "Malformed result from Docker runtime (empty JSON payload).",
                        }
                        return

                    # Usually a single json.dumps() line, but tolerate extra noise.
                    parsed: dict | None = None
                    for candidate in reversed(lines):
                        cand = _extract_json_candidate(candidate)
                        try:
                            obj = json.loads(cand)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict):
                            parsed = obj
                            break
                    if parsed is None:
                        blob = _extract_json_candidate("\n".join(lines))
                        try:
                            obj = json.loads(blob)
                            if isinstance(obj, dict):
                                parsed = obj
                        except json.JSONDecodeError as exc:
                            yield {
                                "stdout": "",
                                "stderr": "",
                                "error": f"Failed to decode Docker result JSON: {exc}",
                            }
                            return

                    yield parsed or {
                        "stdout": "",
                        "stderr": "",
                        "error": "Failed to decode Docker result JSON.",
                    }
                    return
                if in_result:
                    # Keep original line (not cleaned) so JSON isn't altered, but
                    # strip ANSI in case it's injected into the payload line.
                    lines.append(_strip_ansi(line))