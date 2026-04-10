import asyncio
import contextlib
import io
import json
import logging
import socket
import tarfile
from dataclasses import dataclass, field

import docker
from docker.errors import APIError, NotFound

from anton.core.backends.base import Cell, ScratchpadRuntime
from anton.core.backends.constants import _BOOT_SCRIPT_PATH, _LLM_PROVIDER_PKG_PATH
from anton.core.backends.utils import _compute_timeouts
from anton.core.backends.wire import (
    CELL_DELIM,
    PROGRESS_MARKER,
    RESULT_END,
    RESULT_START,
)
from anton.core.settings import CoreSettings

from .settings import DockerBackendSettings

logger = logging.getLogger(__name__)

backend_settings = DockerBackendSettings()
core_settings = CoreSettings()


def _make_docker_client() -> docker.DockerClient:
    return docker.from_env(version=backend_settings.docker_api_version)


@dataclass
class DockerScratchpadRuntime(ScratchpadRuntime):
    client: docker.DockerClient = field(default_factory=_make_docker_client)
    sock: socket.socket | None = None

    async def start(self) -> None:
        container = self._get_or_run_container()

        # Idempotency: if we already have an active exec socket, don't create a new one.
        if self.sock is not None:
            try:
                underlying = getattr(self.sock, "_sock", None)
                if underlying is not None and underlying.fileno() != -1:
                    return
            except Exception:
                # Treat any socket inspection failure as "not started" and re-create.
                pass
            with contextlib.suppress(Exception):
                self.sock.close()
            self.sock = None

        # Only copy files when creating a new exec session (not on every cell call).
        self._copy_files_to_container(container)

        exec_id = self.client.api.exec_create(
            container.id,
            cmd=["python", "/scratchpad_boot.py"],
            stdin=True,
            tty=True,
        )
        logger.info(f"Exec session created with ID '{exec_id}' for Docker scratchpad container '{self.name}'.")

        self.sock = self.client.api.exec_start(exec_id, socket=True)
        # Ensure recv/send can be integrated with asyncio without blocking the loop.
        # docker-py returns a wrapper; the underlying socket is at ._sock.
        self.sock._sock.setblocking(False)
        logger.info(f"Socket created for Docker scratchpad container '{self.name}'.")

    def _get_container(self) -> docker.models.containers.Container:
        logger.info(f"Getting Docker scratchpad container '{self.name}'.")
        container = self.client.containers.get(self.name)
        logger.info(f"Docker scratchpad container '{self.name}' found.")
        container.reload()
        return container

    def _get_or_run_container(self):
        logger.info(f"Getting or running Docker scratchpad container '{self.name}'.")
        try:
            container = self._get_container()
            logger.info(f"Docker scratchpad container '{self.name}' found.")
            if container.status != "running":
                container.start()
                container.reload()
            logger.info(f"Docker scratchpad container '{self.name}' started.")
            return container

        except NotFound:
            try:
                logger.info(f"Docker scratchpad container '{self.name}' not found, running new container.")
                return self.client.containers.run(
                    backend_settings.docker_image,
                    name=self.name,
                    detach=True,
                    stdin_open=True,
                    tty=True,
                    command=["sleep", "infinity"],
                    environment=self._extra_env,
                    network=backend_settings.docker_network,
                )
            except APIError as e:
                # In case of a rare race where something else created it after NotFound.
                if getattr(e, "status_code", None) == 409:
                    logger.info(f"Docker scratchpad container '{self.name}' found after race, starting container.")
                    container = self.client.containers.get(self.name)
                    container.reload()
                    if container.status != "running":
                        container.start()
                        container.reload()
                    return container
                logger.error(f"Error running Docker scratchpad container '{self.name}': {e}")
                raise

    def _copy_files_to_container(
        self,
        container: docker.models.containers.Container,
    ) -> None:
        # Copy the boot script
        logger.info(f"Copying boot script to Docker scratchpad container '{self.name}'.")
        data = io.BytesIO()

        with tarfile.open(fileobj=data, mode="w") as tar:
            tar.add(_BOOT_SCRIPT_PATH, arcname="scratchpad_boot.py")

        data.seek(0)
        container.put_archive("/", data.read())
        logger.info(f"Boot script copied to Docker scratchpad container '{self.name}'.")

        # Copy the llm provider directory
        # This is required for the get_llm() function to work
        logger.info(f"Copying llm provider files to Docker scratchpad container '{self.name}'.")
        data = io.BytesIO()

        with tarfile.open(fileobj=data, mode="w") as tar:
            tar.add(_LLM_PROVIDER_PKG_PATH, arcname="llm")
        data.seek(0)

        container.put_archive("/", data.read())
        logger.info(f"Llm provider files copied to Docker scratchpad container '{self.name}'.")

    async def reset(self) -> None:
        logger.info(f"Resetting Docker scratchpad container '{self.name}'.")
        # 1) Drop old exec session
        if self.sock is not None:
            with contextlib.suppress(Exception):
                self.sock.close()
            self.sock = None
            logger.info(f"Old exec session closed for Docker scratchpad container '{self.name}'.")

        container = self._get_or_run_container()

        # 2) Clear in-memory history
        # TODO: Cells will be cleared for the current turn, but re-introduced on the next.
        self.cells.clear()
        # Remove the session file
        # TODO: Re-introduce when session persistence is implemented.
        # if anton_settings.scratchpad_persist_session:
        #     container.exec_run(["rm", "-f", anton_settings.scratchpad_session_path])
        #     logger.info(f"Session file removed for Docker scratchpad container '{self.name}'.")
        # else:
        #     logger.info(f"Session file not removed for Docker scratchpad container '{self.name}'.")
        # logger.info(f"In-memory history cleared for Docker scratchpad container '{self.name}'.")

        # 3) Restart container (kills prior exec session)
        logger.info(f"Restarting Docker scratchpad container '{self.name}'.")
        await asyncio.to_thread(container.restart)
        await asyncio.to_thread(container.reload)

        # 4) Ensure latest boot script is present
        self._copy_files_to_container(container)

        # 5) Start a fresh boot exec + reconnect socket
        exec_id = self.client.api.exec_create(
            container.id,
            cmd=["python", "/scratchpad_boot.py"],
            stdin=True,
            tty=True,
        )
        logger.info(f"New exec session created with ID '{exec_id}' for Docker scratchpad container '{self.name}'.")
        self.sock = self.client.api.exec_start(exec_id, socket=True)
        self.sock._sock.setblocking(False)
        logger.info(f"Socket created for Docker scratchpad container '{self.name}'.")

    async def close(self, cleanup: bool = False) -> None:
        logger.info(f"Stopping Docker scratchpad container '{self.name}'.")
        # Drop any active exec session socket to avoid leaks and inconsistent state
        if self.sock is not None:
            with contextlib.suppress(Exception):
                self.sock.close()
            self.sock = None

        try:
            container = self.client.containers.get(self.name)
        except NotFound:
            logger.info(f"Docker scratchpad container '{self.name}' not found, skipping stop.")
            return

        await asyncio.to_thread(container.stop)
        await asyncio.to_thread(container.reload)
        logger.info(f"Docker scratchpad container '{self.name}' stopped.")
        # Cleanup is set to False by default to preserve the container for the next turn,
        # of the same conversation.
        if cleanup:
            await asyncio.to_thread(container.remove)
            logger.info(f"Docker scratchpad container '{self.name}' removed.")

    async def cancel(self) -> None:
        logger.info(f"Cancelling the current execution for Docker scratchpad container '{self.name}'.")
        # Similar to reset(), but without clearing in-memory history.
        # 1) Drop old exec session
        if self.sock is not None:
            with contextlib.suppress(Exception):
                self.sock.close()
            self.sock = None
            logger.info(f"Old exec session closed for Docker scratchpad container '{self.name}'.")

        # 2) Restart container (kills prior exec session)
        logger.info(f"Restarting Docker scratchpad container '{self.name}'.")
        container = self._get_or_run_container()
        await asyncio.to_thread(container.restart)
        await asyncio.to_thread(container.reload)

        # 3) Record the cancelled execution in a cell
        # TODO: This will only be recorded for the current turn, but lost on the next.
        self.cells.append(
            Cell(
                code="# (cancelled by user)",
                stdout="",
                stderr="",
                error="Cancelled by user.",
                description="Cancelled.",
            )
        )

        # 4) Ensure latest boot script is present
        self._copy_files_to_container(container)

        # 5) Start a fresh boot exec + reconnect socket
        exec_id = self.client.api.exec_create(
            container.id,
            cmd=["python", "/scratchpad_boot.py"],
            stdin=True,
            tty=True,
        )
        logger.info(f"New exec session created with ID '{exec_id}' for Docker scratchpad container '{self.name}'.")
        self.sock = self.client.api.exec_start(exec_id, socket=True)
        self.sock._sock.setblocking(False)
        logger.info(f"Socket created for Docker scratchpad container '{self.name}'.")

    async def install_packages(self, packages: list[str]) -> str:
        logger.info(f"Installing packages '{', '.join(packages)}' for Docker scratchpad container '{self.name}'.")
        container = self.client.containers.get(self.name)

        try:
            exec_result = await asyncio.wait_for(
                asyncio.to_thread(
                    container.exec_run,
                    ["pip", "install", *packages],
                ),
                timeout=300,
            )
        except asyncio.TimeoutError:
            return f"Timed out while installing packages: {', '.join(packages)}"

        exit_code = getattr(exec_result, "exit_code", None)
        if exit_code not in (0, None):
            output = getattr(exec_result, "output", b"")
            output_str = output.decode(errors="replace") if isinstance(output, bytes | bytearray) else str(output)

            logger.error(f"Error while installing packages: {', '.join(packages)}\n{output_str}")
            return f"Error while installing packages: {', '.join(packages)}\n{output_str}"

        logger.info(f"Packages '{', '.join(packages)}' installed for Docker scratchpad container '{self.name}'.")
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
        logger.info(f"Executing streaming code for Docker scratchpad container '{self.name}'.")
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

        payload = code + "\n" + CELL_DELIM + "\n"
        logger.info(f"Sending payload to Docker scratchpad container '{self.name}': {payload}.")

        loop = asyncio.get_running_loop()
        await loop.sock_sendall(self.sock._sock, payload.encode())

        total_timeout, inactivity_timeout = _compute_timeouts(core_settings, estimated_seconds)

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
            logger.info(f"Timeout error for Docker scratchpad container '{self.name}', restarting container.")
            try:
                container = self.client.containers.get(self.name)
                await asyncio.to_thread(container.restart)
                logger.info(f"Container restarted for Docker scratchpad container '{self.name}'.")
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
            logger.info(f"No result data for Docker scratchpad container '{self.name}', returning empty result.")
            result_data = {"stdout": "", "stderr": "", "error": "Process exited unexpectedly."}

        for pkg in result_data.get("auto_installed") or []:
            self._installed_packages.add(str(pkg).lower())
            logger.info(f"Package '{pkg}' auto-installed for Docker scratchpad container '{self.name}'.")

        cell = Cell(
            code=code,
            stdout=result_data.get("stdout", ""),
            stderr=result_data.get("stderr", ""),
            error=result_data.get("error"),
            description=description,
            estimated_time=estimated_time,
            logs=result_data.get("logs", ""),
        )
        logger.info(f"Cell created for Docker scratchpad container '{self.name}': {cell}.")
        self.cells.append(cell)
        yield cell

    async def _read_result(
        self,
        *,
        total_timeout: float = core_settings.cell_timeout_default,
        inactivity_timeout: float = core_settings.cell_inactivity_timeout,
    ):
        """Async generator that reads lines from stdout until result delimiters.

        Yields:
            str — progress messages (lines starting with _PROGRESS_MARKER)
            dict — the final JSON result (always the last item)

        Raises asyncio.TimeoutError with a descriptive message.

        After a progress() call is received, the inactivity window is extended
        to core_settings.cell_inactivity_after_progress (60s) so that long-running work
        that signals liveness isn't killed prematurely.
        """
        import re as _re
        import time as _time

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
                    raise asyncio.TimeoutError(f"Cell timed out after {total_timeout:.0f}s total") from None
                raise asyncio.TimeoutError(
                    f"Cell killed after {current_inactivity:.0f}s of inactivity (no output or progress() calls)"
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

                if cleaned.startswith(PROGRESS_MARKER):
                    current_inactivity = max(current_inactivity, core_settings.cell_inactivity_after_progress)
                    message = cleaned[len(PROGRESS_MARKER) :].strip()
                    yield message
                    continue

                # Markers may have leading framing/echo bytes, so treat as substring.
                if RESULT_START in cleaned:
                    in_result = True
                    continue
                if RESULT_END in cleaned:
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
