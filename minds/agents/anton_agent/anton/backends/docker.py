import asyncio
import contextlib
import io
import json
import socket
import tarfile
from dataclasses import dataclass, field
from uuid import UUID

import docker
from docker.errors import APIError, NotFound

from minds.common.logger import get_logger

from ...settings import AntonAgentSettings
from .base import (
    _BOOT_SCRIPT_PATH,
    _CELL_DELIM,
    _CELL_INACTIVITY_AFTER_PROGRESS,
    _CELL_INACTIVITY_TIMEOUT,
    _CELL_TIMEOUT_DEFAULT,
    _LLM_PROVIDER_PKG_PATH,
    _PROGRESS_MARKER,
    _RESULT_END,
    _RESULT_START,
    Cell,
    ScratchpadRuntime,
)

logger = get_logger(__name__)

anton_settings = AntonAgentSettings()


def _make_docker_client() -> docker.DockerClient:
    return docker.from_env(version=anton_settings.docker_api_version)


@dataclass
class DockerScratchpadRuntime(ScratchpadRuntime):
    client: docker.DockerClient = field(default_factory=_make_docker_client)
    sock: socket.socket | None = None

    async def start(self) -> None:
        container = self._get_or_run_container()

        # (Re)copy the bootstrap each time (safe + keeps it up to date)
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
                    anton_settings.docker_image,
                    name=self.name,
                    detach=True,
                    stdin_open=True,
                    tty=True,
                    command=["sleep", "infinity"],
                    environment=self._extra_env,
                    network=anton_settings.docker_network,
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

        # Clear in-memory history
        self.cells.clear()
        logger.info(f"In-memory history cleared for Docker scratchpad container '{self.name}'.")

        # 2) Record the cancelled execution in a cell
        self.cells.append(
            Cell(
                code="# (cancelled by user)",
                stdout="",
                stderr="",
                error="Cancelled by user.",
                description="Cancelled.",
            )
        )

        # 3) Restart container (kills prior exec session)
        logger.info(f"Restarting Docker scratchpad container '{self.name}'.")
        container = self._get_or_run_container()
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

        container = self.client.containers.get(self.name)
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

        # 3) Ensure latest boot script is present
        self._copy_files_to_container(container)

        # 4) Start a fresh boot exec + reconnect socket
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

        payload = code + "\n" + _CELL_DELIM + "\n"
        logger.info(f"Sending payload to Docker scratchpad container '{self.name}': {payload}.")

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

                if cleaned.startswith(_PROGRESS_MARKER):
                    current_inactivity = max(current_inactivity, _CELL_INACTIVITY_AFTER_PROGRESS)
                    message = cleaned[len(_PROGRESS_MARKER) :].strip()
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

    async def report_exists(
        self,
        organization_id: UUID,
        user_id: UUID,
        conversation_id: UUID,
        message_id: UUID,
    ) -> bool:
        logger.info(f"Checking if report exists for Docker scratchpad container '{self.name}'.")
        report_path = (
            f"{anton_settings.root_workspace_dir}/{str(organization_id)}/{str(user_id)}/"
            f"{anton_settings.output_dir}/{conversation_id}/{message_id}/"
            f"{anton_settings.output_file_name}"
        )
        logger.info(f"Report path for Docker scratchpad container '{self.name}': {report_path}.")

        try:
            container = self._get_container()
        except NotFound:
            logger.info(f"Container not found for Docker scratchpad container '{self.name}', returning False.")
            # We are not too concerned about the container not being found here,
            # this is possible for past conversations that did not use Anton.
            # We just want to indicate whether or not the report exists.
            return False

        try:
            # Another option is to use exec_run(["test", "-f", report_path])
            # this is more robust, but it requires the container to be running.
            _, _ = container.get_archive(report_path)
        except NotFound:
            logger.info(f"Report not found for Docker scratchpad container '{self.name}', returning False.")
            return False

        logger.info(f"Report found for Docker scratchpad container '{self.name}', returning True.")
        return True

    async def get_report(
        self,
        organization_id: UUID,
        user_id: UUID,
        conversation_id: UUID,
        message_id: UUID,
    ) -> str:
        logger.info(f"Getting report for Docker scratchpad container '{self.name}'.")
        report_path = (
            f"{anton_settings.root_workspace_dir}/{str(organization_id)}/{str(user_id)}/"
            f"{anton_settings.output_dir}/{conversation_id}/{message_id}/"
            f"{anton_settings.output_file_name}"
        )
        logger.info(f"Report path for Docker scratchpad container '{self.name}': {report_path}.")

        try:
            container = self._get_container()
        except NotFound as err:
            logger.error(
                f"Container not found for Docker scratchpad container '{self.name}', raising FileNotFoundError."
            )
            raise FileNotFoundError("A scratchpad container is not available for this message") from err

        try:
            bits, _ = container.get_archive(report_path)
        except NotFound as err:
            logger.error(f"Report not found for Docker scratchpad container '{self.name}', raising FileNotFoundError.")
            raise FileNotFoundError("A report is not available for this message") from err

        file_obj = io.BytesIO(b"".join(bits))

        with tarfile.open(fileobj=file_obj) as tar:
            member = tar.getmembers()[0]
            html = tar.extractfile(member).read().decode()

        logger.info(f"Report for Docker scratchpad container '{self.name}': {html}.")
        return html
