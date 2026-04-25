"""RemoteScratchpadRuntime — HTTP-based scratchpad backend.

Connects to the scratchpad web service running on a user's
Lightsail instance (sp_{hash}.4nton.ai). Implements the same
ScratchpadRuntime ABC as LocalScratchpadRuntime but delegates
all execution to the remote service via REST + SSE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator

from anton.core.backends.base import Cell, ScratchpadRuntime


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
        import aiohttp

        url = f"{self._endpoint_url}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=body or {}, headers=self._headers(), timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"Remote scratchpad error ({resp.status}): {text}")
                return await resp.json()

    async def _get(self, path: str, params: dict | None = None) -> dict:
        """GET from the remote service and return parsed JSON."""
        import aiohttp

        url = f"{self._endpoint_url}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, params=params, headers=self._headers(), timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"Remote scratchpad error ({resp.status}): {text}")
                return await resp.json()

    async def _sse(self, path: str, body: dict) -> AsyncIterator[dict]:
        """POST to an SSE endpoint and yield parsed events."""
        import aiohttp

        url = f"{self._endpoint_url}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=body, headers=self._headers(), timeout=aiohttp.ClientTimeout(total=600)
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"Remote scratchpad error ({resp.status}): {text}")

                buffer = ""
                async for chunk in resp.content:
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

    async def close(self) -> None:
        try:
            await self._post("/scratchpad/close", {"name": self.name})
        except Exception:
            pass  # Best effort on close

    async def cancel(self) -> None:
        await self._post("/scratchpad/cancel", {"name": self.name})

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
        body = {
            "name": self.name,
            "code": code,
            "description": description,
            "estimated_time": estimated_time,
            "estimated_seconds": estimated_seconds,
        }

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
                )
                self.cells.append(cell)
                yield cell

            elif event_type == "error":
                error_msg = event.get("error", "Unknown remote error")
                cell = Cell(
                    code=code,
                    stdout="",
                    stderr="",
                    error=error_msg,
                    description=description,
                )
                self.cells.append(cell)
                yield cell

    async def cleanup(self) -> None:
        try:
            await self._post("/scratchpad/cleanup", {"name": self.name})
        except Exception:
            pass


class RemoteLightsailScratchpadRuntime(RemoteScratchpadRuntime):
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
            endpoint_url=endpoint_url,
            api_key=api_key,
            coding_provider=coding_provider,
            coding_model=coding_model,
            coding_api_key=coding_api_key,
            coding_base_url=coding_base_url,
            cells=cells,
            workspace_path=workspace_path,
        )
        self._endpoint_url = self._resolve_endpoint(endpoint_url.rstrip("/"))

    async def _resolve_endpoint(self, endpoint_url: str) -> str:
        """Resolve the Cloudflare endpoint to a direct IP endpoint.

        Calls /resolve on the Cloudflare Worker which returns the instance's
        direct IP. Caches the result for subsequent calls.
        """
        import aiohttp

        url = f"{endpoint_url}/resolve"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=self._headers(), timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"Failed to resolve remote scratchpad ({resp.status}): {text}")
                data = await resp.json()

        endpoint = data.get("endpoint", "")
        if not endpoint:
            raise RuntimeError(f"No endpoint returned from /resolve: {data}")

        return endpoint.rstrip("/")


def remote_scratchpad_runtime_factory(
    *,
    name: str,
    coding_provider: str,
    coding_model: str,
    coding_api_key: str,
    coding_base_url: str,
    cells: list[Cell] | None,
    workspace_path: Path | None,
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


def remote_lightsail_scratchpad_runtime_factory(
    *,
    name: str,
    coding_provider: str,
    coding_model: str,
    coding_api_key: str,
    coding_base_url: str,
    cells: list[Cell] | None,
    workspace_path: Path | None,
    endpoint_url: str = "",
    api_key: str = "",
) -> ScratchpadRuntime:
    return RemoteLightsailScratchpadRuntime(
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
