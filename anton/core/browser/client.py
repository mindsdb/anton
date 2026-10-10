"""HTTP client for the browser instance's API.

Every call carries ``Authorization: Bearer <MindsHub credential>``; the
Cloudflare worker checks it belongs to the instance's owner before anything
reaches the instance (mindshub_services ENG-3295).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import quote

import httpx2 as httpx

#: Page loads and clicks that navigate can take a while; the service caps its
#: own navigation at 30s, so leave room for that plus the worker hop.
REQUEST_TIMEOUT_S = 45.0


@dataclass
class BrowserError(Exception):
    """A non-2xx answer from the instance, or no answer at all."""

    status: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class BrowserClient:
    def __init__(
        self,
        base_url: str,
        credential: Callable[[], Awaitable[str]],
        *,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._credential = credential
        self._http = http or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
        self._owns_http = http is None

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def _request(self, method: str, path: str, *, json: dict | None = None, params: dict | None = None):
        token = await self._credential()
        if not token:
            raise BrowserError(401, "no_credential", "no MindsHub credential is available for the browser")
        try:
            response = await self._http.request(
                method,
                f"{self._base}{path}",
                json=json,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as exc:
            raise BrowserError(0, "unreachable", f"the browser instance did not answer ({type(exc).__name__})") from exc
        if response.status_code >= 400:
            code, message = "http_error", response.text[:300]
            try:
                body = response.json()
                code = str(body.get("error") or code)
                message = str(body.get("message") or body.get("hint") or message)
            except Exception:
                pass
            raise BrowserError(response.status_code, code, message)
        return response

    async def open_session(self, session_id: str, url: str | None = None) -> dict:
        body: dict = {"session_id": session_id}
        if url:
            body["url"] = url
        return (await self._request("POST", "/sessions", json=body)).json()

    async def page(self, session_id: str, *, max_text: int = 3000) -> dict:
        return (await self._request("GET", f"/sessions/{quote(session_id)}/page", params={"max_text": max_text})).json()

    async def act(self, session_id: str, action: dict) -> dict:
        return (await self._request("POST", f"/sessions/{quote(session_id)}/action", json=action)).json()

    async def screenshot(self, session_id: str, *, annotate: bool = True) -> bytes:
        params = {"annotate": "true" if annotate else "false", "format": "jpeg", "quality": 70}
        return (await self._request("GET", f"/sessions/{quote(session_id)}/screenshot", params=params)).content

    async def embed(self, session_id: str) -> dict:
        """A viewer URL another origin can frame (``POST /_embed`` on the worker)."""
        return (await self._request("POST", "/_embed", json={"session_id": session_id})).json()

    async def sites(self, session_id: str) -> list[dict]:
        return (await self._request("GET", f"/sessions/{quote(session_id)}/sites")).json().get("sites", [])

    async def note_site(self, session_id: str, site: str, account_hint: str = "") -> dict:
        body = {"site": site, "account_hint": account_hint}
        return (await self._request("POST", f"/sessions/{quote(session_id)}/sites", json=body)).json().get("site", {})

    async def close_session(self, session_id: str) -> None:
        await self._request("DELETE", f"/sessions/{quote(session_id)}")
