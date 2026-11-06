"""HTTP client for the browser API the tool drives.

Two hosts serve the same routes. The MindsHub browser router checks the
caller's MindsHub credential and forwards to that person's own browser. The
Cowork desktop app's own browser listens on 127.0.0.1 and checks the key it
made for this launch. Every call carries ``Authorization: Bearer
<credential>``, and every answer is read into the models in ``models.py``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar
from urllib.parse import quote

import httpx2 as httpx
from pydantic import BaseModel, ValidationError

from anton.core.browser.models import (
    ActionResult,
    Control,
    ErrorBody,
    Page,
    SavedTab,
    SavedTabAnswer,
    SavedTabsAnswer,
    SessionInfo,
    Site,
    SiteAnswer,
    SitesAnswer,
    Tab,
    TabAnswer,
    TabsAnswer,
)

#: Page loads and clicks that navigate can take a while; the hosts cap their
#: own navigation at 30 s, so leave room for that plus the router hop.
REQUEST_TIMEOUT_S = 45.0
#: The router starts the person's browser when the host scopes the turn, and
#: answers its own typed 504 when one request passes its deadline of about
#: 105 s. A router client's calls wait this long until the router has answered
#: once: past that deadline, so the router's answer arrives first, and under
#: the edge proxy's 125 s limit, so its HTML error page never does.
FIRST_CALL_TIMEOUT_S = 115.0

_Model = TypeVar("_Model", bound=BaseModel)


@dataclass
class BrowserError(Exception):
    """A non-2xx answer from the browser, an answer it can't read, or none at all."""

    status: int
    code: str
    message: str
    #: Set on ``approval_required``: the held step's id, for the person's
    #: approval and the agent's one resend.
    pending_id: str | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def _error(response: httpx.Response) -> BrowserError:
    """The host's ``{"error", "message"}``, or the status and the start of the body."""
    try:
        body = ErrorBody.model_validate_json(response.content)
    except ValidationError:
        return BrowserError(response.status_code, "http_error", response.text[:300])
    return BrowserError(response.status_code, body.error, body.message, body.pending_id)


class BrowserClient:
    def __init__(
        self,
        *,
        base_url: str,
        credential: Callable[[], Awaitable[str]],
        trust_env: bool,
        first_call_timeout_s: float,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        """``trust_env`` off keeps system and environment proxies away from the
        in-app browser's launch key. ``first_call_timeout_s`` bounds every call
        until the host has answered once, then ``REQUEST_TIMEOUT_S`` does."""
        self._base = base_url.rstrip("/")
        self._credential = credential
        self._http = http or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, trust_env=trust_env)
        self._owns_http = http is None
        self._first_call_timeout_s = first_call_timeout_s
        self._answered = False

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def _request(self, method: str, path: str, *, json: dict | None = None, params: dict | None = None) -> httpx.Response:
        token = await self._credential()
        if not token:
            raise BrowserError(401, "no_credential", "no credential is available for the browser")
        timeout = REQUEST_TIMEOUT_S if self._answered else self._first_call_timeout_s
        try:
            response = await self._http.request(
                method,
                f"{self._base}{path}",
                json=json,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise BrowserError(0, "unreachable", f"the browser did not answer ({type(exc).__name__})") from exc
        self._answered = True
        if response.status_code >= 400:
            raise _error(response)
        return response

    async def _read(
        self, model: type[_Model], method: str, path: str, *, json: dict | None = None, params: dict | None = None,
    ) -> _Model:
        """One call's answer as ``model``. An answer that isn't that shape is the
        host's fault, not the agent's, so it raises ``bad_response``."""
        response = await self._request(method, path, json=json, params=params)
        try:
            return model.model_validate_json(response.content)
        except ValidationError as exc:
            raise BrowserError(
                response.status_code, "bad_response",
                f"the browser's answer to {method} {path} is not the expected shape ({exc.error_count()} problems)",
            ) from exc

    async def open_session(self, *, session_id: str, url: str | None = None) -> None:
        body: dict = {"session_id": session_id}
        if url:
            body["url"] = url
        await self._request("POST", "/sessions", json=body)

    async def page(self, *, session_id: str, max_text: int = 3000) -> Page:
        return await self._read(Page, "GET", f"/sessions/{quote(session_id)}/page", params={"max_text": max_text})

    async def act(self, *, session_id: str, step: dict) -> ActionResult:
        return await self._read(ActionResult, "POST", f"/sessions/{quote(session_id)}/action", json=step)

    async def screenshot(self, *, session_id: str, annotate: bool = True) -> tuple[bytes, tuple[int, int] | None]:
        """The JPEG and its (width, height), when the host says: the space the
        model's x/y coordinates are in."""
        params = {"annotate": "true" if annotate else "false", "format": "jpeg", "quality": 70}
        response = await self._request("GET", f"/sessions/{quote(session_id)}/screenshot", params=params)
        try:
            size = (int(response.headers["x-image-width"]), int(response.headers["x-image-height"]))
        except (KeyError, ValueError):
            size = None
        return response.content, size

    async def sites(self, *, session_id: str) -> list[Site]:
        return (await self._read(SitesAnswer, "GET", f"/sessions/{quote(session_id)}/sites")).sites

    async def note_site(self, *, session_id: str, site: str, account_hint: str = "") -> Site:
        body = {"site": site, "account_hint": account_hint}
        return (await self._read(SiteAnswer, "POST", f"/sessions/{quote(session_id)}/sites", json=body)).site

    async def session_info(self, *, session_id: str) -> SessionInfo:
        """The session's control record and address. Answers even while the person holds it."""
        return await self._read(SessionInfo, "GET", f"/sessions/{quote(session_id)}")

    async def control(self, *, session_id: str) -> Control:
        """Who holds the browser. Answers even while the person holds it."""
        return (await self.session_info(session_id=session_id)).control

    async def hand_to_person(self, *, session_id: str, reason: str) -> Control:
        """Give the browser to the person. While they hold it the host refuses
        the agent's actions and page reads, and shows them it's their turn.
        Only the person hands it back, with Done in the browser; the agent
        never asks for it back."""
        body = {"holder": "human", "reason": reason[:300]}
        return await self._read(Control, "POST", f"/sessions/{quote(session_id)}/control", json=body)

    async def tabs(self, *, session_id: str) -> list[Tab]:
        """The open tabs; the active one is the tab actions, page reads and the viewer use."""
        return (await self._read(TabsAnswer, "GET", f"/sessions/{quote(session_id)}/tabs")).tabs

    async def new_tab(self, *, session_id: str, url: str | None = None) -> Tab:
        body = {"url": url} if url else {}
        return (await self._read(TabAnswer, "POST", f"/sessions/{quote(session_id)}/tabs", json=body)).tab

    async def activate_tab(self, *, session_id: str, tab_id: str) -> Tab:
        path = f"/sessions/{quote(session_id)}/tabs/{quote(tab_id, safe='')}/activate"
        return (await self._read(TabAnswer, "POST", path)).tab

    async def close_tab(self, *, session_id: str, tab_id: str) -> None:
        """Closing the last tab leaves a blank one."""
        await self._request("DELETE", f"/sessions/{quote(session_id)}/tabs/{quote(tab_id, safe='')}")

    async def saved_tabs(self, *, project: str) -> list[SavedTab]:
        """The project's saved tabs, newest first."""
        return (await self._read(SavedTabsAnswer, "GET", "/saved-tabs", params={"project": project})).tabs

    async def save_tab(self, *, project: str, name: str, url: str, notes: str = "") -> SavedTab:
        """Upserts by name within the project."""
        body = {"project": project, "name": name, "url": url, "notes": notes}
        return (await self._read(SavedTabAnswer, "POST", "/saved-tabs", json=body)).tab

    async def delete_saved_tab(self, *, project: str, name: str) -> None:
        await self._request("DELETE", "/saved-tabs", params={"project": project, "name": name})

    async def close_session(self, *, session_id: str) -> None:
        await self._request("DELETE", f"/sessions/{quote(session_id)}")
