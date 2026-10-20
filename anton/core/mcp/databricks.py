"""Databricks helpers for the Genie MCP connection.

`canonical_workspace_host` is the one gate between a workspace address a
user typed and any URL built from it, so a token is only ever sent to a
Databricks-owned host. `genie_mcp_url` builds the Genie MCP address from it.
`mint_token` exchanges a service principal's client credentials for an access
token. `wait_for_genie_answer` polls Genie until an answer is done, since
`genie_ask` only starts one.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx2

if TYPE_CHECKING:
    from anton.core.mcp.client import McpSession

_HOST_SUFFIXES = (".cloud.databricks.com", ".azuredatabricks.net", ".gcp.databricks.com")
_LABEL = re.compile(r"^[a-z0-9-]{1,63}$")
_QUERY = re.compile(r"^o=\d+$")
_SPACE_ID = re.compile(r"[0-9a-f]{32}")

GENIE_ONE_PATH = "/ai-gateway/mcp-services/system.ai.genie_one_mcp"
PER_SPACE_PATH = "/api/2.0/mcp/genie/{space_id}"
TOKEN_PATH = "/oidc/v1/token"
# The per-space MCP server only accepts all-apis, so it is the one scope requested.
TOKEN_SCOPE = "all-apis"
_TOKEN_TIMEOUT_SECONDS = 10.0
_CREDENTIAL_ERRORS = ("invalid_client", "unauthorized_client")
# Genie's own tool description asks for 2 to 5 s between polls; answers took 57 to 81 s live.
GENIE_POLL_INTERVAL_SECONDS = 3.0
GENIE_WAIT_SECONDS = 240.0
GENIE_IN_PROGRESS = "in_progress"


class InvalidWorkspaceHost(ValueError):
    """The value is not a Databricks workspace address."""


class InvalidGenieSpace(ValueError):
    """The value is not a Genie space id."""


class GenieCredentialsRefused(RuntimeError):
    """Databricks did not accept the service principal's client ID or secret."""


class GenieRequestRefused(RuntimeError):
    """Databricks answered the token request with something other than a token."""


class GenieUnavailable(RuntimeError):
    """The workspace could not be reached or answered with a server error."""


class GenieAnswerFailed(RuntimeError):
    """Genie reported an error, or no status, while it was polled for an answer."""


def canonical_workspace_host(value: str) -> str:
    """Reduce a workspace address to its bare lowercase host.

    Accepts a bare host or an `https://` URL whose path is empty or `/` and
    whose query is empty or `o=<digits>`. Raises `InvalidWorkspaceHost`,
    naming the broken rule and never more than the host, for anything else:
    other schemes, ports, user info, fragments, trailing dots, IP literals,
    and hosts outside the Databricks cloud suffixes. No side effects.
    """
    text = value.strip()
    if not text:
        raise InvalidWorkspaceHost("The workspace address is empty.")
    if "://" not in text:
        text = f"https://{text}"
    parts = urlsplit(text)
    if parts.scheme != "https":
        raise InvalidWorkspaceHost("The workspace address must use https.")
    if parts.fragment or "#" in text:
        raise InvalidWorkspaceHost("The workspace address must not have a fragment.")
    if parts.path not in ("", "/"):
        raise InvalidWorkspaceHost("The workspace address must not have a path.")
    if parts.query and not _QUERY.match(parts.query):
        raise InvalidWorkspaceHost("The workspace address may only carry an o=<digits> query.")
    host = parts.netloc.lower()
    if "@" in host or ":" in host:
        raise InvalidWorkspaceHost("The workspace address must not have user info or a port.")
    suffix = next((s for s in _HOST_SUFFIXES if host.endswith(s)), None)
    if suffix is None or len(host) == len(suffix):
        raise InvalidWorkspaceHost("The workspace host must be a Databricks workspace host.")
    if not all(_LABEL.match(label) for label in host.split(".")):
        raise InvalidWorkspaceHost("The workspace host has an invalid label.")
    return host


def genie_mcp_url(host: str, space_id: str | None = None) -> str:
    """Build the Genie MCP server address for a workspace.

    `host` goes through `canonical_workspace_host` first. With no `space_id`
    the result is the Genie One address; otherwise it is the per-space
    address. The id is stripped, then must be 32 lowercase hex characters,
    so nothing path-like can reach the URL. Raises `InvalidWorkspaceHost` or
    `InvalidGenieSpace` (a fixed message that never echoes the id). No side
    effects.
    """
    base = f"https://{canonical_workspace_host(host)}"
    if space_id is None:
        return base + GENIE_ONE_PATH
    cleaned = space_id.strip()
    if not _SPACE_ID.fullmatch(cleaned):
        raise InvalidGenieSpace("A Genie space id is 32 lowercase hex characters.")
    return base + PER_SPACE_PATH.format(space_id=cleaned)


def _error_code(response: httpx2.Response) -> str | None:
    """Read the OAuth `error` field from a response body, or None.

    Returns None for a body that is not a JSON object or has no string
    `error`. Never returns or logs any other body text. No side effects.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"]
    return None


async def mint_token(
    host: str,
    client_id: str,
    client_secret: str,
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> str:
    """Exchange a service principal's client credentials for an access token.

    `host` goes through `canonical_workspace_host` before any request. Sends
    one POST to the workspace token endpoint with HTTP Basic auth and no
    redirects. Returns the access token. Raises `InvalidWorkspaceHost`,
    `GenieCredentialsRefused` (invalid_client or unauthorized_client on any
    4xx, or a 401 with no readable error), `GenieRequestRefused` (any other 3xx or 4xx, or
    a 200 without a token) or `GenieUnavailable` (network error or 5xx, with
    the cause kept). Messages never carry the secret, the token or body text.
    `transport` is for tests.
    """
    url = f"https://{canonical_workspace_host(host)}{TOKEN_PATH}"
    try:
        async with httpx2.AsyncClient(
            transport=transport, timeout=_TOKEN_TIMEOUT_SECONDS, follow_redirects=False
        ) as client:
            response = await client.post(
                url,
                data={"grant_type": "client_credentials", "scope": TOKEN_SCOPE},
                auth=httpx2.BasicAuth(client_id, client_secret),
            )
    except httpx2.HTTPError as exc:
        raise GenieUnavailable("The Databricks workspace could not be reached") from exc

    status = response.status_code
    if status >= 500:
        raise GenieUnavailable(f"The Databricks workspace failed the token request (HTTP {status})")
    error = _error_code(response) if status >= 400 else None
    if error in _CREDENTIAL_ERRORS or (status == 401 and error is None):
        raise GenieCredentialsRefused("Databricks refused the service principal's client ID or secret")
    if status != 200:
        raise GenieRequestRefused(f"Databricks refused the token request (HTTP {status})")
    try:
        payload = response.json()
    except ValueError:
        payload = None
    token = payload.get("access_token") if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token:
        raise GenieRequestRefused("Databricks answered the token request without a token")
    return token


async def wait_for_genie_answer(
    session: "McpSession",
    conversation_id: str,
    response_id: str,
    *,
    interval: float = GENIE_POLL_INTERVAL_SECONDS,
    limit: float = GENIE_WAIT_SECONDS,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll `genie_poll_response` until Genie's status leaves `in_progress`.

    Polls once at once, then every `interval` seconds while less than
    `limit` seconds have passed on `clock`. Returns the last poll's
    structured fields, whose status is still `in_progress` when the limit
    ran out. Raises `GenieAnswerFailed` (a fixed message, never the reply
    text) when a poll reports an error or carries no status; MCP errors from
    the session propagate. `sleep` and `clock` are for tests.
    """
    started = clock()
    arguments = {"conversation_id": conversation_id, "response_id": response_id}
    while True:
        structured, _, is_error = await session.call_tool_structured("genie_poll_response", arguments)
        if is_error or not isinstance(structured, dict) or "status" not in structured:
            raise GenieAnswerFailed("Genie did not report the answer's status")
        if structured["status"] != GENIE_IN_PROGRESS or clock() - started + interval > limit:
            return structured
        await sleep(interval)
