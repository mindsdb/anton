"""Databricks helpers for the Genie MCP connection.

`canonical_workspace_host` is the one gate between a workspace address a
user typed and any URL built from it, so a token is only ever sent to a
Databricks-owned host. `genie_mcp_url` builds the Genie MCP address from it.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

_HOST_SUFFIXES = (".cloud.databricks.com", ".azuredatabricks.net", ".gcp.databricks.com")
_LABEL = re.compile(r"^[a-z0-9-]{1,63}$")
_QUERY = re.compile(r"^o=\d+$")
_SPACE_ID = re.compile(r"[0-9a-f]{32}")

GENIE_ONE_PATH = "/ai-gateway/mcp-services/system.ai.genie_one_mcp"
PER_SPACE_PATH = "/api/2.0/mcp/genie/{space_id}"


class InvalidWorkspaceHost(ValueError):
    """The value is not a Databricks workspace address."""


class InvalidGenieSpace(ValueError):
    """The value is not a Genie space id."""


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
