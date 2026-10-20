"""Databricks helpers for the Genie MCP connection.

`canonical_workspace_host` is the one gate between a workspace address a
user typed and any URL built from it, so a token is only ever sent to a
Databricks-owned host.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

_HOST_SUFFIXES = (".cloud.databricks.com", ".azuredatabricks.net", ".gcp.databricks.com")
_LABEL = re.compile(r"^[a-z0-9-]{1,63}$")
_QUERY = re.compile(r"^o=\d+$")


class InvalidWorkspaceHost(ValueError):
    """The value is not a Databricks workspace address."""


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
