"""Per-engine MCP server addresses and the credential each connection sends.

``TurnRequestV1.oauth`` carries only ``{"turn_key", "connections": [{"engine",
"name"}]}`` — no method, no per-connector config (deliberately: a prior,
unused ``base_url`` kwarg on ``TurnKeyDataVault`` was already removed as dead
code). Since a given engine's MCP server URL is a fixed constant, not a
per-connection or per-environment value, a static table here is the single
source of truth for it — nothing else (the wire format, the connector spec
JSON) needs its own copy to keep in sync.

Databricks Genie is the exception: its address is built from the stored
workspace host, and its token is minted for the turn from the service
principal's client credentials (`resolve_mcp_endpoint`).
"""

from __future__ import annotations

from anton.core.mcp.databricks import canonical_workspace_host, genie_mcp_url, mint_token
from anton.utils.datasources import hold_secret_value

#: engine -> MCP server base URL. Add an entry here, nowhere else, when a
#: new engine gets an MCP connector.
MCP_SERVER_URLS: dict[str, str] = {
    "hubspot": "https://mcp.hubspot.com",
    "notion": "https://mcp.notion.com/mcp",
}

#: The `_method` values whose connections speak MCP.
MCP_METHODS: frozenset[str] = frozenset({"mcp", "genie-mcp"})


def mcp_server_url(engine: str) -> str | None:
    """The fixed MCP server base URL for `engine`, or None if it has no MCP connector."""
    return MCP_SERVER_URLS.get(engine)


async def resolve_mcp_endpoint(engine: str, method: str, fields: dict[str, str]) -> tuple[str, str] | None:
    """Return the `(url, bearer token)` one MCP connection opens its session with.

    `method` "mcp" uses the engine's fixed URL and the stored `access_token`.
    `("databricks", "genie-mcp")` builds the Genie address from
    `workspace_url` and optional `genie_space_id`, then mints a token from
    `client_id` and `_client_secret`. Returns None when the pair is unknown
    or a field is missing. Raises the `databricks` errors (a bad host or
    space id before any request). Side effect: the secret and the minted
    token are held for this turn's `scrub_credentials`.
    """
    if method == "mcp":
        url, access_token = mcp_server_url(engine), fields.get("access_token")
        return (url, access_token) if url and access_token else None
    if (engine, method) != ("databricks", "genie-mcp"):
        return None
    workspace_url, client_id = fields.get("workspace_url"), fields.get("client_id")
    client_secret = fields.get("_client_secret")
    if not workspace_url or not client_id or not client_secret:
        return None
    host = canonical_workspace_host(workspace_url)
    url = genie_mcp_url(host, fields.get("genie_space_id") or None)
    label = f"DS_{engine.upper()}"
    hold_secret_value(f"{label}__CLIENT_SECRET", client_secret)
    token = await mint_token(host, client_id, client_secret)
    hold_secret_value(f"{label}__ACCESS_TOKEN", token)
    return url, token
