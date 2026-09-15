"""Per-engine MCP server URLs.

``TurnRequestV1.oauth`` carries only ``{"turn_key", "connections": [{"engine",
"name"}]}`` — no method, no per-connector config (deliberately: a prior,
unused ``base_url`` kwarg on ``TurnKeyDataVault`` was already removed as dead
code). Since a given engine's MCP server URL is a fixed constant, not a
per-connection or per-environment value, a static table here is the single
source of truth for it — nothing else (the wire format, the connector spec
JSON) needs its own copy to keep in sync.
"""

from __future__ import annotations

#: engine -> MCP server base URL. Add an entry here, nowhere else, when a
#: new engine gets an MCP connector.
MCP_SERVER_URLS: dict[str, str] = {
    "hubspot": "https://mcp.hubspot.com",
}


def mcp_server_url(engine: str) -> str | None:
    """The fixed MCP server base URL for `engine`, or None if it has no MCP connector."""
    return MCP_SERVER_URLS.get(engine)
