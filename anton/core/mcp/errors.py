"""Permanent vs. transient MCP failure classification.

Mirrors the same split the OAuth token-refresh path already makes for a dead
grant vs. a network blip (see ``auth/datavault/services.py``'s
``get_valid_access_token``): a permanent failure means the stored credential
itself is no good and only "reconnect {engine}" fixes it; a transient one
(rate limit, transport hiccup, a mid-turn session death) gets one silent
retry with the same stored token before anything reaches the user — never a
confusing raw error, and never confused for a broken connection when it
wasn't.
"""

from __future__ import annotations

import httpx2
import mcp


class McpError(Exception):
    """Base for the two classified MCP failure kinds."""


class McpTransientError(McpError):
    """A rate limit or transport hiccup — worth one retry with a fresh call."""


class McpPermanentError(McpError):
    """A rejected or expired credential — surface as "reconnect {engine}", never retried."""


#: JSON-RPC / HTTP status codes that mean the credential itself was rejected,
#: not that something transient went wrong in transit.
_AUTH_FAILURE_CODES = frozenset({401, 403})


def classify_mcp_error(exc: Exception) -> McpError:
    """Classify `exc` as `McpPermanentError` or `McpTransientError`.

    Idempotent: an exception that's already one of the two is returned
    as-is. Everything else defaults to transient — an exception type this
    function doesn't recognize (a new httpx2 error, a library upgrade
    changing a type) is far more likely to be a transport blip than a
    credential problem, and the fail-open direction here is safe: a
    genuinely dead credential just needs one extra retry (which will fail
    the same way) before the caller's own final-attempt handling surfaces
    "reconnect" anyway — see `McpSession.call_tool`.
    """
    if isinstance(exc, McpError):
        return exc
    if isinstance(exc, httpx2.HTTPStatusError) and exc.response.status_code in _AUTH_FAILURE_CODES:
        return McpPermanentError(str(exc))
    if isinstance(exc, mcp.MCPError):
        code = getattr(getattr(exc, "error", None), "code", None)
        if code in _AUTH_FAILURE_CODES:
            return McpPermanentError(str(exc))
        return McpTransientError(str(exc))
    return McpTransientError(str(exc))
