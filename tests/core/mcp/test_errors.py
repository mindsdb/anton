from __future__ import annotations

import httpx2
import mcp

from anton.core.mcp.errors import McpPermanentError, McpTransientError, classify_mcp_error


def test_classification_is_idempotent():
    permanent = McpPermanentError("dead")
    transient = McpTransientError("blip")
    assert classify_mcp_error(permanent) is permanent
    assert classify_mcp_error(transient) is transient


def test_unrecognized_exception_defaults_to_transient():
    assert isinstance(classify_mcp_error(ConnectionError("reset")), McpTransientError)
    assert isinstance(classify_mcp_error(TimeoutError()), McpTransientError)


def test_http_401_and_403_are_permanent():
    request = httpx2.Request("GET", "https://mcp.example.test/mcp")
    for status in (401, 403):
        response = httpx2.Response(status, request=request)
        exc = httpx2.HTTPStatusError("auth failed", request=request, response=response)
        assert isinstance(classify_mcp_error(exc), McpPermanentError)


def test_http_5xx_is_transient():
    request = httpx2.Request("GET", "https://mcp.example.test/mcp")
    response = httpx2.Response(503, request=request)
    exc = httpx2.HTTPStatusError("unavailable", request=request, response=response)
    assert isinstance(classify_mcp_error(exc), McpTransientError)


def test_mcp_protocol_error_is_transient_not_matched_against_http_codes():
    """mcp.MCPError.code is a JSON-RPC error code (the SDK's reserved range
    is -32000..-32700 — INTERNAL_ERROR, CONNECTION_CLOSED, ...), never an
    HTTP status. Even a contrived MCPError whose code happens to equal 401
    must NOT be treated as an auth failure — that would be coincidence, not
    signal, since JSON-RPC has no error code of its own for "credential
    rejected" (that's an HTTP-layer concern, handled by the branch above)."""
    real_code_error = mcp.MCPError(code=-32603, message="Internal error")  # INTERNAL_ERROR
    assert isinstance(classify_mcp_error(real_code_error), McpTransientError)

    coincidental_401_error = mcp.MCPError(code=401, message="looks like an HTTP status, isn't one")
    assert isinstance(classify_mcp_error(coincidental_401_error), McpTransientError)
