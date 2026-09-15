from __future__ import annotations

import httpx2

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
