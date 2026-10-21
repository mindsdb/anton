from __future__ import annotations

import base64
from urllib.parse import parse_qs

import httpx2
import pytest

from anton.core.mcp.databricks import (
    TOKEN_SCOPE,
    GenieCredentialsRefused,
    GenieRequestRefused,
    GenieUnavailable,
    InvalidGenieSpace,
    InvalidWorkspaceHost,
    canonical_workspace_host,
    genie_mcp_url,
    mint_token,
)

HOST = "dbc-1.cloud.databricks.com"
SPACE = "01f0a1b2c3d4e5f60718293a4b5c6d7e"
SECRET = "sentinel-secret-1"
TOKEN = "sentinel-token"


@pytest.mark.parametrize(
    "value,host",
    [
        ("https://dbc-143742e4-8e44.cloud.databricks.com", "dbc-143742e4-8e44.cloud.databricks.com"),
        ("https://DBC-1.cloud.databricks.com/", "dbc-1.cloud.databricks.com"),
        ("https://dbc-1.cloud.databricks.com/?o=3683687545921416", "dbc-1.cloud.databricks.com"),
        ("dbc-1.cloud.databricks.com", "dbc-1.cloud.databricks.com"),
        ("https://adb-123.7.azuredatabricks.net", "adb-123.7.azuredatabricks.net"),
        ("https://1234.5.gcp.databricks.com", "1234.5.gcp.databricks.com"),
    ],
)
def test_a_workspace_address_is_reduced_to_its_host(value, host):
    assert canonical_workspace_host(value) == host


@pytest.mark.parametrize(
    "value",
    [
        "http://dbc-1.cloud.databricks.com",
        "https://dbc-1.cloud.databricks.com:8443",
        "https://dbc-1.cloud.databricks.com/sql/1.0",
        "https://user@dbc-1.cloud.databricks.com",
        "https://cloud.databricks.com",
        "https://dbc-1.cloud.databricks.com.evil.example",
        "https://evilcloud.databricks.com",
        "https://evil.example;.cloud.databricks.com",
        "https://evil%2fx.cloud.databricks.com",
        "https://evil\\x.cloud.databricks.com",
        "https://dbc-1.cloud.databricks.com.",
        "https://dbc-1.cloud.databricks.com/?o=1&x=2",
        "https://10.0.0.5",
        "https://169.254.169.254",
        "",
        "   ",
    ],
)
def test_anything_else_is_refused(value):
    with pytest.raises(InvalidWorkspaceHost):
        canonical_workspace_host(value)


def test_no_space_id_gives_the_genie_one_address():
    assert (
        genie_mcp_url(f"https://{HOST}/")
        == "https://dbc-1.cloud.databricks.com/ai-gateway/mcp-services/system.ai.genie_one_mcp"
    )


def test_a_space_id_gives_the_per_space_address():
    url = genie_mcp_url(HOST, SPACE)
    assert url == "https://dbc-1.cloud.databricks.com/api/2.0/mcp/genie/01f0a1b2c3d4e5f60718293a4b5c6d7e"
    assert url.endswith(SPACE)


def test_a_space_id_is_stripped_of_surrounding_whitespace():
    assert genie_mcp_url(HOST, f"  {SPACE}\n") == genie_mcp_url(HOST, SPACE)


@pytest.mark.parametrize(
    "space_id",
    [
        "../x",
        "a/b",
        "x?y",
        "",
        " ",
        "a" * 200,
        SPACE.upper(),
        SPACE[:-1],
        SPACE + "0",
        SPACE[:-1] + "g",
        SPACE[:-1] + "٣",
    ],
)
def test_a_malformed_space_id_is_refused(space_id):
    with pytest.raises(InvalidGenieSpace) as raised:
        genie_mcp_url(HOST, space_id)
    assert space_id.strip() == "" or space_id.strip() not in str(raised.value)


def test_a_bad_host_is_refused_before_the_space_id_is_looked_at():
    with pytest.raises(InvalidWorkspaceHost):
        genie_mcp_url("https://evil.example", "../x")


def _transport(handler):
    """Record every request the handler sees and answer with its response."""
    seen: list[httpx2.Request] = []

    def wrapped(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)

    return httpx2.MockTransport(wrapped), seen


def _assert_no_leak(exc: BaseException, *extra: str) -> None:
    text = f"{exc!s} {exc!r}"
    for secret in (SECRET, TOKEN, *extra):
        assert secret not in text


async def test_the_token_comes_from_the_workspace_with_client_credentials():
    body = {"access_token": TOKEN, "token_type": "Bearer", "expires_in": 3600}
    transport, seen = _transport(lambda r: httpx2.Response(200, json=body))

    token = await mint_token(HOST, "client-1", SECRET, transport=transport)

    assert token == TOKEN
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == f"https://{HOST}/oidc/v1/token"
    assert parse_qs(request.content.decode()) == {
        "grant_type": ["client_credentials"],
        "scope": [TOKEN_SCOPE],
    }
    expected = base64.b64encode(f"client-1:{SECRET}".encode()).decode()
    assert request.headers["authorization"] == f"Basic {expected}"


async def test_refused_credentials_are_named_without_the_secret():
    error = {"error": "invalid_client", "error_description": "Client authentication failed"}
    transport, _ = _transport(lambda r: httpx2.Response(401, json=error))

    with pytest.raises(GenieCredentialsRefused) as info:
        await mint_token(HOST, "client-1", SECRET, transport=transport)

    _assert_no_leak(info.value, "Client authentication failed")


@pytest.mark.parametrize("code", ["invalid_client", "unauthorized_client"])
async def test_a_rejected_client_on_a_400_is_refused_credentials(code):
    transport, _ = _transport(lambda r: httpx2.Response(400, json={"error": code}))

    with pytest.raises(GenieCredentialsRefused) as info:
        await mint_token(HOST, "client-1", SECRET, transport=transport)

    _assert_no_leak(info.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(401, text="<html>nope sentinel-token</html>"),
        httpx2.Response(401, json={"message": "sentinel-token"}),
    ],
)
async def test_a_401_with_no_readable_error_is_still_refused_credentials(response):
    transport, _ = _transport(lambda r: response)

    with pytest.raises(GenieCredentialsRefused) as info:
        await mint_token(HOST, "client-1", SECRET, transport=transport)

    _assert_no_leak(info.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(400, json={"error": "invalid_scope", "error_description": "secret detail"}),
        httpx2.Response(403, text="forbidden sentinel-token"),
        httpx2.Response(200, json={"token_type": "Bearer"}),
        httpx2.Response(200, json={"access_token": ""}),
        httpx2.Response(200, json={"access_token": 5}),
        httpx2.Response(200, text="not json sentinel-token"),
    ],
)
async def test_any_other_refusal_is_a_request_refusal_without_body_text(response):
    transport, _ = _transport(lambda r: response)

    with pytest.raises(GenieRequestRefused) as info:
        await mint_token(HOST, "client-1", SECRET, transport=transport)

    _assert_no_leak(info.value, "secret detail", "forbidden")


async def test_an_unreachable_workspace_is_unavailable():
    def refuse(request):
        raise httpx2.ConnectError("boom sentinel-secret-1", request=request)

    transport, _ = _transport(refuse)
    with pytest.raises(GenieUnavailable) as info:
        await mint_token(HOST, "client-1", SECRET, transport=transport)
    assert isinstance(info.value.__cause__, httpx2.ConnectError)
    _assert_no_leak(info.value)

    transport, _ = _transport(lambda r: httpx2.Response(503, text="down sentinel-token"))
    with pytest.raises(GenieUnavailable) as info:
        await mint_token(HOST, "client-1", SECRET, transport=transport)
    _assert_no_leak(info.value, "down")


async def test_a_redirect_is_not_followed():
    transport, seen = _transport(
        lambda r: httpx2.Response(302, headers={"location": "https://elsewhere.example/token"})
    )

    with pytest.raises(GenieRequestRefused) as info:
        await mint_token(HOST, "client-1", SECRET, transport=transport)

    assert len(seen) == 1
    _assert_no_leak(info.value, "elsewhere")


async def test_the_host_is_checked_before_any_request():
    transport, seen = _transport(lambda r: httpx2.Response(200, json={"access_token": TOKEN}))

    with pytest.raises(InvalidWorkspaceHost):
        await mint_token("evil.example", "client-1", SECRET, transport=transport)

    assert seen == []
