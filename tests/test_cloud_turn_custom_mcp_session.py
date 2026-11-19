"""A cloud turn's custom MCP tools: built from the block with no request, in one fixed form, each calling the gateway."""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import json
import random
import socket
import ssl
import threading
from pathlib import Path
from unittest.mock import MagicMock

import httpx2 as httpx
import pytest

import anton.core.llm.client as llm_client_mod
import anton.core.mcp.wiring as wiring_mod
from anton.cloud_turn.contract import TurnRequestV1
from anton.cloud_turn.custom_mcp import (
    CUT_LINE,
    DROPPED_LINE,
    MAX_RESULT_CHARS,
    build_custom_mcp_tools,
)
from anton.cloud_turn.session import (
    _WORKSPACE_PATH_ENV,
    CLOUD_TOOL_ALLOWLIST,
    build_cloud_chat_session,
)
from anton.core.tools.tool_defs import ToolDef
from tests.test_cloud_turn_custom_api_helper import _self_signed
from tests.test_cloud_turn_session import _mock_llm

_FIXTURES = Path(__file__).parent / "fixtures"
_CALL_FILE = _FIXTURES / "custom_mcp_call_v1.json"
#: Copied byte for byte from mindshub_inference's docs/contracts/custom-mcp-call-v1.json
#: at this commit. Copy again, and update both, when the gateway's file changes.
_CALL_SOURCE = "10a8e6bbc8b1079273701e69e2ca1834443b81c0"
_CALL_SHA256 = "69b55b8e401f7e8d628f93bb487b85ae52254293e8c5e64d2fea6956350f1e96"
_CALL = json.loads(_CALL_FILE.read_text(encoding="utf-8"))
_LIMITS = json.loads((_FIXTURES / "custom_mcp_limits_v1.json").read_text(encoding="utf-8"))
_USABLE_TEXTS = [sample["text"] for sample in _LIMITS["samples"] if sample["usable"]]
_EXAMPLE = _CALL["call_request"]
_TURN_KEY = "mdb_turnkey.secret"
_BASE_URL = "https://minds.internal/v1"
_GATEWAY_URL = "https://minds.internal/v1/datasources/custom-mcp/call"
_CONNECTION_NAME = "cm-0123456789abcdef0123456789abcdef"
_OTHER_NAME = "cm-ffffffffffffffffffffffffffffffff"
_PREFIX = "custom_mcp__" + _CONNECTION_NAME + "__"


def _text(name: str, description: str, schema: dict) -> str:
    """A definition as auth stores it: compact JSON with sorted keys, non-ASCII text as it is."""
    return json.dumps(
        {"name": name, "description": description, "input_schema": schema},
        ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    )


#: The call contract's example tool, with a schema for its two arguments.
_SEARCH = _text(
    _EXAMPLE["tool"],
    "Searches issues.",
    {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["query", "limit"],
    },
)


def _block(**overrides) -> dict:
    """A producer-shaped block: the contract's connection with its example tool, and a second connection."""
    return {
        "protocol_version": 1,
        "connections": [
            {
                "connection_id": _EXAMPLE["connection_id"],
                "name": _CONNECTION_NAME,
                "label": "Issues",
                "credential_version": _EXAMPLE["credential_version"],
                "tools": [_SEARCH, *_USABLE_TEXTS],
            },
            {
                "connection_id": 2,
                "name": _OTHER_NAME,
                "label": "Docs",
                "credential_version": 4,
                # Keys out of order, as a writer that does not sort would send them.
                "tools": ['{"name":"read_page","input_schema":{"type":"object","properties":{"b":{},"a":{}}},"description":"Reads a page."}'],
            },
        ],
        **overrides,
    }


def _request(**fields) -> TurnRequestV1:
    """A turn request read from its wire line, carrying a custom MCP block and no oauth block."""
    body = {
        "protocol_version": 1,
        "conversation_id": "conv_1",
        "correlation_id": _EXAMPLE["correlation_id"],
        "input": "hello",
        "llm": {"provider": "minds-cloud", "api_key": _TURN_KEY, "base_url": _BASE_URL},
        "custom_mcp": _block(),
    }
    body.update(fields)
    return TurnRequestV1.from_json(json.dumps(body))


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    """Build a real cloud ChatSession from a request, with no model and no network."""
    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))
    for name in ("ANTON_OPENAI_API_KEY", "ANTON_OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm_client_mod.LLMClient, "from_settings", classmethod(lambda cls, settings: _mock_llm()))

    def build(request: TurnRequestV1):
        session = build_cloud_chat_session(request)
        session._scratchpads = MagicMock(available_packages=[])
        return session

    return build


@pytest.fixture
def gateway(monkeypatch):
    """Answer every gateway call from a script, recording each request."""
    seen: list[httpx.Request] = []
    answers: list[httpx.Response] = []
    real_client = httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answers.pop(0)

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return seen, answers


def _tool_bytes(session) -> str:
    return json.dumps(session._build_tools())


def _custom_tool(session, bare_name: str):
    return next(t for t in session.tool_registry.get_tool_defs() if t.name.endswith("__" + bare_name))


def _call(session, bare_name: str, arguments: dict):
    session._build_tools()
    return asyncio.run(_custom_tool(session, bare_name).handler(session, arguments))


def test_the_call_fixture_is_the_gateways_file():
    assert hashlib.sha256(_CALL_FILE.read_bytes()).hexdigest() == _CALL_SHA256, (
        f"{_CALL_FILE.name} differs from mindshub_inference's at {_CALL_SOURCE}. Copy it again and update the pin."
    )


def test_building_the_tools_makes_no_request(cloud, monkeypatch):
    attempts: list = []

    def refuse_connect(sock, address):
        attempts.append(address)
        raise OSError("no network in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse_connect)

    cloud(_request())._build_tools()

    assert attempts == []


def test_every_definition_at_the_limits_registers_after_the_built_in_tools_sorted_by_name(cloud):
    """The limits file's at-limit texts go through the parser, the builder and the real _build_tools()."""
    session = cloud(_request())

    names = [tool["name"] for tool in session._build_tools()]

    custom = [name for name in names if name.startswith("custom_mcp__")]
    assert len(custom) == 2 + len(_USABLE_TEXTS)
    assert custom == sorted(custom)
    assert names == [name for name in names if not name.startswith("custom_mcp__")] + custom
    assert set(names) == (set(CLOUD_TOOL_ALLOWLIST) - {"generate_image"}) | set(custom)


def test_custom_tools_come_right_after_the_built_in_mcp_tools(cloud, monkeypatch):
    """A turn with an oauth block whose discovery finds one built-in MCP tool."""
    async def built_in_handler(session, tc_input):
        """Never called: the test only reads the tool list."""
        return ""

    built_in = ToolDef(
        name="hubspot__search", description="Searches.", input_schema={"type": "object"}, handler=built_in_handler
    )
    monkeypatch.setattr(wiring_mod, "discover_mcp_tools", lambda vault, connections: ([built_in], []))
    oauth = {"turn_key": "tk_abc", "connections": [{"engine": "hubspot", "name": "acme"}]}

    names = [tool["name"] for tool in cloud(_request(oauth=oauth))._build_tools()]

    custom = [name for name in names if name.startswith("custom_mcp__")]
    start = names.index(built_in.name) + 1
    assert custom and names[start:start + len(custom)] == custom


def test_the_tool_list_is_the_same_bytes_on_every_build_of_one_turn(cloud):
    session = cloud(_request())

    assert _tool_bytes(session) == _tool_bytes(session)


def test_two_turns_with_the_same_block_send_the_same_tool_list(cloud):
    assert _tool_bytes(cloud(_request())) == _tool_bytes(cloud(_request()))


def _reversed_keys(value):
    """``value`` with every object's keys in reverse order, at every depth."""
    if isinstance(value, dict):
        return {key: _reversed_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reversed_keys(item) for item in value]
    return value


def test_shuffled_connections_tools_and_schema_keys_give_the_same_bytes(cloud):
    shuffled = _block()
    rng = random.Random(7)
    rng.shuffle(shuffled["connections"])
    for connection in shuffled["connections"]:
        rng.shuffle(connection["tools"])
        connection["tools"] = [
            json.dumps(_reversed_keys(json.loads(text)), ensure_ascii=False, separators=(",", ":"))
            for text in connection["tools"]
        ]

    assert _tool_bytes(cloud(_request(custom_mcp=shuffled))) == _tool_bytes(cloud(_request()))


def test_schema_keys_are_sorted_at_every_depth_and_arrays_keep_their_order(cloud):
    session = cloud(_request())

    (read_page,) = [t for t in session._build_tools() if t["name"].endswith("__read_page")]
    (search,) = [t for t in session._build_tools() if t["name"].endswith("__search_issues")]

    assert list(read_page["input_schema"]["properties"]) == ["a", "b"]
    assert list(search["input_schema"]) == ["properties", "required", "type"]
    assert search["input_schema"]["required"] == ["query", "limit"]


def test_label_and_credential_version_never_change_the_tool_list(cloud):
    changed = _block()
    for connection in changed["connections"]:
        connection["label"] = connection["label"] + " renamed"
        connection["credential_version"] += 1

    assert _tool_bytes(cloud(_request(custom_mcp=changed))) == _tool_bytes(cloud(_request()))


def test_a_dropped_block_builds_the_tool_list_of_a_turn_without_one(cloud):
    broken = _block(protocol_version=2)

    assert _tool_bytes(cloud(_request(custom_mcp=broken))) == _tool_bytes(cloud(_request(custom_mcp=None)))


@pytest.mark.parametrize(
    "correlation_id",
    [None, "corr_1", _EXAMPLE["correlation_id"].replace("-", "")],
    ids=["absent", "not-a-uuid", "no-hyphens"],
)
def test_a_custom_mcp_turn_needs_the_correlation_id_the_producer_sends(cloud, correlation_id):
    with pytest.raises(ValueError, match="custom MCP turn credentials are unavailable"):
        cloud(_request(correlation_id=correlation_id))


def test_a_custom_mcp_turn_fails_when_its_base_is_not_this_turns(cloud):
    llm = {"provider": "minds-cloud", "api_key": _TURN_KEY, "base_url": "http://minds.internal/v1"}

    with pytest.raises(ValueError, match="cannot carry custom MCP calls"):
        cloud(_request(llm=llm))


def test_a_call_sends_the_contracts_body_to_the_gateway_with_the_turn_key(cloud, gateway):
    seen, answers = gateway
    answers.append(httpx.Response(200, json=_CALL["call_response"]))
    session = cloud(_request())

    outcome = _call(session, _EXAMPLE["tool"], _EXAMPLE["arguments"])

    (request,) = seen
    assert str(request.url) == _GATEWAY_URL
    assert request.method == "POST"
    assert request.headers["authorization"] == f"Bearer {_TURN_KEY}"
    assert json.loads(request.content) == _EXAMPLE
    assert _TURN_KEY not in request.content.decode()
    assert outcome.ok is True
    text, structured = _CALL["call_response"]["content"]
    assert outcome.content == text["text"] + "\n" + json.dumps(structured["value"], ensure_ascii=False)


def test_two_calls_in_a_row_work_from_one_session(cloud, gateway):
    seen, answers = gateway
    answers.extend([httpx.Response(200, json=_CALL["call_response"]), httpx.Response(200, json=_CALL["call_response"])])
    session = cloud(_request())

    first = _call(session, _EXAMPLE["tool"], _EXAMPLE["arguments"])
    second = _call(session, _EXAMPLE["tool"], _EXAMPLE["arguments"])

    assert len(seen) == 2
    assert first.ok and second.ok
    assert first.content == second.content


def test_a_tool_error_with_dropped_parts_says_a_part_could_not_be_shown(cloud, gateway):
    _, answers = gateway
    answers.append(httpx.Response(200, json=_CALL["call_response_dropped"]))

    outcome = _call(cloud(_request()), _EXAMPLE["tool"], _EXAMPLE["arguments"])

    assert outcome.ok is False
    assert outcome.content == "The image could not be shown.\n" + DROPPED_LINE


def test_a_call_on_the_second_connection_sends_its_own_binding(cloud, gateway):
    seen, answers = gateway
    answers.append(
        httpx.Response(200, json={**_CALL["call_response"], "connection_id": 2, "credential_version": 4})
    )

    outcome = _call(cloud(_request()), "read_page", {})

    body = json.loads(seen[0].content)
    assert (body["connection_id"], body["credential_version"], body["tool"]) == (2, 4, "read_page")
    assert outcome.ok is True


def test_an_answer_for_another_binding_is_not_shown(cloud, gateway):
    _, answers = gateway
    answers.append(httpx.Response(200, json={**_CALL["call_response"], "credential_version": 9}))

    outcome = _call(cloud(_request()), _EXAMPLE["tool"], _EXAMPLE["arguments"])

    assert outcome.ok is False
    assert outcome.content == 'The custom server "Issues" sent an answer that could not be read.'


_FIXED = {
    "needs_reconnect": 'The connection to the custom server "Issues" needs to be reconnected before this tool can run again.',
    "tool_not_allowed": 'This tool is switched off for the custom server "Issues", so it cannot run.',
    "response_too_large": (
        "This tool returned more text than can be shown, so none of it is shown. "
        "Ask for less, for example with a filter or a smaller page."
    ),
    "rate_limited": 'The custom server "Issues" has had too many calls in a short time. Wait a moment, then try again.',
    "destination_forbidden": 'The custom server "Issues" is at an address that is not allowed, so this tool cannot run.',
    "invalid_request": (
        'The custom server "Issues" was not called: the arguments are not a valid JSON object or are too large.'
    ),
    "invalid_connection": 'The custom server "Issues" is not available in this conversation.',
    "gateway_disabled": 'The custom server "Issues" cannot be called for the rest of this reply, so do not try this tool again.',
    "unauthorized": 'The custom server "Issues" cannot be called for the rest of this reply, so do not try this tool again.',
}
_UNAVAILABLE = 'The custom server "Issues" is temporarily unavailable. Try again shortly.'
#: The exact reason per gateway code. Codes the desktop MCP path shares keep its reasons.
_REASON = {
    "unauthorized": "custom_mcp_turn_key_refused",
    "gateway_disabled": "custom_mcp_gateway_disabled",
    "invalid_request": "custom_mcp_invalid_request",
    "invalid_connection": "custom_mcp_unknown_connection",
    "authorization_unavailable": "mcp_transient_error",
    "destination_forbidden": "mcp_egress_blocked",
    "tool_not_allowed": "custom_mcp_tool_switched_off",
    "needs_reconnect": "mcp_permanent_error",
    "rate_limited": "mcp_transient_error",
    "timed_out": "mcp_transient_error",
    "response_too_large": "custom_mcp_too_large",
    "listing_too_large": "mcp_transient_error",
    "upstream_unavailable": "mcp_transient_error",
}


def test_every_gateway_code_has_an_expected_reason():
    assert set(_REASON) == {error["code"] for error in _CALL["errors"]}


@pytest.mark.parametrize("error", _CALL["errors"], ids=[error["code"] for error in _CALL["errors"]])
def test_each_gateway_code_gives_its_fixed_message_and_never_the_gateways_text(cloud, gateway, error):
    _, answers = gateway
    status = error.get("leaves_as", error["status"])
    detail = "planted gateway detail"
    answers.append(httpx.Response(status, json={"code": error["code"], "detail": detail, "request_id": "r"}))

    outcome = _call(cloud(_request()), _EXAMPLE["tool"], _EXAMPLE["arguments"])

    assert outcome.ok is False
    assert outcome.content == _FIXED.get(error["code"], _UNAVAILABLE)
    assert detail not in outcome.content
    assert outcome.reason == _REASON[error["code"]]


@pytest.mark.parametrize(
    "response",
    [httpx.Response(503, text="<html>proxy</html>"), httpx.Response(500, json={"code": "planted_code"})],
    ids=["not-json", "unknown-code"],
)
def test_a_refusal_without_a_known_code_reads_as_temporarily_unavailable(cloud, gateway, response):
    _, answers = gateway
    answers.append(response)

    outcome = _call(cloud(_request()), _EXAMPLE["tool"], _EXAMPLE["arguments"])

    assert outcome.content == _UNAVAILABLE


def test_a_gateway_that_cannot_be_reached_reads_as_temporarily_unavailable(cloud, monkeypatch):
    real_client = httpx.AsyncClient

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(refuse), **kw))

    outcome = _call(cloud(_request()), _EXAMPLE["tool"], _EXAMPLE["arguments"])

    assert outcome.ok is False
    assert outcome.content == _UNAVAILABLE


def test_a_redirect_is_not_followed_so_the_turn_key_goes_nowhere_else(cloud, gateway):
    seen, answers = gateway
    answers.extend(
        [
            httpx.Response(302, headers={"Location": "https://elsewhere.example/v1/datasources/custom-mcp/call"}),
            httpx.Response(200, json=_CALL["call_response"]),
        ]
    )

    outcome = _call(cloud(_request()), _EXAMPLE["tool"], _EXAMPLE["arguments"])

    assert [str(request.url) for request in seen] == [_GATEWAY_URL]
    assert outcome.content == _UNAVAILABLE


def _refusal_of_size(size: int) -> bytes:
    """A ``tool_not_allowed`` refusal body of exactly ``size`` bytes."""
    empty = json.dumps({"code": "tool_not_allowed", "pad": ""}).encode()
    return json.dumps({"code": "tool_not_allowed", "pad": "p" * (size - len(empty))}).encode()


@pytest.mark.parametrize(
    ("size", "expected"),
    [(65_536, _FIXED["tool_not_allowed"]), (65_537, _UNAVAILABLE)],
    ids=["at-the-cap", "one-byte-over"],
)
def test_a_refusal_body_over_64_kib_is_not_read_for_its_code(cloud, gateway, size, expected):
    _, answers = gateway
    body = _refusal_of_size(size)
    assert len(body) == size
    answers.append(httpx.Response(403, content=body))

    outcome = _call(cloud(_request()), _EXAMPLE["tool"], _EXAMPLE["arguments"])

    assert outcome.content == expected


_CHUNK = b"x" * 65_536


@pytest.mark.parametrize(("status", "cap"), [(200, 2_097_152), (403, 65_536)], ids=["answer", "refusal"])
def test_a_body_past_its_cap_is_not_read_to_the_end(cloud, gateway, status, cap):
    """A 64 MiB body: reading stops one chunk past the cap, and a long answer reads as too large."""
    _, answers = gateway
    pulled: list[int] = []

    async def endless():
        for _ in range(1024):
            pulled.append(len(_CHUNK))
            yield _CHUNK

    answers.append(httpx.Response(status, content=endless()))

    outcome = _call(cloud(_request()), _EXAMPLE["tool"], _EXAMPLE["arguments"])

    assert cap < sum(pulled) <= cap + len(_CHUNK)
    assert outcome.ok is False
    assert outcome.content == (_FIXED["response_too_large"] if status == 200 else _UNAVAILABLE)
    if status == 200:
        assert outcome.reason == "custom_mcp_too_large"


@pytest.fixture
def tls_gateway(tmp_path):
    """A local TLS gateway answering the contract's call response; yields (base URL, cert path, requests seen)."""
    pytest.importorskip("cryptography")
    cert_path, key_path = _self_signed(tmp_path)
    seen: list[dict] = []

    class Gateway(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            """Record the call and answer the contract's response."""
            length = int(self.headers["Content-Length"])
            seen.append(
                {"line": self.requestline, "authorization": self.headers["Authorization"],
                 "body": json.loads(self.rfile.read(length))}
            )
            payload = json.dumps(_CALL["call_response"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            """Keep the test output quiet."""

    class QuietServer(http.server.HTTPServer):
        def handle_error(self, request, client_address):
            """A refused handshake is the expected outcome of one test, not noise."""

    server = QuietServer(("127.0.0.1", 0), Gateway)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://localhost:{server.server_address[1]}/v1", cert_path, seen
    finally:
        server.shutdown()
        server.server_close()


def _call_over_tls(base_url: str):
    """One call of the contract's example tool, built from the block and sent to ``base_url``."""
    block = _request().custom_mcp
    (tool,) = [
        t for t in build_custom_mcp_tools(
            block, gateway_base_url=base_url, turn_key=_TURN_KEY, correlation_id=_EXAMPLE["correlation_id"]
        )
        if t.name.endswith("__" + _EXAMPLE["tool"])
    ]
    return asyncio.run(tool.handler(None, _EXAMPLE["arguments"]))


def test_a_call_trusts_the_default_tls_store_and_ignores_proxy_settings(tls_gateway, monkeypatch):
    """The CA comes from SSL_CERT_FILE, as the custom API helper's default context reads it; the proxy is never used."""
    base_url, cert_path, seen = tls_gateway
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_path))
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)

    outcome = _call_over_tls(base_url)

    assert seen == [
        {
            "line": "POST /v1/datasources/custom-mcp/call HTTP/1.1",
            "authorization": f"Bearer {_TURN_KEY}",
            "body": _EXAMPLE,
        }
    ]
    assert outcome.ok is True


def test_a_gateway_whose_certificate_is_not_trusted_gets_nothing(tls_gateway, monkeypatch):
    base_url, _cert_path, seen = tls_gateway
    for name in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
        monkeypatch.delenv(name, raising=False)

    outcome = _call_over_tls(base_url)

    assert seen == []
    assert outcome.content == _UNAVAILABLE


def test_a_connection_with_no_usable_label_is_named_plainly(cloud, gateway):
    _, answers = gateway
    answers.append(httpx.Response(409, json={"code": "needs_reconnect", "detail": "d", "request_id": "r"}))
    block = _block()
    block["connections"][0]["label"] = "\n"

    outcome = _call(cloud(_request(custom_mcp=block)), _EXAMPLE["tool"], _EXAMPLE["arguments"])

    assert outcome.content == "The connection to the custom server needs to be reconnected before this tool can run again."


def _text_answer(text: str) -> dict:
    return {**_CALL["call_response"], "content": [{"type": "text", "text": text}]}


def test_an_answer_of_exactly_the_cap_reaches_the_model_whole(cloud, gateway):
    _, answers = gateway
    answers.append(httpx.Response(200, json=_text_answer("x" * MAX_RESULT_CHARS)))

    outcome = _call(cloud(_request()), _EXAMPLE["tool"], _EXAMPLE["arguments"])

    assert outcome.content == "x" * MAX_RESULT_CHARS


def test_an_answer_one_character_over_the_cap_is_cut_with_the_fixed_line(cloud, gateway):
    _, answers = gateway
    answers.append(httpx.Response(200, json=_text_answer("x" * (MAX_RESULT_CHARS + 1))))

    outcome = _call(cloud(_request()), _EXAMPLE["tool"], _EXAMPLE["arguments"])

    assert MAX_RESULT_CHARS == 60_000
    assert outcome.content == "x" * MAX_RESULT_CHARS + "\n" + CUT_LINE
    assert outcome.ok is True
