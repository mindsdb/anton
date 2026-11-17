"""Guarded `McpSession`s through the real pinned transport and a real TLS server.

The server listens on loopback with a certificate valid only for `HOST`, so a
request reaches it only when the transport dials the checked address and keeps
`HOST` as the TLS name. Loopback is admitted by an address rule built here;
production sessions use `is_public_destination`.
"""

from __future__ import annotations

import asyncio
import datetime
import ipaddress
import socket
import ssl
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx2
import pytest
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from anton.core.mcp.client import McpSession
from anton.core.mcp.egress import EgressGuard, is_public_destination
from anton.core.mcp.errors import McpEgressBlocked, McpTransientError
from anton.core.mcp.wiring import close_mcp_sessions

HOST = "mcp.pinned.test"
LOOPBACK = "127.0.0.1"
METADATA = "169.254.169.254"


@dataclass
class TlsServer:
    url: str
    ssl_context: ssl.SSLContext
    seen: list[tuple[str, str, bool]] = field(default_factory=list)


def _write_certificate(directory) -> tuple[str, str]:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, HOST)])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(HOST)]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = directory / "cert.pem", directory / "key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    return str(cert_path), str(key_path)


@pytest.fixture(scope="module")
def tls_server(tmp_path_factory) -> Iterator[TlsServer]:
    """An MCP server over TLS on loopback that records every request it receives."""
    cert_path, key_path = _write_certificate(tmp_path_factory.mktemp("tls"))
    mcp_server = MCPServer(name="pinned")

    @mcp_server.tool()
    def echo(text: str) -> str:
        return text

    app = mcp_server.streamable_http_app(
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False)
    )
    seen: list[tuple[str, str, bool]] = []

    async def recording_app(scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            seen.append((scope["method"], headers.get(b"host", b"").decode(), b"authorization" in headers))
        await app(scope, receive, send)

    probe = socket.socket()
    probe.bind((LOOPBACK, 0))
    port = probe.getsockname()[1]
    probe.close()
    server = uvicorn.Server(
        uvicorn.Config(
            recording_app, host=LOOPBACK, port=port, log_level="error", ssl_certfile=cert_path, ssl_keyfile=key_path
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("TLS test server did not start")
        time.sleep(0.05)
    try:
        yield TlsServer(
            url=f"https://{HOST}:{port}/mcp",
            ssl_context=ssl.create_default_context(cafile=cert_path),
            seen=seen,
        )
    finally:
        server.should_exit = True
        thread.join(5)


@pytest.fixture(autouse=True)
def _clear_seen(tls_server: TlsServer) -> None:
    tls_server.seen.clear()


class Answers:
    """A resolver whose answer a test can change between requests."""

    def __init__(self, *addresses: str) -> None:
        self.addresses = list(addresses)

    async def __call__(self, host: str, port: int):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port)) for address in self.addresses]


def _loopback_only(address) -> bool:
    return address == ipaddress.ip_address(LOOPBACK)


def _guard(server: TlsServer, answers: Answers, *, admit_loopback: bool = True) -> EgressGuard:
    return EgressGuard(
        resolver=answers,
        address_rule=_loopback_only if admit_loopback else is_public_destination,
        verify=server.ssl_context,
    )


def _session(server: TlsServer, guard: EgressGuard) -> McpSession:
    return McpSession(url=server.url, access_token="test-token", egress_guard=guard)


async def _assert_task_not_cancelled() -> None:
    task = asyncio.current_task()
    assert task is not None and task.cancelling() == 0
    await asyncio.sleep(0.01)


def test_a_guarded_session_uses_the_public_address_rule_by_default():
    session = McpSession(url="https://mcp.example.com/mcp", guarded=True)
    assert session._egress is not None
    assert session._egress.address_rule is is_public_destination


def test_a_guarded_session_needs_a_url():
    with pytest.raises(ValueError):
        McpSession(server=object(), guarded=True)


async def test_requests_reach_the_checked_address_with_the_original_host(tls_server):
    async with _session(tls_server, _guard(tls_server, Answers(LOOPBACK))) as session:
        tools = await session.list_tools()
        content, is_error = await session.call_tool("echo", {"text": "hi"})
    assert [tool.name for tool in tools] == ["echo"]
    assert (content, is_error) == ("hi", False)
    port = tls_server.url.split(":")[2].split("/")[0]
    assert tls_server.seen
    assert {host for _, host, _ in tls_server.seen} == {f"{HOST}:{port}"}
    assert all(has_auth for _, _, has_auth in tls_server.seen)


async def test_a_private_answer_at_connect_sends_nothing(tls_server):
    session = _session(tls_server, _guard(tls_server, Answers("10.0.0.5"), admit_loopback=False))
    with pytest.raises(McpEgressBlocked):
        await session.__aenter__()
    await _assert_task_not_cancelled()
    assert tls_server.seen == []


async def test_a_mixed_answer_at_connect_sends_nothing(tls_server):
    session = _session(tls_server, _guard(tls_server, Answers(LOOPBACK, METADATA)))
    with pytest.raises(McpEgressBlocked):
        await session.__aenter__()
    assert tls_server.seen == []


async def test_an_http_url_is_refused_before_any_request(tls_server):
    guard = _guard(tls_server, Answers(LOOPBACK))
    session = McpSession(url=tls_server.url.replace("https://", "http://"), access_token="t", egress_guard=guard)
    with pytest.raises(McpEgressBlocked):
        await session.__aenter__()
    assert tls_server.seen == []


@pytest.mark.parametrize("operation", ["list_tools", "call_tool"])
async def test_a_rebound_answer_after_connect_is_refused_and_not_retried(tls_server, operation):
    answers = Answers(LOOPBACK)
    session = _session(tls_server, _guard(tls_server, answers))
    await session.__aenter__()
    seen_before = len(tls_server.seen)
    answers.addresses = [METADATA]
    with pytest.raises(McpEgressBlocked):
        if operation == "list_tools":
            await session.list_tools()
        else:
            await session.call_tool("echo", {"text": "hi"})
    await _assert_task_not_cancelled()
    with pytest.raises(McpEgressBlocked):
        await session.call_tool("echo", {"text": "again"})
    await session.__aexit__(None, None, None)
    assert [method for method, _, _ in tls_server.seen[seen_before:]] in ([], ["DELETE"])


async def test_a_session_opened_in_a_gather_child_reports_a_refusal_in_the_caller(tls_server):
    answers = Answers(LOOPBACK)
    session = _session(tls_server, _guard(tls_server, answers))
    await asyncio.gather(session.__aenter__())
    seen_before = len(tls_server.seen)
    answers.addresses = [METADATA]
    with pytest.raises(McpEgressBlocked):
        await session.call_tool("echo", {"text": "hi"})
    await _assert_task_not_cancelled()
    assert tls_server.seen[seen_before:] == []
    # Closed the way production closes discovery sessions: closing from a task
    # other than the opener fails on its own, with or without a refusal.
    await close_mcp_sessions([session])


async def test_a_refusal_at_connect_in_a_gather_child_is_reported(tls_server):
    session = _session(tls_server, _guard(tls_server, Answers(METADATA)))
    with pytest.raises(McpEgressBlocked):
        await asyncio.gather(session.__aenter__())
    await _assert_task_not_cancelled()
    assert tls_server.seen == []


class FlakyAnswers(Answers):
    """A resolver that fails `failures` lookups after `fail_after` answers, then recovers."""

    def __init__(self, *addresses: str, fail_after: int, failures: int = 1) -> None:
        super().__init__(*addresses)
        self.calls = 0
        self.fail_after = fail_after
        self.failures = failures

    async def __call__(self, host: str, port: int):
        self.calls += 1
        if self.fail_after < self.calls <= self.fail_after + self.failures:
            raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
        return await super().__call__(host, port)


async def test_a_failed_lookup_during_a_call_is_not_recorded_as_a_refusal(tls_server):
    answers = FlakyAnswers(LOOPBACK, fail_after=10**6)
    guard = _guard(tls_server, answers)
    async with _session(tls_server, guard) as session:
        answers.fail_after = answers.calls
        content, is_error = await session.call_tool("echo", {"text": "hi"})
        await _assert_task_not_cancelled()
        assert (content, is_error) == ("hi", False)
        assert guard.refused is None
        assert await session.call_tool("echo", {"text": "again"}) == ("again", False)


async def test_a_lookup_that_keeps_failing_is_a_transient_error(tls_server):
    answers = FlakyAnswers(LOOPBACK, fail_after=10**6, failures=10**6)
    guard = _guard(tls_server, answers)
    session = _session(tls_server, guard)
    await session.__aenter__()
    answers.fail_after = answers.calls
    with pytest.raises(McpTransientError):
        await session.call_tool("echo", {"text": "hi"})
    await _assert_task_not_cancelled()
    assert guard.refused is None
    await close_mcp_sessions([session])


async def test_an_environment_proxy_is_ignored(tls_server, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    async with _session(tls_server, _guard(tls_server, Answers(LOOPBACK))) as session:
        assert [tool.name for tool in await session.list_tools()] == ["echo"]


@pytest.mark.parametrize(
    "url",
    ["https://elsewhere.test/mcp", f"http://{HOST}/mcp", f"https://{HOST}:8443/mcp"],
    ids=["other host", "plain http", "other port"],
)
async def test_a_request_outside_the_pinned_origin_is_answered_locally_and_recorded(url):
    resolver = Answers(LOOPBACK)
    guard = EgressGuard(resolver=resolver, address_rule=_loopback_only)
    transport = guard.transport(HOST, 443)
    response = await transport.handle_async_request(
        httpx2.Request("GET", url, headers={"Authorization": "Bearer test-token"})
    )
    await transport.aclose()
    assert response.status_code == 421
    assert isinstance(guard.refused, McpEgressBlocked)
