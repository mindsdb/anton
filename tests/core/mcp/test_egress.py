from __future__ import annotations

import ipaddress
import socket

import pytest

from anton.core.mcp.egress import check_server_url, is_public_destination, resolve_public
from anton.core.mcp.errors import McpEgressBlocked, McpHostUnresolved, McpPermanentError


def _resolver(*addresses: str):
    """A fake getaddrinfo answering every name with `addresses`."""

    async def resolve(host: str, port: int):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port)) for address in addresses]

    return resolve


async def _unresolvable(host: str, port: int):
    raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.251",
        "::1",
        "::ffff:169.254.169.254",
        "64:ff9b::a9fe:a9fe",
        "fec0::1",
        "ff02::1",
        "fe80::1",
        "fd00::1",
    ],
)
def test_non_public_addresses_are_refused(address):
    assert not is_public_destination(ipaddress.ip_address(address))


@pytest.mark.parametrize("address", ["93.184.215.14", "2606:4700:4700::1111", "::ffff:93.184.215.14"])
def test_public_addresses_pass(address):
    assert is_public_destination(ipaddress.ip_address(address))


def test_a_plain_https_url_passes_and_returns_its_host():
    assert check_server_url("https://mcp.example.com/mcp") == "mcp.example.com"


@pytest.mark.parametrize(
    "url",
    [
        "http://mcp.example.com/mcp",
        "ftp://mcp.example.com/mcp",
        "https://user:secret@mcp.example.com/mcp",
        "https://user@mcp.example.com/mcp",
        "https://mcp.example.com/mcp?key=secret",
        "https://mcp.example.com/mcp#frag",
        "https:///mcp",
        "mcp.example.com/mcp",
        "https://[::1/mcp",
    ],
)
def test_unusable_urls_are_refused(url):
    with pytest.raises(McpEgressBlocked):
        check_server_url(url)


def test_a_refusal_is_permanent_so_it_is_never_retried():
    assert issubclass(McpEgressBlocked, McpPermanentError)


async def test_a_name_resolving_to_public_addresses_returns_them():
    addresses = await resolve_public("mcp.example.com", 443, resolver=_resolver("93.184.215.14"))
    assert addresses == [ipaddress.ip_address("93.184.215.14")]


async def test_a_name_resolving_to_a_private_address_is_refused():
    with pytest.raises(McpEgressBlocked):
        await resolve_public("mcp.example.com", 443, resolver=_resolver("10.0.0.5"))


async def test_one_private_address_in_a_mixed_answer_refuses_the_host():
    with pytest.raises(McpEgressBlocked):
        await resolve_public("mcp.example.com", 443, resolver=_resolver("93.184.215.14", "169.254.169.254"))


async def test_a_private_literal_host_is_refused_without_a_lookup():
    async def must_not_resolve(host, port):
        raise AssertionError("a literal address needs no lookup")

    with pytest.raises(McpEgressBlocked):
        await resolve_public("[::1]", 443, resolver=must_not_resolve)


async def test_an_unresolvable_host_is_reported_as_unresolved():
    with pytest.raises(McpHostUnresolved):
        await resolve_public("nowhere.example.com", 443, resolver=_unresolvable)


async def test_an_answer_with_no_parsable_address_is_reported_as_unresolved():
    async def garbage(host, port):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("not-an-ip", port))]

    with pytest.raises(McpHostUnresolved):
        await resolve_public("mcp.example.com", 443, resolver=garbage)


async def test_a_private_answer_is_a_refusal_not_an_unresolved_host():
    with pytest.raises(McpEgressBlocked) as caught:
        await resolve_public("mcp.example.com", 443, resolver=_resolver("10.0.0.5"))
    assert not isinstance(caught.value, McpHostUnresolved)
