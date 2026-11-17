"""Which custom MCP server URLs and addresses anton may connect to.

A custom server's URL is chosen by a user, so connecting to it from a cloud
pod or a desktop is a server-side request forgery risk: a name can point at
the cloud metadata service, a neighbour on the private network, or the
user's own machine. Only plain public https endpoints pass.

The address rule matches mindshub_inference's `minds/common/egress.py`
`is_public_destination`. `is_global` alone is not enough: it is true for
NAT64, site-local and some multicast addresses.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

from anton.core.mcp.errors import McpEgressBlocked

Address = ipaddress.IPv4Address | ipaddress.IPv6Address

#: `(host, port) -> getaddrinfo records`; injected by tests.
Resolver = Callable[[str, int], Awaitable[list[tuple[Any, ...]]]]


def is_public_destination(address: Address) -> bool:
    """Whether `address` is on the public internet.

    An IPv4-mapped IPv6 address is judged as the IPv4 address it carries, so
    `::ffff:169.254.169.254` cannot pass by changing its spelling.
    """
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        or getattr(address, "is_site_local", False)
    )


def check_server_url(url: str) -> str:
    """Return the hostname of `url`, or raise `McpEgressBlocked`.

    Accepts https only, with a hostname, and no user, password, query or
    fragment. A query or user part is where a fixed credential would ride,
    and fixed credentials are out of scope for custom servers.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError as exc:
        raise McpEgressBlocked("MCP server URL could not be parsed") from exc
    if parts.scheme != "https":
        raise McpEgressBlocked("MCP server URL must use https")
    if parts.username is not None or parts.password is not None:
        raise McpEgressBlocked("MCP server URL must not carry a user or password")
    if parts.query or parts.fragment:
        raise McpEgressBlocked("MCP server URL must not carry a query or fragment")
    if not host:
        raise McpEgressBlocked("MCP server URL has no host")
    return host


async def _loop_getaddrinfo(host: str, port: int) -> list[tuple[Any, ...]]:
    """Resolve `host` on the running loop's resolver, without blocking it."""
    loop = asyncio.get_running_loop()
    return await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)


async def resolve_public(host: str, port: int, *, resolver: Resolver | None = None) -> list[Address]:
    """Every address `host` resolves to, when all of them are public.

    Raises `McpEgressBlocked` when the name does not resolve or any answer
    is not public: one private address in a mixed answer refuses the whole
    host, since a later connection attempt could land on it.
    """
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None:
        addresses: list[Address] = [literal]
    else:
        try:
            records = await (resolver or _loop_getaddrinfo)(host, port)
        except (OSError, UnicodeError) as exc:
            raise McpEgressBlocked(f"MCP server host {host!r} does not resolve") from exc
        addresses = []
        for record in records:
            try:
                addresses.append(ipaddress.ip_address(record[4][0]))
            except (IndexError, ValueError):
                continue
    if not addresses:
        raise McpEgressBlocked(f"MCP server host {host!r} does not resolve")
    if not all(is_public_destination(address) for address in addresses):
        raise McpEgressBlocked(f"MCP server host {host!r} is not a public address")
    return addresses
