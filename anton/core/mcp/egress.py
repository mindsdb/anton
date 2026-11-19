"""Which custom MCP server URLs and addresses anton may connect to.

A custom server's URL is chosen by a user, so connecting to it from a cloud
pod or a desktop is a server-side request forgery risk: a name can point at
the cloud metadata service, a neighbour on the private network, or the
user's own machine. Only plain public https endpoints pass.

`is_global` alone is not enough: it is true for NAT64, site-local and some
multicast addresses, so those are refused explicitly. Ranges that carry an
IPv4 address inside an IPv6 one (6to4, Teredo, NAT64) are refused outright,
whatever the Python version's `ipaddress` makes of them.

`PinnedHostTransport` resolves and checks the host on every request and
dials the checked address itself, as cowork-server's
`developer_validation.PinnedHostTransport` does, so a name cannot pass the
check and then resolve somewhere private before the connection is made.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import ssl
from collections.abc import Awaitable, Callable
from typing import Any

import httpx2

from anton.core.mcp.errors import McpEgressBlocked, McpHostUnresolved

Address = ipaddress.IPv4Address | ipaddress.IPv6Address

#: `(host, port) -> getaddrinfo records`; injected by tests.
Resolver = Callable[[str, int], Awaitable[list[tuple[Any, ...]]]]

#: Connect timeout for an attempt that has another address after it.
_FALLBACK_CONNECT_SECONDS = 3.0
#: Addresses tried per request. A cloud host can answer with a dozen or more.
_MAX_CONNECTION_ATTEMPTS = 4
#: What a refused request is answered with. Raising inside the transport
#: reaches the session's caller as a cancellation or a transient error,
#: depending on which task opened the session; a response does not.
_REFUSED_STATUS = 421
#: What a request is answered with when the host did not resolve, for the
#: same reason: a transient failure the session classifies and retries.
_UNRESOLVED_STATUS = 503


#: IPv6 ranges refused on every Python version. Python 3.12.4 and later
#: already treat them as private; earlier 3.11 releases let 6to4 through, and
#: 2002:a9fe:a9fe::1 carries the cloud metadata address. Refusing the whole
#: range, rather than judging the embedded IPv4, also keeps a 6to4 address
#: carrying a public IPv4 out, as the desktop app's guard does.
_ALWAYS_REFUSED_V6 = tuple(
    ipaddress.ip_network(network)
    for network in (
        "64:ff9b::/96",  # NAT64
        "2001::/23",  # IETF protocol assignments, Teredo among them
        "2002::/16",  # 6to4
        "3fff::/20",  # documentation
        "5f00::/16",  # segment routing
    )
)


def is_public_destination(address: Address) -> bool:
    """Whether `address` is on the public internet.

    An IPv4-mapped IPv6 address is judged as the IPv4 address it carries, so
    `::ffff:169.254.169.254` cannot pass by changing its spelling.
    """
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if isinstance(address, ipaddress.IPv6Address) and any(address in network for network in _ALWAYS_REFUSED_V6):
        return False
    return address.is_global and not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        or getattr(address, "is_site_local", False)
    )


def check_server_url(url: str) -> tuple[str, int]:
    """Return the ASCII hostname and port of `url`, or raise `McpEgressBlocked`.

    Parsed with httpx2's own URL parser, so the host checked here is the one
    every request names. Accepts https only, with a hostname of valid DNS
    labels, a port from 1 to 65535 (443 when absent), and no whitespace,
    user, password, query or fragment. A query or user part is where a fixed
    credential would ride, and fixed credentials are out of scope.
    """
    if any(character.isspace() for character in url):
        raise McpEgressBlocked("MCP server URL must not contain whitespace")
    try:
        parsed = httpx2.URL(url)
    except (httpx2.InvalidURL, ValueError) as exc:
        raise McpEgressBlocked("MCP server URL could not be parsed") from exc
    if parsed.scheme != "https":
        raise McpEgressBlocked("MCP server URL must use https")
    if parsed.userinfo:
        raise McpEgressBlocked("MCP server URL must not carry a user or password")
    if parsed.query or parsed.fragment:
        raise McpEgressBlocked("MCP server URL must not carry a query or fragment")
    host = ascii_host(parsed)
    if not host:
        raise McpEgressBlocked("MCP server URL has no host")
    if not _is_dns_name(host):
        raise McpEgressBlocked("MCP server URL host is not a valid name")
    port = 443 if parsed.port is None else parsed.port
    if not 1 <= port <= 65535:
        raise McpEgressBlocked("MCP server URL has no usable port")
    return host, port


def ascii_host(url: httpx2.URL) -> str:
    """The host as the request names it on the wire: the ASCII (IDNA 2008)
    form for an international name, so its lookup, TLS name and Host header
    all name one domain."""
    return url.raw_host.decode("ascii").casefold()


def _is_dns_name(host: str) -> bool:
    """Whether `host` is an address literal or a name of 1 to 63 character labels, 253 at most."""
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        pass
    labels = host.removesuffix(".").split(".")
    return len(host) <= 253 and all(1 <= len(label) <= 63 for label in labels)


async def _loop_getaddrinfo(host: str, port: int) -> list[tuple[Any, ...]]:
    """Resolve `host` on the running loop's resolver, without blocking it."""
    loop = asyncio.get_running_loop()
    return await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)


async def resolve_public(
    host: str,
    port: int,
    *,
    resolver: Resolver | None = None,
    address_rule: Callable[[Address], bool] = is_public_destination,
) -> list[Address]:
    """Every address `host` resolves to, when all of them pass `address_rule`.

    Raises `McpHostUnresolved` when the name does not resolve, and
    `McpEgressBlocked` when any answer fails the rule: one private address
    in a mixed answer refuses the whole host, since a later connection
    attempt could land on it. `address_rule` is replaced only by tests that
    serve from loopback.
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
        except UnicodeError as exc:
            raise McpEgressBlocked(f"MCP server host {host!r} is not a valid name") from exc
        except OSError as exc:
            raise McpHostUnresolved(f"MCP server host {host!r} does not resolve") from exc
        addresses = []
        for record in records:
            try:
                addresses.append(ipaddress.ip_address(record[4][0]))
            except (IndexError, ValueError):
                continue
    if not addresses:
        raise McpHostUnresolved(f"MCP server host {host!r} does not resolve")
    if not all(address_rule(address) for address in addresses):
        raise McpEgressBlocked(f"MCP server host {host!r} is not a public address")
    return addresses


def _connection_attempts(addresses: list[Address], *, total_seconds: float | None) -> list[tuple[Address, float | None]]:
    """The checked addresses to try, in order, each with its connect timeout.

    Ported from cowork-server's `egress.connection_attempts`. Families
    alternate from the resolver's first answer, so a family with no route
    costs one short attempt. Every attempt but the last gets the fallback
    timeout and the last gets what is left of `total_seconds`; `None` means
    no limit. Never adds an address.
    """
    first_family = [address for address in addresses if address.version == addresses[0].version]
    other_family = [address for address in addresses if address.version != addresses[0].version]
    ordered: list[Address] = []
    for index in range(max(len(first_family), len(other_family))):
        ordered.extend(family[index] for family in (first_family, other_family) if index < len(family))
    ordered = ordered[:_MAX_CONNECTION_ATTEMPTS]

    last_seconds = None
    if total_seconds is not None:
        while len(ordered) > 1 and total_seconds - _FALLBACK_CONNECT_SECONDS * (len(ordered) - 1) <= 0:
            ordered.pop()
        last_seconds = total_seconds - _FALLBACK_CONNECT_SECONDS * (len(ordered) - 1)
    return [(address, _FALLBACK_CONNECT_SECONDS) for address in ordered[:-1]] + [(ordered[-1], last_seconds)]


class EgressGuard:
    """One guarded session's pinning settings and its refusal record.

    The transport writes the first refusal here; `McpSession` reads it to
    raise `McpEgressBlocked` to its caller. A refusal on any request marks
    the whole session refused.
    """

    def __init__(
        self,
        *,
        resolver: Resolver | None = None,
        address_rule: Callable[[Address], bool] = is_public_destination,
        verify: ssl.SSLContext | bool = True,
    ) -> None:
        """`resolver`, `address_rule` and `verify` are replaced only by
        tests, which serve from loopback with their own certificate."""
        self.resolver = resolver
        self.address_rule = address_rule
        self.verify = verify
        self.refused: McpEgressBlocked | None = None

    def refuse(self, error: McpEgressBlocked) -> None:
        """Record `error` unless an earlier refusal is already recorded."""
        if self.refused is None:
            self.refused = error

    def transport(self, host: str, port: int) -> PinnedHostTransport:
        """A transport that sends requests only over https to checked addresses of `host`:`port`."""
        return PinnedHostTransport(host, port, self)


class PinnedHostTransport(httpx2.AsyncBaseTransport):
    """Send requests for one https origin to an address checked at that moment.

    The original hostname stays in the Host header and is the TLS name the
    certificate is checked against. A request for any other scheme, host or
    port, or a host that fails the check, is never sent: it is answered
    locally with 421 and the refusal is recorded on the guard.
    """

    def __init__(self, host: str, port: int, guard: EgressGuard) -> None:
        """Pin `host` (the ASCII form `check_server_url` returns) and `port`;
        `guard` supplies the checks and records refusals."""
        self._host = host.casefold()
        self._port = port
        self._guard = guard
        self._inner = httpx2.AsyncHTTPTransport(verify=guard.verify)

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        """Dial a checked address of the pinned origin, or answer without a socket."""
        # The wire form, not `url.host`: the system resolver and `ssl` encode a
        # Unicode name by older IDNA rules, which can name a different domain.
        host = ascii_host(request.url)
        if request.url.scheme != "https" or host != self._host or (request.url.port or 443) != self._port:
            return self._refuse(McpEgressBlocked("MCP request is not to the server's own https origin"), request)
        try:
            addresses = await resolve_public(
                host,
                self._port,
                resolver=self._guard.resolver,
                address_rule=self._guard.address_rule,
            )
        except McpHostUnresolved:
            # A lookup can fail for a moment: an outage, not a refusal, so it
            # is not recorded and the session's retry looks the name up again.
            return httpx2.Response(_UNRESOLVED_STATUS, request=request)
        except McpEgressBlocked as exc:
            return self._refuse(exc, request)
        timeouts = request.extensions.get("timeout", {})
        attempts = _connection_attempts(addresses, total_seconds=timeouts.get("connect"))
        for address, connect_seconds in attempts[:-1]:
            try:
                return await self._inner.handle_async_request(self._pinned(request, host, address, connect_seconds))
            except (httpx2.ConnectError, httpx2.ConnectTimeout):
                continue
        address, connect_seconds = attempts[-1]
        return await self._inner.handle_async_request(self._pinned(request, host, address, connect_seconds))

    async def aclose(self) -> None:
        """Close the inner connection pool."""
        await self._inner.aclose()

    def _refuse(self, error: McpEgressBlocked, request: httpx2.Request) -> httpx2.Response:
        """Record `error` on the guard and answer `request` locally with 421."""
        self._guard.refuse(error)
        return httpx2.Response(_REFUSED_STATUS, request=request)

    @staticmethod
    def _pinned(request: httpx2.Request, host: str, address: Address, connect_seconds: float | None) -> httpx2.Request:
        """`request` aimed at `address`, verified as `host`, with its own connect timeout."""
        timeouts = {**request.extensions.get("timeout", {}), "connect": connect_seconds}
        return httpx2.Request(
            request.method,
            request.url.copy_with(host=str(address)),
            headers=request.headers,
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": host, "timeout": timeouts},
        )
