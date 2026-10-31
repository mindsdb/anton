"""The `approve_held_write` tool: ask the user about a request the write relay holds.

On a stack that runs the write relay, the relay is the agent's only network
path to each host it relays. It forwards a GET or HEAD and holds every other
request, answering 428 with the held request's `pending_id` and SHA-256, so
nothing reaches the host when the code sends it. The model then calls this
tool. anton reads the held request from the relay's control address
(`write_relay_url`), shows the relay's card text on an Approve or Cancel card
whose question id names the held request, and on Approve waits for the relay
to send it.

anton never makes the relay send: it only reads and cancels. The recording
proxy signs the approval the relay checks, and only for the prompt Cowork
streamed under that question id, compared with the relay's card text byte
for byte. So the card's prompt is the relay's text exactly: nothing added,
removed or scrubbed.

A cell's own check (`scratchpad.hold_unapproved_write`) leaves out a request
to a relayed host (`held_by_relay`), so one request raises one card.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Literal, NamedTuple

import httpx2 as httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from anton.core.interaction.elicit import AskOption
from anton.core.tools.registry import ToolOutcome
from anton.core.utils.scratchpad import RefusalCause, ask_approval_choice, refusal_why

if TYPE_CHECKING:
    from anton.core.session import ChatSession
    from anton.core.utils.write_check import HttpWrite

logger = logging.getLogger(__name__)

# The relay is a sibling container on an internal network: one that doesn't
# answer within these isn't answering.
RELAY_TIMEOUT = httpx.Timeout(5.0, connect=2.0)
# A held request's card shows a body of at most 64 KiB, each character at
# most nine bytes once escaped for the card and for JSON, and the host's
# answer at most 16 KiB.
MAX_REPLY_BYTES = 2 * 1024 * 1024
# After Approve, how often anton reads the held request and for how long.
# The recording proxy posts its approval once Cowork accepts the answer, and
# the approval is good for 120 seconds, so a request still pending at the
# limit is cancelled before a late approval could send it.
POLL_INTERVAL_S = 0.5
POLL_LIMIT_S = 90.0

PENDING_ID = re.compile(r"[0-9a-f]{32}")
# A host and an optional port, as an entry of `write_relay_hosts` and an
# `http://` URL write them. Anything else, such as user info, a percent
# escape, a backslash or a part the code fills in when it runs, is no match.
_AUTHORITY = r"([a-z0-9.-]+)(?::([0-9]{1,5}))?"
_RELAY_ENTRY = re.compile(_AUTHORITY, re.IGNORECASE)
_HTTP_URL = re.compile(r"http://" + _AUTHORITY + r"(?=[/?#]|\Z)", re.IGNORECASE)

HeldState = Literal["pending", "sending", "sent", "failed", "unknown", "cancelled", "expired"]
# The states after which the relay never sends the request.
_DONE: frozenset[str] = frozenset({"sent", "failed", "unknown", "cancelled", "expired"})


def _host_key(host: str, port: str | None) -> str | None:
    """`host:port`, lowercased, port 80 when none is written; None for a port out of range."""
    number = int(port) if port else 80
    return f"{host.lower()}:{number}" if 0 < number < 65536 else None


def parse_relay_hosts(text: str) -> frozenset[str]:
    """The `host:port`s in the `write_relay_hosts` setting, lowercased:
    entries separated by newlines or commas, an entry without a port meaning
    port 80. A bad entry is skipped with a WARNING that names it."""
    hosts: set[str] = set()
    for entry in (e.strip() for e in re.split(r"[\n,]", text)):
        if not entry:
            continue
        match = _RELAY_ENTRY.fullmatch(entry)
        key = _host_key(match[1], match[2]) if match else None
        if key is None:
            logger.warning("write approval: skipped the write relay host entry %r: an entry is host:port", entry)
            continue
        hosts.add(key)
    return frozenset(hosts)


def relayed_host(url: str) -> str | None:
    """The `host:port` an `http://` URL sends to, lowercased, port 80 when
    none is written; None for any other URL, and for one whose host or port
    the code fills in when it runs."""
    match = _HTTP_URL.match(url)
    return _host_key(match[1], match[2]) if match else None


def held_by_relay(write: HttpWrite, hosts: frozenset[str]) -> bool:
    """Whether the write relay holds `write`: it writes out an `http://` URL
    to one of `hosts`. An address the code picks when it runs could be any
    host, so the cell's check keeps it."""
    return write.url_code is None and relayed_host(write.url) in hosts


class _Wire(BaseModel):
    # The relay may add fields; anton reads the ones it knows. A reply's
    # values stay out of error text, since the warnings log these errors.
    model_config = ConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True)


class HeldResult(_Wire):
    """The host's answer to a request the relay sent: its status, and its
    body as UTF-8 text, at most 16 KiB of it."""

    status: int
    body: str
    truncated: bool
    content_type: str


class HeldRequest(_Wire):
    """A request the relay holds, as `GET /held/<pending_id>` answers."""

    pending_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    # The same lowercase hex the recording proxy parses out of the question id.
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    host: str
    method: str
    card_text: str
    state: HeldState
    expires_at: int
    result: HeldResult | None = None


class _CancelReply(_Wire):
    state: str


class _Reply(NamedTuple):
    status: int
    body: bytes


class RelayUnavailable(Exception):
    """The relay gave no answer anton can use: unreachable, too slow, or a
    reply that isn't what it should be."""


class WriteRelay:
    """The relay's control address: read a held request, or cancel it."""

    def __init__(self, url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        # trust_env=False: the relay is a sibling container on an internal
        # network, never behind the egress proxy a deployment may set.
        self._client = httpx.AsyncClient(
            base_url=url.rstrip("/"), timeout=RELAY_TIMEOUT, trust_env=False, transport=transport,
        )

    async def __aenter__(self) -> WriteRelay:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._client.aclose()

    async def held(self, pending_id: str) -> HeldRequest | None:
        """The held request, or None when the relay holds none by that id."""
        reply = await self._call("GET", f"/held/{pending_id}")
        if reply.status == 404:
            return None
        if reply.status != 200:
            raise RelayUnavailable(f"the relay answered {reply.status}")
        try:
            held = HeldRequest.model_validate_json(reply.body)
        except ValidationError as exc:
            raise RelayUnavailable("the relay's answer isn't a held request") from exc
        if held.pending_id != pending_id:
            raise RelayUnavailable("the relay answered for another held request")
        return held

    async def cancel(self, pending_id: str) -> str | None:
        """The held request's state after the cancel: `cancelled`, or the
        state it was already in when it wasn't pending. None when the relay
        holds none by that id."""
        reply = await self._call("POST", f"/held/{pending_id}/cancel")
        if reply.status == 404:
            return None
        if reply.status not in (200, 409):
            raise RelayUnavailable(f"the relay answered {reply.status}")
        try:
            return _CancelReply.model_validate_json(reply.body).state
        except ValidationError as exc:
            raise RelayUnavailable("the relay's answer to the cancel has no state") from exc

    async def _call(self, method: str, path: str) -> _Reply:
        try:
            async with self._client.stream(method, path) as response:
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body += chunk
                    if len(body) > MAX_REPLY_BYTES:
                        raise RelayUnavailable("the relay answered more than anton reads")
                return _Reply(status=response.status_code, body=bytes(body))
        except httpx.TimeoutException as exc:
            raise RelayUnavailable("the relay gave no answer in time") from exc
        except httpx.HTTPError as exc:
            raise RelayUnavailable(f"the relay could not be reached ({type(exc).__name__})") from exc


async def handle_approve_held_write(session: ChatSession, tc_input: dict) -> ToolOutcome:
    """The `approve_held_write` tool. The id is checked before any request,
    so the model's text never reaches the relay's path."""
    pending_id = tc_input.get("pending_id")
    if not isinstance(pending_id, str) or not PENDING_ID.fullmatch(pending_id):
        return _outcome(
            "Not asked: pending_id must be the 32 lowercase hex characters of the pending_id in the "
            "host's 428 held_for_approval reply. Nothing was sent.",
        )
    url = getattr(session, "_write_relay_url", None)
    if not isinstance(url, str):
        return _outcome("Not asked: this session has no write relay. Nothing was sent.")
    async with WriteRelay(url) as relay:
        return await approve_held_write(session, relay, pending_id=pending_id)


async def approve_held_write(
    session: ChatSession,
    relay: WriteRelay,
    *,
    pending_id: str,
    poll_interval_s: float = POLL_INTERVAL_S,
    poll_limit_s: float = POLL_LIMIT_S,
) -> ToolOutcome:
    """Ask the user about one held request, and report what came of it.

    Every outcome is ok=True, as the cell gate's refusals are: a held
    request the user doesn't approve is their decision, not a tool failure,
    and the error streak's nudge would send the model to try another way.
    """
    try:
        held = await relay.held(pending_id)
    except RelayUnavailable as exc:
        _log(pending_id, "relay unavailable")
        return _outcome(
            f"Not asked: anton couldn't read the held request from the write relay ({exc}). Nothing was "
            "sent; the relay drops the request when it expires.",
        )
    if held is None:
        return _outcome("That held request is unknown or expired; nothing was sent.")
    if held.state != "pending":
        return _outcome(
            f"The {_request(held)} isn't waiting for approval any more. {_state_text(held)}",
        )
    choice = await ask_approval_choice(
        session, held.card_text,
        options=(AskOption(value="approve", label="Approve", detail="Send this request once"),),
        cancel_detail="Drop the request, and nothing is sent",
        log_label="held write approval",
        question_id=f"write:held:{held.pending_id}:{held.sha256}",
    )
    if choice.cause is not None:
        await _cancel_quietly(relay, pending_id)
        _log(pending_id, choice.cause)
        return _refusal(held, choice.cause)
    return await _await_send(relay, held, poll_interval_s=poll_interval_s, poll_limit_s=poll_limit_s)


async def _await_send(
    relay: WriteRelay, held: HeldRequest, *, poll_interval_s: float, poll_limit_s: float,
) -> ToolOutcome:
    """After Approve: read the held request until the relay is done with it,
    or the limit passes. A request still pending then is cancelled, since no
    approval reached the relay."""
    approved = f"The user approved the {_request(held)}."
    loop = asyncio.get_running_loop()
    deadline = loop.time() + poll_limit_s
    last = held
    while True:
        await asyncio.sleep(poll_interval_s)
        try:
            current = await relay.held(held.pending_id)
        except RelayUnavailable:
            # One missed read; the next may answer.
            current = last
        if current is None:
            return _unconfirmed(held, approved)
        last = current
        if current.state in _DONE:
            return _approved_outcome(current, approved)
        if loop.time() >= deadline:
            break
    if last.state == "sending":
        return _approved_outcome(last, approved)
    return await _cancel_unapproved(relay, held, approved=approved, poll_limit_s=poll_limit_s)


async def _cancel_unapproved(
    relay: WriteRelay, held: HeldRequest, *, approved: str, poll_limit_s: float,
) -> ToolOutcome:
    """Cancel a request no approval reached within the limit, and report it.
    An approval that lands just before the cancel wins: the relay answers
    with the state it moved to, and anton reports what came of it instead."""
    try:
        state = await relay.cancel(held.pending_id)
    except RelayUnavailable:
        _log(held.pending_id, "approved, not sent in time, the cancel failed")
        return _outcome(
            f"{approved} But the write relay didn't report it sent within {poll_limit_s:g} seconds, and "
            "anton couldn't cancel it. The write records show whether it was sent; the relay drops a held "
            "request no approval reaches when it expires. Don't send it again: tell the user.",
        )
    if state == "cancelled":
        _log(held.pending_id, "approved, no approval reached the relay, cancelled")
        return _outcome(
            f"{approved} But no approval reached the write relay within {poll_limit_s:g} seconds, so anton "
            "cancelled the held request. Nothing was sent. Tell the user the change wasn't made.",
        )
    try:
        current = await relay.held(held.pending_id) if state is not None else None
    except RelayUnavailable:
        current = None
    if current is None or current.state == "pending":
        return _unconfirmed(held, approved)
    return _approved_outcome(current, approved)


def _approved_outcome(current: HeldRequest, approved: str) -> ToolOutcome:
    """An approved request the relay sent, is sending, or is done with."""
    _log(current.pending_id, f"approved, {current.state}")
    return _outcome(f"{approved} {_state_text(current)}")


def _unconfirmed(held: HeldRequest, approved: str) -> ToolOutcome:
    """An approved request the relay no longer reports on."""
    _log(held.pending_id, "approved, the relay no longer holds it")
    return _outcome(
        f"{approved} The write relay no longer holds it, so anton can't confirm what came of it. The write "
        "records show whether it was sent. Don't send it again: tell the user.",
    )


def _request(held: HeldRequest) -> str:
    return f"held {held.method} request to {held.host}"


def _state_text(held: HeldRequest) -> str:
    """What came of a request the relay isn't holding for approval any more, for the model."""
    if held.state == "sent":
        result = held.result
        if result is None:
            return "The write relay sent it once."
        cut = ", cut at 16 KiB" if result.truncated else ""
        return (
            f"The write relay sent it once, and the host answered {result.status} "
            f"({result.content_type or 'no content type'}{cut}):\n\n{result.body}"
        )
    if held.state == "sending":
        return (
            "The write relay is sending it but hasn't confirmed the outcome. The write records will show "
            "it. Don't send it again."
        )
    if held.state == "failed":
        return "The write relay couldn't reach the host, so nothing was sent."
    if held.state == "unknown":
        return (
            "The write relay started sending it, but the connection broke or timed out, so it isn't known "
            "whether the host made the change. The write records show what the relay knows. Don't send it "
            "again: tell the user."
        )
    if held.state == "cancelled":
        return "It was cancelled before it was sent. Nothing was sent."
    return "It expired before it was sent. Nothing was sent."


def _refusal(held: HeldRequest, cause: RefusalCause) -> ToolOutcome:
    """The cell gate's refusal (`scratchpad._refusal`), for a held request.
    reason=write_not_approved, so the turn's report lists the step as one
    that did nothing."""
    return ToolOutcome(
        content=(
            f"Not sent: the {_request(held)} needs the user's approval, and "
            f"{refusal_why(cause)}. Nothing was sent. The user didn't approve this change, so it must not be "
            "made another way: don't send the request again, rewrite the cell, build the URL at run time, or "
            "use another tool for it. Tell the user what the request would change and that it needs their "
            "approval."
        ),
        ok=True,
        reason="write_not_approved",
    )


def _outcome(content: str) -> ToolOutcome:
    # ok=True, and no reason: only a refusal carries one (`_refusal`).
    return ToolOutcome(content=content, ok=True)


async def _cancel_quietly(relay: WriteRelay, pending_id: str) -> None:
    """Cancel a held request the user didn't approve. A cancel that fails
    changes nothing: no approval can come for a card that is closed, and
    the relay drops the request when it expires."""
    try:
        await relay.cancel(pending_id)
    except RelayUnavailable as exc:
        logger.warning("write approval: held request %s: the cancel failed (%s)", pending_id, exc)


def _log(pending_id: str, outcome: str) -> None:
    # WARNING, as the cell gate logs its decisions: hosts log at WARNING by
    # default. The id and the outcome only, never the card or the answer.
    logger.warning("write approval: held request %s: %s", pending_id, outcome)
