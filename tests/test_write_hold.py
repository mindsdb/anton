"""`approve_held_write`: the card for a request the write relay holds.

The relay holds every request but a GET or HEAD to a host it relays and
answers 428 with a pending id. The tool reads the held request from the
relay, shows the relay's card text on an Approve or Cancel card whose
question id names the held request, and on Approve waits for the relay to
send it. The recording proxy signs the approval only for the prompt Cowork
streamed under that question id, compared with the relay's card text byte
for byte, so the card's prompt must be the relay's text exactly.

These drive the tool against a fake relay on httpx's MockTransport and an
elicitor that answers at once, on a real session, and read the card off the
session's stream as a host would.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx2 as httpx
import pytest

from anton.core.interaction.elicit import AskAnswer, AskRequest
from anton.core.interaction.emitter import TurnEmitter
from anton.core.llm.provider import StreamAskUser
from anton.core.settings import CoreSettings
from anton.core.tools.registry import ToolOutcome
from anton.core.utils import write_hold
from anton.core.utils.write_hold import WriteRelay, approve_held_write, parse_relay_hosts, relayed_host

RELAY_URL = "http://write-relay:8090"
PENDING = "0123456789abcdef0123456789abcdef"
SHA = "5e" * 32
HELD_ID = f"write:held:{PENDING}:{SHA}"
HOST = "gw.example.internal:8765"
# What the relay renders, with what the cell gate's card or a scrubber would
# change: an HTML comment, a backtick run, a `$`, a token-shaped value, a
# non-ASCII letter and trailing spaces. The tool must change none of it.
CARD = (
    "The agent's request waits at the write relay. Approve sends it once, exactly as shown. Cancel, "
    "Skip or no answer drops it, and nothing is sent.\n\n"
    "````text\n"
    f"POST http://{HOST}/api/orders\n"
    "content-type: application/json\n"
    "authorization: (hidden)\n"
    "\n"
    '{"qty": 5, "note": "<!-- ``` $x$ café", '
    '"token": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJlc2lnbmF0dXJl"}   \n'
    "````"
)
SENT = {"status": 201, "body": '{"order": 9}', "truncated": False, "content_type": "application/json"}

APPROVE = AskAnswer(status="answered", values=("approve",))
CANCEL = AskAnswer(status="answered", values=("cancel",))
SKIP = AskAnswer(status="cancelled")
TIMEOUT = AskAnswer(status="timeout")

# Polls fast enough for a test: the tool reads every 0.5 s for 90 s.
FAST = {"poll_interval_s": 0.01, "poll_limit_s": 0.1}


def _held(state: str, **over: object) -> dict:
    """The relay's `GET /held/<id>` answer."""
    return {
        "pending_id": PENDING, "sha256": SHA, "host": HOST, "method": "POST", "card_text": CARD,
        "state": state, "expires_at": 1_900_000_000, "result": SENT if state == "sent" else None, **over,
    }


@dataclass
class _Relay:
    """The relay's control address. Each `GET /held/<id>` answers the next of
    `reads`, the last one repeating; a status int answers that status, a dict
    the held request. `POST /held/<id>/cancel` answers `cancel`, and a 409
    moves later reads to the state it names."""

    reads: list[dict | int]
    cancel: tuple[int, dict] = (200, {"state": "cancelled"})
    requests: list[httpx.Request] = field(default_factory=list)
    # The control addresses the tool built a client for (`relay_for_tool`).
    urls: list[str] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST" and request.url.path == f"/held/{PENDING}/cancel":
            status, body = self.cancel
            if status == 409:
                self.reads = [_held(body["state"])]
            return httpx.Response(status, json=body)
        if request.method == "GET" and request.url.path == f"/held/{PENDING}":
            answer = self.reads.pop(0) if len(self.reads) > 1 else self.reads[0]
            if isinstance(answer, int):
                return httpx.Response(answer, json={"error": "unknown"})
            return httpx.Response(200, json=answer)
        return httpx.Response(404, json={"error": "unknown"})

    def client(self) -> WriteRelay:
        return WriteRelay(RELAY_URL, transport=httpx.MockTransport(self.handle))

    @property
    def gets(self) -> int:
        return sum(1 for r in self.requests if r.method == "GET")

    @property
    def cancels(self) -> int:
        return sum(1 for r in self.requests if r.method == "POST")


class _Elicitor:
    """Answers every question with `answer` as soon as it is asked, or raises `raises`."""

    supported_kinds = ("choice",)
    answer_hint = "hint"
    timeout_s = 300

    def __init__(self, *, answer: AskAnswer | None = None, raises: Exception | None = None) -> None:
        self.answer = answer
        self.raises = raises
        self.requests: list[AskRequest] = []
        self.question_ids: list[str] = []

    async def begin(self, question_id: str, request: AskRequest) -> None:
        pass

    async def ask(self, question_id: str, request: AskRequest) -> AskAnswer:
        self.requests.append(request)
        self.question_ids.append(question_id)
        if self.raises is not None:
            raise self.raises
        return self.answer

    async def end(self, question_id: str) -> None:
        pass


@pytest.fixture
async def held_session(make_session):
    """A session with the relay set and `elicitor` asking, its stream open."""
    made = []

    def _factory(elicitor: _Elicitor | None, *, write_approval: bool = False):
        session = make_session(
            settings=CoreSettings(write_relay_url=RELAY_URL, write_approval=write_approval), elicitor=elicitor,
        )
        session.emitter = TurnEmitter()
        made.append(session)
        return session

    yield _factory
    for session in made:
        await session.close()


def _cards(session) -> list[StreamAskUser]:
    """The cards the session streamed, as the host reads them."""
    events = []
    while not session.emitter.empty():
        events.append(session.emitter.get_nowait())
    return [e for e in events if isinstance(e, StreamAskUser)]


async def _approve(session, relay: _Relay) -> ToolOutcome:
    async with relay.client() as client:
        return await approve_held_write(session, client, pending_id=PENDING, **FAST)


# ── The card ──────────────────────────────────────────────────────────────

async def test_the_card_is_the_relay_s_text_exactly_under_the_held_request_s_id(held_session):
    elicitor = _Elicitor(answer=CANCEL)
    session = held_session(elicitor)
    await _approve(session, _Relay(reads=[_held("pending")]))
    [card] = _cards(session)
    assert card.request.prompt == CARD, "the recording proxy compares it with the relay's text byte for byte"
    assert card.id == HELD_ID == elicitor.question_ids[0]
    assert [o.value for o in card.request.options] == ["approve", "cancel"]
    assert card.request.allow_custom is False, "a typed answer approves nothing"


async def test_approve_waits_for_the_send_and_returns_the_host_s_answer(held_session):
    relay = _Relay(reads=[_held("pending"), _held("pending"), _held("sending"), _held("sent")])
    outcome = await _approve(held_session(_Elicitor(answer=APPROVE)), relay)
    assert outcome.ok is True and outcome.reason == ""
    assert outcome.content == (
        f"The user approved the held POST request to {HOST}. The write relay sent it once, and the host "
        'answered 201 (application/json):\n\n{"order": 9}'
    )
    assert relay.gets == 4 and relay.cancels == 0


async def test_approve_reports_a_cut_answer(held_session):
    cut = {**SENT, "status": 200, "body": "x" * 10, "truncated": True}
    relay = _Relay(reads=[_held("pending"), _held("sent", result=cut)])
    outcome = await _approve(held_session(_Elicitor(answer=APPROVE)), relay)
    assert "the host answered 200 (application/json, cut at 16 KiB)" in outcome.content


async def test_approve_with_no_approval_reaching_the_relay_cancels_and_sends_nothing(held_session):
    relay = _Relay(reads=[_held("pending")])
    outcome = await _approve(held_session(_Elicitor(answer=APPROVE)), relay)
    assert outcome.ok is True
    assert "no approval reached the write relay within 0.1 seconds, so anton cancelled the held request. " \
           "Nothing was sent." in outcome.content
    assert relay.cancels == 1


async def test_an_approval_that_lands_as_anton_cancels_is_reported_as_sent(held_session):
    relay = _Relay(reads=[_held("pending")], cancel=(409, {"state": "sent"}))
    outcome = await _approve(held_session(_Elicitor(answer=APPROVE)), relay)
    assert "The write relay sent it once, and the host answered 201" in outcome.content
    assert relay.cancels == 1


async def test_approve_with_the_send_still_running_at_the_limit_says_the_outcome_is_unconfirmed(held_session):
    relay = _Relay(reads=[_held("pending"), _held("sending")])
    outcome = await _approve(held_session(_Elicitor(answer=APPROVE)), relay)
    assert "hasn't confirmed the outcome. The write records will show it. Don't send it again." in outcome.content
    assert relay.cancels == 0, "a request the relay is sending can't be cancelled"


@pytest.mark.parametrize("state, says", [
    ("failed", "couldn't reach the host, so nothing was sent."),
    ("unknown", "it isn't known whether the host made the change"),
    ("cancelled", "It was cancelled before it was sent. Nothing was sent."),
    ("expired", "It expired before it was sent. Nothing was sent."),
])
async def test_approve_reports_each_way_the_relay_finishes(held_session, state, says):
    relay = _Relay(reads=[_held("pending"), _held(state)])
    outcome = await _approve(held_session(_Elicitor(answer=APPROVE)), relay)
    assert outcome.ok is True and says in outcome.content
    assert relay.cancels == 0


async def test_a_missed_read_while_waiting_is_read_again(held_session):
    relay = _Relay(reads=[_held("pending"), 503, _held("sent")])
    outcome = await _approve(held_session(_Elicitor(answer=APPROVE)), relay)
    assert "the host answered 201" in outcome.content


async def test_a_request_the_relay_forgets_while_waiting_is_unconfirmed(held_session):
    relay = _Relay(reads=[_held("pending"), 404])
    outcome = await _approve(held_session(_Elicitor(answer=APPROVE)), relay)
    assert "The write records show whether it was sent. Don't send it again" in outcome.content


# ── No approval ───────────────────────────────────────────────────────────

def _refused(why: str) -> str:
    return (
        f"Not sent: the held POST request to {HOST} needs the user's approval, and {why}. Nothing was sent. "
        "The user didn't approve this change, so it must not be made another way: don't send the request "
        "again, rewrite the cell, build the URL at run time, or use another tool for it. Tell the user what "
        "the request would change and that it needs their approval."
    )


@pytest.mark.parametrize("elicitor, why", [
    (_Elicitor(answer=CANCEL), "they declined it"),
    (_Elicitor(answer=SKIP), "they declined it"),
    (_Elicitor(answer=AskAnswer(status="answered", text="yes")), "they declined it"),
    (_Elicitor(answer=TIMEOUT), "no one answered in time"),
    (_Elicitor(raises=RuntimeError("broker down")), "the approval request failed"),
], ids=["cancel", "skip", "typed", "timeout", "error"])
async def test_no_approve_cancels_the_held_request_and_never_waits(held_session, elicitor, why):
    relay = _Relay(reads=[_held("pending")])
    outcome = await _approve(held_session(elicitor), relay)
    assert outcome == ToolOutcome(content=_refused(why), ok=True, reason="write_not_approved")
    assert relay.cancels == 1
    assert relay.gets == 1, "nothing polls for a send that can't come"


async def test_an_earlier_unanswered_card_refuses_without_asking(held_session):
    elicitor = _Elicitor(answer=APPROVE)
    session = held_session(elicitor, write_approval=True)
    session._write_gate.unanswered = True
    relay = _Relay(reads=[_held("pending")])
    outcome = await _approve(session, relay)
    assert outcome.content == _refused("an earlier approval request in this turn went unanswered")
    assert elicitor.requests == [] and _cards(session) == []
    assert relay.cancels == 1 and relay.gets == 1


async def test_a_held_card_that_times_out_refuses_the_turn_s_later_cards(held_session):
    session = held_session(_Elicitor(answer=TIMEOUT), write_approval=True)
    await _approve(session, _Relay(reads=[_held("pending")]))
    assert session._write_gate.unanswered is True


async def test_no_way_to_ask_cancels_and_refuses(held_session):
    session = held_session(_Elicitor(answer=APPROVE))
    session.emitter = None
    relay = _Relay(reads=[_held("pending")])
    outcome = await _approve(session, relay)
    assert outcome.content == _refused("there is no way to ask them here")
    assert relay.cancels == 1


async def test_a_cancel_the_relay_doesn_t_take_still_refuses(held_session, caplog):
    relay = _Relay(reads=[_held("pending")], cancel=(500, {"error": "boom"}))
    with caplog.at_level(logging.WARNING, logger="anton.core.utils.write_hold"):
        outcome = await _approve(held_session(_Elicitor(answer=CANCEL)), relay)
    assert outcome.content == _refused("they declined it")
    assert f"write approval: held request {PENDING}: the cancel failed (the relay answered 500)" in caplog.messages


# ── What the relay says before any card ───────────────────────────────────

async def test_an_unknown_held_request_is_reported_with_no_card(held_session):
    elicitor = _Elicitor(answer=APPROVE)
    relay = _Relay(reads=[404])
    outcome = await _approve(held_session(elicitor), relay)
    assert outcome == ToolOutcome(
        content="That held request is unknown or expired; nothing was sent.", ok=True,
    )
    assert elicitor.requests == [] and relay.cancels == 0


@pytest.mark.parametrize("state, says", [
    ("sent", "The write relay sent it once, and the host answered 201"),
    ("sending", "The write relay is sending it but hasn't confirmed the outcome."),
    ("failed", "couldn't reach the host, so nothing was sent."),
    ("unknown", "it isn't known whether the host made the change"),
    ("cancelled", "It was cancelled before it was sent. Nothing was sent."),
    ("expired", "It expired before it was sent. Nothing was sent."),
])
async def test_a_held_request_no_longer_pending_is_reported_with_no_card(held_session, state, says):
    elicitor = _Elicitor(answer=APPROVE)
    relay = _Relay(reads=[_held(state)])
    outcome = await _approve(held_session(elicitor), relay)
    assert outcome.ok is True
    assert outcome.content.startswith(f"The held POST request to {HOST} isn't waiting for approval any more. ")
    assert says in outcome.content
    assert elicitor.requests == [] and relay.cancels == 0


@pytest.mark.parametrize("answer", [
    500,
    _held("pending", pending_id="f" * 32),
    _held("pending", sha256=SHA.upper()),
    _held("pending", sha256="5e" * 31),
    {k: v for k, v in _held("pending").items() if k != "card_text"},
    _held("waiting"),
], ids=["error-status", "another-id", "upper-case-hash", "short-hash", "no-card-text", "unknown-state"])
async def test_an_answer_that_isn_t_a_held_request_raises_no_card(held_session, answer):
    elicitor = _Elicitor(answer=APPROVE)
    relay = _Relay(reads=[answer])
    outcome = await _approve(held_session(elicitor), relay)
    assert outcome.ok is True
    assert outcome.content.startswith("Not asked: anton couldn't read the held request from the write relay")
    assert "Nothing was sent" in outcome.content
    assert elicitor.requests == [], "a card the proxy can't name, or for another request, is never shown"


async def test_an_unreachable_relay_raises_no_card(held_session):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    elicitor = _Elicitor(answer=APPROVE)
    async with WriteRelay(RELAY_URL, transport=httpx.MockTransport(refuse)) as client:
        outcome = await approve_held_write(held_session(elicitor), client, pending_id=PENDING, **FAST)
    assert "(the relay could not be reached (ConnectError))" in outcome.content
    assert elicitor.requests == []


async def test_an_answer_longer_than_anton_reads_raises_no_card(held_session, monkeypatch):
    monkeypatch.setattr(write_hold, "MAX_REPLY_BYTES", 64)
    elicitor = _Elicitor(answer=APPROVE)
    outcome = await _approve(held_session(elicitor), _Relay(reads=[_held("pending")]))
    assert "the relay answered more than anton reads" in outcome.content
    assert elicitor.requests == []


# ── The tool ──────────────────────────────────────────────────────────────

def _tool_names(session) -> set[str]:
    session._build_tools()
    return {t.name for t in session.tool_registry.get_tool_defs()}


async def test_the_tool_is_offered_with_the_relay_and_a_way_to_ask(make_session):
    session = make_session(settings=CoreSettings(write_relay_url=RELAY_URL), elicitor=_Elicitor())
    assert "approve_held_write" in _tool_names(session)
    [tool] = [t for t in session.tool_registry.get_tool_defs() if t.name == "approve_held_write"]
    assert tool.input_schema["required"] == ["pending_id"]
    assert "428" in tool.description and "held_for_approval" in tool.description
    assert "nothing was written" in tool.description and "A person must approve" in tool.description


class _PathOnly(_Elicitor):
    supported_kinds = ("path",)


@pytest.mark.parametrize("settings, elicitor", [
    (CoreSettings(), _Elicitor()),
    (CoreSettings(write_relay_url="  "), _Elicitor()),
    (CoreSettings(write_relay_url=RELAY_URL), None),
    (CoreSettings(write_relay_url=RELAY_URL), _PathOnly()),
], ids=["no-relay", "blank-relay", "no-elicitor", "no-choice-cards"])
async def test_the_tool_is_absent_without_the_relay_or_a_way_to_ask(make_session, settings, elicitor):
    session = make_session(settings=settings, elicitor=elicitor)
    assert "approve_held_write" not in _tool_names(session)


async def test_the_relay_url_setting_reads_its_environment_variable(make_session, monkeypatch):
    monkeypatch.setenv("ANTON_WRITE_RELAY_URL", RELAY_URL)
    session = make_session(settings=CoreSettings(), elicitor=_Elicitor())
    assert session._write_relay_url == RELAY_URL
    assert "approve_held_write" in _tool_names(session)


@pytest.fixture
def relay_for_tool(monkeypatch):
    """The relay the tool builds from the session's setting, on MockTransport."""
    relay = _Relay(reads=[_held("pending")])

    def build(url: str) -> WriteRelay:
        relay.urls.append(url)
        return WriteRelay(url, transport=httpx.MockTransport(relay.handle))

    monkeypatch.setattr(write_hold, "WriteRelay", build)
    return relay


async def test_the_tool_asks_through_the_session_s_relay(held_session, relay_for_tool):
    elicitor = _Elicitor(answer=CANCEL)
    session = held_session(elicitor)
    session._build_tools()
    outcome = await session.tool_registry.dispatch_tool(session, "approve_held_write", {"pending_id": PENDING})
    assert outcome.reason == "write_not_approved" and outcome.ok is True
    assert relay_for_tool.urls == [RELAY_URL]
    assert [str(r.url) for r in relay_for_tool.requests] == [
        f"{RELAY_URL}/held/{PENDING}", f"{RELAY_URL}/held/{PENDING}/cancel",
    ]
    assert elicitor.question_ids == [HELD_ID]


@pytest.mark.parametrize("tc_input", [
    {"pending_id": PENDING.upper()},
    {"pending_id": PENDING[:-1]},
    {"pending_id": PENDING + "0"},
    {"pending_id": PENDING + "\n"},
    {"pending_id": f"../{PENDING[3:]}"},
    {"pending_id": f"{PENDING[:16]}/cancel"},
    {"pending_id": int("1" * 32)},
    {"pending_id": ""},
    {},
], ids=["upper-case", "short", "long", "newline", "dot-dot", "slash", "number", "empty", "missing"])
async def test_a_malformed_pending_id_sends_no_request(held_session, relay_for_tool, tc_input):
    elicitor = _Elicitor(answer=APPROVE)
    session = held_session(elicitor)
    session._build_tools()
    outcome = await session.tool_registry.dispatch_tool(session, "approve_held_write", tc_input)
    assert outcome.ok is True and outcome.reason == ""
    assert "pending_id must be the 32 lowercase hex characters" in outcome.content
    assert relay_for_tool.requests == [] and relay_for_tool.urls == [], "the model's text never reaches the relay"
    assert elicitor.requests == []


# ── The relay's hosts ─────────────────────────────────────────────────────

def test_relay_hosts_are_lowercased_with_port_80_by_default(caplog):
    with caplog.at_level(logging.WARNING, logger="anton.core.utils.write_hold"):
        hosts = parse_relay_hosts("GW.example.internal:8765,\nplanner:80, other ,, h:0, h:65536, [::1]:80, a/b")
    assert hosts == {"gw.example.internal:8765", "planner:80", "other:80"}
    assert caplog.messages == [
        f"write approval: skipped the write relay host entry {entry!r}: an entry is host:port"
        for entry in ("h:0", "h:65536", "[::1]:80", "a/b")
    ]


@pytest.mark.parametrize("url, host", [
    ("http://gw:8765/api", "gw:8765"),
    ("HTTP://GW:8765?q=1", "gw:8765"),
    ("http://gw:8765", "gw:8765"),
    ("http://gw/api", "gw:80"),
    ("http://gw:8765#top", "gw:8765"),
    ("https://gw:8765/api", None),
    ("http://user@gw:8765/api", None),
    ("http://gw:8765@evil/api", None),
    ("http://evil\\@gw:8765/api", None),
    ("http://{host}:8765/api", None),
    ("http://gw:{port}/api", None),
    ("http://g%77:8765/api", None),
    ("http://gw:8765\\api", None),
    (" http://gw:8765/api", None),
    ("{base}/api", None),
])
def test_only_a_plain_http_url_names_a_relayed_host(url, host):
    assert relayed_host(url) == host


def test_the_card_text_has_every_character_a_scrubber_or_the_gate_s_card_would_change():
    """The test card is worth its name: the cell gate's fence and the output
    scrubber would each change it, so a tool that ran either fails above."""
    from anton.core.utils.scratchpad import _fence
    from anton.utils.datasources import scrub_output

    assert scrub_output(CARD) != CARD
    assert _fence(CARD) != CARD
