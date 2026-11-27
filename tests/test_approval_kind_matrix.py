"""Which cards reach which elicitor: each source of a card in `SOURCES`
against no elicitor, an approvals-only elicitor and a "choice" elicitor.

An approval card (`ApprovalRequest`) reaches an elicitor that lists "choice"
or APPROVAL_KIND. Every other question reaches a "choice" elicitor only. So a
host that lists APPROVAL_KIND without "choice", for a turn nobody is known to
be watching, keeps the cards that approve a write and refuses the rest at
once, before any card is published.

Each source runs the real code that raises its card, through the real
`elicit()`. The two that the model reaches through a tool (`ask_user` and
`approve_held_write`) go through the session's tool registry, so a tool the
session doesn't offer counts as refused.

The browser raises its step card in two places, and each has its own row. The
router's browser (a pod turn) holds a commit step itself, and the tool asks
once the router answers `approval_required`. The desktop browser (an
in-process turn) holds nothing, so the tool asks before it sends the step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import httpx2 as httpx
import pytest

from anton.core.browser import tool as browser_tool
from anton.core.browser.config import BrowserConfig
from anton.core.interaction.elicit import APPROVAL_KIND, AskAnswer, AskRequest
from anton.core.interaction.emitter import TurnEmitter
from anton.core.llm.provider import StreamAskUser
from anton.core.settings import CoreSettings
from anton.core.utils import write_hold
from anton.core.utils.scratchpad import hold_unapproved_write
from anton.core.utils.write_hold import WriteRelay

from tests.test_browser_tool import CHECKOUT, LOOPBACK, SID, FakeHost, asking_session
from tests.test_write_approval_gate import UPDATE_CELL, _gated_session, _Pad
from tests.test_write_hold import PENDING, RELAY_URL, _held, _Relay

ELICITORS: dict[str, tuple[str, ...] | None] = {
    "none": None,
    "approval": (APPROVAL_KIND,),
    "choice": ("choice",),
}

# The elicitors each source's card reaches; every other one refuses it at once.
REACHES: dict[str, frozenset[str]] = {
    "write gate card": frozenset({"approval", "choice"}),
    "write relay held card": frozenset({"approval", "choice"}),
    "router browser held card": frozenset({"approval", "choice"}),
    "desktop browser card": frozenset({"approval", "choice"}),
    "ask_user question": frozenset({"choice"}),
    "browser hand-over": frozenset({"choice"}),
    "past-limit question": frozenset({"choice"}),
}

# How the id of a source's card starts, where that names the path that raised
# it. The router's held step card carries the browser's ids; the desktop card
# gets the fresh `write:` id `ask_approval_choice` gives a card that names none.
CARD_IDS: dict[str, str] = {
    "write relay held card": "write:held:",
    "router browser held card": "browser:approve:",
    "desktop browser card": "write:",
}


class _Skips:
    """Takes `kinds` and answers every card at once with Skip, recording each
    question id."""

    answer_hint = "hint"
    timeout_s = 300

    def __init__(self, kinds: tuple[str, ...]) -> None:
        self.supported_kinds = kinds
        self.question_ids: list[str] = []

    async def begin(self, question_id: str, request: AskRequest) -> None:
        pass

    async def ask(self, question_id: str, request: AskRequest) -> AskAnswer:
        self.question_ids.append(question_id)
        return AskAnswer(status="cancelled")

    async def end(self, question_id: str) -> None:
        pass


@dataclass
class _Seen:
    """What one source did: the question ids the elicitor was asked, and the
    ids of the cards published on the turn's stream."""

    asked: list[str]
    published: list[str]


@dataclass
class _Context:
    make_session: Callable
    monkeypatch: pytest.MonkeyPatch
    tmp_path: Path


def _asked(elicitor: _Skips | None) -> list[str]:
    return [] if elicitor is None else list(elicitor.question_ids)


def _published(emitter: TurnEmitter) -> list[str]:
    ids = []
    while not emitter.empty():
        event = emitter.get_nowait()
        if isinstance(event, StreamAskUser):
            ids.append(event.id)
    return ids


def _offered(session) -> set[str]:
    session._build_tools()
    return {tool.name for tool in session.tool_registry.get_tool_defs()}


async def _write_gate_card(context: _Context, elicitor: _Skips | None) -> _Seen:
    session = await _gated_session(context.tmp_path)
    session.elicitor = elicitor
    await hold_unapproved_write(session, pad=_Pad(), code=UPDATE_CELL)
    return _Seen(asked=_asked(elicitor), published=_published(session.emitter))


async def _write_relay_held_card(context: _Context, elicitor: _Skips | None) -> _Seen:
    relay = _Relay(reads=[_held("pending")])
    context.monkeypatch.setattr(
        write_hold, "WriteRelay", lambda url: WriteRelay(url, transport=httpx.MockTransport(relay.handle)),
    )
    session = context.make_session(settings=CoreSettings(write_relay_url=RELAY_URL), elicitor=elicitor)
    session.emitter = TurnEmitter()
    try:
        if "approve_held_write" in _offered(session):
            await session.tool_registry.dispatch_tool(session, "approve_held_write", {"pending_id": PENDING})
        return _Seen(asked=_asked(elicitor), published=_published(session.emitter))
    finally:
        await session.close()


async def _router_browser_held_card(context: _Context, elicitor: _Skips | None) -> _Seen:
    session = asking_session(FakeHost(pages=[CHECKOUT]), elicitor)
    await browser_tool.handle_browser(session, {"action": "act", "step": {"action": "click", "element": 7}})
    published = [event.id for event in session.events if isinstance(event, StreamAskUser)]
    return _Seen(asked=_asked(elicitor), published=published)


async def _desktop_browser_card(context: _Context, elicitor: _Skips | None) -> _Seen:
    desktop = BrowserConfig(base_url=LOOPBACK, profile=SID, token="k")
    session = asking_session(FakeHost(pages=[CHECKOUT]), elicitor, config=desktop)
    await browser_tool.handle_browser(session, {"action": "act", "step": {"action": "click", "element": 7}})
    published = [event.id for event in session.events if isinstance(event, StreamAskUser)]
    return _Seen(asked=_asked(elicitor), published=published)


async def _ask_user_question(context: _Context, elicitor: _Skips | None) -> _Seen:
    session = context.make_session(elicitor=elicitor)
    session.emitter = TurnEmitter()
    tc_input = {
        "question": "Which database should I read from?",
        "options": [{"value": "pg", "label": "postgres"}, {"value": "my", "label": "mysql"}],
    }
    try:
        if "ask_user" in _offered(session):
            await session.tool_registry.dispatch_tool(session, "ask_user", tc_input)
        return _Seen(asked=_asked(elicitor), published=_published(session.emitter))
    finally:
        await session.close()


async def _browser_hand_over(context: _Context, elicitor: _Skips | None) -> _Seen:
    session = asking_session(FakeHost(), elicitor)
    await browser_tool.handle_browser(session, {"action": "hand_to_user", "reason": "Sign in"})
    published = [event.id for event in session.events if isinstance(event, StreamAskUser)]
    return _Seen(asked=_asked(elicitor), published=published)


async def _past_limit_question(context: _Context, elicitor: _Skips | None) -> _Seen:
    session = context.make_session(elicitor=elicitor)
    session.emitter = TurnEmitter()
    published = []
    decision = session._ask_to_continue_past_limit("spend_ceiling", "")
    try:
        async for kind, item in decision:
            if kind == "event" and isinstance(item, StreamAskUser):
                published.append(item.id)
    finally:
        await decision.aclose()
        await session.close()
    return _Seen(asked=_asked(elicitor), published=published)


SOURCES: dict[str, Callable[[_Context, _Skips | None], Awaitable[_Seen]]] = {
    "write gate card": _write_gate_card,
    "write relay held card": _write_relay_held_card,
    "router browser held card": _router_browser_held_card,
    "desktop browser card": _desktop_browser_card,
    "ask_user question": _ask_user_question,
    "browser hand-over": _browser_hand_over,
    "past-limit question": _past_limit_question,
}


def test_the_table_names_every_source():
    assert set(REACHES) == set(SOURCES)


@pytest.mark.parametrize("elicitor_name", list(ELICITORS))
@pytest.mark.parametrize("source", list(SOURCES))
async def test_which_cards_reach_which_elicitor(make_session, monkeypatch, tmp_path, source, elicitor_name):
    kinds = ELICITORS[elicitor_name]
    elicitor = None if kinds is None else _Skips(kinds)
    seen = await SOURCES[source](_Context(make_session, monkeypatch, tmp_path), elicitor)

    reaches = elicitor_name in REACHES[source]
    assert (bool(seen.asked), bool(seen.published)) == (reaches, reaches), seen
    if reaches:
        assert seen.asked == seen.published, "the card the host sees is the question the elicitor waits on"
    if reaches and source in CARD_IDS:
        assert seen.asked[0].startswith(CARD_IDS[source]), "the card comes from the path its row names"
