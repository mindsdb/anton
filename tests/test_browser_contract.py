"""The browser contract fixtures, held against the code that reads and writes them.

tests/fixtures/browser_contract pins what the browser tool sends, what both
browser hosts answer, the turn request's browser block and the cloud turn's
session line. The hosts' tests keep copies of the same files, so every
example here is read by the anton code it describes: answers by the models
the client parses them with, requests by the client calls that send them.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import httpx2 as httpx
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from anton.core.browser import client as client_module
from anton.core.browser import tool as browser_tool
from anton.core.browser.client import BrowserClient, BrowserError
from anton.core.browser.config import BrowserConfig
from anton.core.browser.models import (
    ActionResult,
    Control,
    Element,
    ErrorBody,
    Frame,
    HeldStep,
    Page,
    SavedTabAnswer,
    SavedTabsAnswer,
    SessionInfo,
    SiteAnswer,
    SitesAnswer,
    TabAnswer,
    TabsAnswer,
)
from anton.core.browser.page import FRAME_TEXT_HEADING, PAGE_TEXT_HEADING, TEXT_PER_FRAME, format_page, human_check

FIXTURES = Path(__file__).parent / "fixtures" / "browser_contract"
ROUTER = "https://app.example.test/browser"
PROFILE = "t-0123456789abcdef01234567"
PROJECT = "p-89abcdef0123456789abcdef"


class ContractExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    where: str
    kind: Literal["request", "response", "event", "turn_block", "constructor", "classification"]
    status: int | None = None
    refused: bool = False
    body: dict


class ContractFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    limits: dict[str, int | list[str]] = {}
    examples: dict[str, ContractExample]


def _load(name: str) -> ContractFile:
    return ContractFile.model_validate_json((FIXTURES / f"{name}.json").read_text())


def _examples(kind: str) -> list[tuple[str, str, ContractExample]]:
    found = []
    for path in sorted(FIXTURES.glob("*.json")):
        for name, example in _load(path.stem).examples.items():
            if example.kind == kind:
                found.append((path.stem, name, example))
    return found


def test_the_contract_is_exactly_these_files():
    names = sorted(p.stem for p in FIXTURES.glob("*.json"))
    assert names == sorted([
        "action", "browser_config", "commit_steps", "control", "error", "page", "saved_tabs", "session_event",
        "sessions", "sites", "tabs",
    ])
    for name in names:
        assert _load(name).examples, name


#: The dotted names the descriptions use. Any other word, a period and a
#: word with no space between them is two sentences run together.
_DOTTED_NAMES = {"params.browser", "BrowserConfig.to_dict", "BrowserConfig.from_dict", "limits.saved_tabs_per_project"}


def test_no_description_runs_two_sentences_together():
    for path in sorted(FIXTURES.glob("*.json")):
        joined = set(re.findall(r"[A-Za-z_]+\.[A-Za-z_]+", _load(path.stem).description)) - _DOTTED_NAMES
        assert not joined, path.name


def test_every_host_keeps_this_folder_byte_for_byte_readme_included():
    """The README is one of the files all three copies share byte for byte,
    so it says so, and a host keeps its own notes outside its copy."""
    readme = " ".join((FIXTURES / "README.md").read_text().split())
    assert "Each host's tests keep a copy of these files" not in readme
    assert "Each host's tests keep a byte-for-byte copy of this folder, this README included" in readme
    assert "a host keeps its own notes outside its copy" in readme


# ── answers ────────────────────────────────────────────────────────────────

class _OpenedSession(BaseModel):
    """``POST /sessions``'s answer. Anton discards it, so this pins the one
    shape both hosts give, url always present and null before a page loads."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    resumed: bool
    url: str | None


#: The model the client reads each route's answer with. A DELETE answer's
#: body is never read.
ANSWER_MODELS: dict[str, type[BaseModel]] = {
    "POST /sessions": _OpenedSession,
    "GET /sessions/{id}/page": Page,
    "POST /sessions/{id}/action": ActionResult,
    "GET /sessions/{id}": SessionInfo,
    "POST /sessions/{id}/control": Control,
    "GET /sessions/{id}/tabs": TabsAnswer,
    "POST /sessions/{id}/tabs": TabAnswer,
    "POST /sessions/{id}/tabs/{tab}/activate": TabAnswer,
    "GET /sessions/{id}/sites": SitesAnswer,
    "POST /sessions/{id}/sites": SiteAnswer,
    "GET /saved-tabs?project={project}": SavedTabsAnswer,
    "POST /saved-tabs": SavedTabAnswer,
}


@pytest.mark.parametrize(
    "example", [e for _, _, e in _examples("response") if e.status is not None and e.status < 400],
    ids=lambda e: e.where,
)
def test_every_answer_parses_into_the_model_the_client_reads_it_with(example):
    created = example.where == "POST /sessions" and not example.body["resumed"]
    assert example.status == (201 if created else 200)
    if example.where.startswith("DELETE "):
        assert example.body == {"ok": True}
        return
    ANSWER_MODELS[example.where].model_validate(example.body)


def test_the_page_fixture_reads_as_the_model_sees_it():
    pages = _load("page").examples
    signin = Page.model_validate(pages["a sign-in page with a check frame"].body)
    cart = Page.model_validate(pages["a page with no check"].body)
    assert human_check(signin) == "Cloudflare"
    assert human_check(cart) == ""
    text = format_page(signin)
    assert '[3] textbox "Password" (password)' in text
    assert '[4] textbox "One-time code" (secret field)' in text
    assert '[2] textbox "Email" value="ada@example.test"' in text
    assert PAGE_TEXT_HEADING in text
    assert [el.id for el in signin.elements if el.secret] == [3, 4]
    assert "… more elements not shown" in format_page(cart)


def test_the_frame_fixture_reads_as_the_model_sees_it():
    """A read frame's controls carry no number, its text sits under its own
    untrusted heading, an unread frame is one line, and all frames together
    show at most FRAMES_TEXT_CHARS of text."""
    page = Page.model_validate(_load("page").examples["a page with a frame"].body)
    text = format_page(page)
    form = text.index('Frame "https://forms.example.test/newsletter" (no numbers: act by pixel):')
    assert text.index('- textbox "Email"', form) < text.index('- button "Subscribe"', form)
    heading = text.index(FRAME_TEXT_HEADING, form)
    assert text.index("Get the weekly digest. Email Subscribe", heading) == heading + len(FRAME_TEXT_HEADING) + 1
    assert 'Frame "https://js.stripe.com/v3/elements-inner-card" (not read)' in text
    assert 'Frame "about:srcdoc" (not read)' in text
    # Frame controls are reached only by position, so none carries a number.
    assert not [line for line in text.splitlines() if line.startswith("- [")]
    assert all(not hasattr(el, "id") for frame in page.frames for el in frame.elements)
    # The sign-in page lists its check frame under frames, not as an element.
    signin = Page.model_validate(_load("page").examples["a sign-in page with a check frame"].body)
    assert [f.read for f in signin.frames] == [False] and len({el.id for el in signin.elements}) == len(signin.elements)

    many = page.model_copy(update={"frames": [
        Frame(url=f"https://forms.example.test/form-{i}", read=True, text=letter * 1000) for i, letter in enumerate("QXZ")
    ]})
    shown = format_page(many)
    assert "Q" * 1000 in shown and "X" * 1000 in shown and "Z" not in shown
    assert 'Frame "https://forms.example.test/form-2" (no numbers: act by pixel):' in shown


def test_the_model_reads_as_much_of_a_frame_as_a_host_sends():
    assert TEXT_PER_FRAME == _load("page").limits["text_per_frame"]


def test_the_page_fixture_names_every_frame_limit_the_hosts_hold():
    """Each host pins its frame constants against these, the time a host
    gives a read's frames included."""
    limits = _load("page").limits
    assert set(limits) == {
        "frames_read", "elements_per_frame", "text_per_frame", "frame_min_read_px", "frames_phase_ms",
        "payment_frame_hosts", "sign_in_frame_hosts",
    }
    assert limits["frames_phase_ms"] == 2000
    assert "frames_phase_ms" in _load("page").description


def test_the_page_fixture_gives_the_element_order_both_hosts_use():
    """Past the cap both hosts list what the person sees first. One lists an
    open dialog's elements ahead of everything, even when all fit, and the
    other keeps page order when all fit, so the fixture allows both."""
    description = _load("page").description
    assert "It keeps page order when every shown element fits" not in description
    assert "Past the cap it lists the on-screen elements ahead of the off-screen ones, each group in page order" in description
    assert "when every shown element fits, a host may keep page order or use the same order" in description
    assert "A host may list an open dialog's elements (in_dialog) ahead of both groups." in description


def test_a_page_from_a_host_that_predates_frames_says_nothing_about_frames():
    """A host that sends no ``frames`` hasn't said the page has none."""
    body = {key: value for key, value in _load("page").examples["a page with no check"].body.items() if key != "frames"}
    page = Page.model_validate(body)
    assert page.frames is None
    assert "Frame " not in format_page(page)


def test_the_action_fixture_says_whether_the_step_changed_anything():
    actions = _load("action").examples
    assert ActionResult.model_validate(actions["a step that changed nothing"].body).changed is False
    assert ActionResult.model_validate(actions["the answer with the page after"].body).changed is True
    # An older host sends no `changed`: the tool can't tell.
    assert ActionResult.model_validate({"ok": True, "url": "https://shop.example.test/cart"}).changed is None


def test_the_action_fixture_says_a_step_that_changed_the_page_in_place_settles_too():
    """The settle both hosts run: no settle ends for quiet, or for a step
    that started nothing, inside the first settle_quiet_ms (the request
    window, for a debounced search); a step that started nothing answers at
    its end; one that started a request or a navigation, or finds the page
    aria-busy, waits for quiet up to settle_max_ms; one that only changed the
    page in place waits for quiet a moment at most. A navigation's cap is
    each host's own: the router's browser lets it run past settle_max_ms
    while the new page loads, and the desktop app's waits for a navigate
    step's load before the settle and caps every settle at settle_max_ms,
    or sooner inside the step's own time budget, so a navigate whose load
    took its full 30 s settles for about 2.5 s at most."""
    description = _load("action").description
    assert "answers at once" not in description
    assert "a host may first wait up to settle_quiet_ms" not in description
    for sentence in (
        "Neither host ends a settle for quiet, or for a step that started nothing, before settle_quiet_ms from its "
        "start, so a request the step starts by then (a search box's debounced suggestions) is waited for.",
        "A step that by then has changed nothing, started no main-frame request or navigation and left nothing "
        "shown marked aria-busy answers there.",
        "One that started a main-frame request or a navigation, or finds the page aria-busy, waits until those "
        "requests are done, nothing shown is marked aria-busy and the page has shown no change for "
        "settle_quiet_ms, at most settle_max_ms from the settle's start.",
        "One that only changed the page in place waits for that quiet at most settle_quiet_ms past the host's "
        "first look at the page after the step, so a page that changes on its own (a ticker) doesn't hold every "
        "step to the cap.",
        "A navigation's cap is each host's own.",
        "The router's browser waits up to 5 s each for a navigation to commit and for its new page's "
        "DOMContentLoaded (a navigate step's own load has waited up to 30 s for it), then for the new page's "
        "network to go idle too, and stops at the later of settle_max_ms from the settle's start and 1.5 s after "
        "that DOMContentLoaded wait, never past 31.5 s from the step's start.",
        "The desktop app's browser waits for a navigate, back, forward or reload step's page to finish loading, "
        "up to 30 s, before the settle, and stops every settle, a navigation's too, at settle_max_ms from its start "
        "or 0.5 s before the step's 33 s budget ends, whichever comes first.",
    ):
        assert sentence in description, sentence
    assert "a navigation's too, at settle_max_ms from its start." not in description


def test_the_control_fixture_tells_done_by_the_person_from_any_other_agent_record():
    controls = _load("control").examples
    fresh = SessionInfo.model_validate(controls["a session the agent holds"].body).control
    handed = Control.model_validate(controls["handed to the person"].body)
    done = SessionInfo.model_validate(controls["done by the person"].body).control
    assert (handed.holder, handed.by) == ("human", "agent")
    assert (done.holder, done.by) == ("agent", "person") and done.seq > handed.seq
    assert (fresh.holder, fresh.by) == ("agent", "agent") and fresh.seq < handed.seq
    assert SessionInfo.model_validate(controls["a session before a page loads"].body).url is None


def test_a_project_keeps_at_most_the_pinned_number_of_saved_tabs():
    saved = _load("saved_tabs")
    assert saved.limits == {"saved_tabs_per_project": 100}
    assert "100" in _load("error").examples["too_many_saved_tabs"].body["message"]


def test_a_held_step_reads_what_the_browser_says_it_holds():
    body = _load("error").examples["approval_required"].body
    assert set(body["held"]) == set(HeldStep.model_fields)
    held = ErrorBody.model_validate(body).held
    assert held == HeldStep(kind="commit", action="click", names=["Pay $42.00"], url="https://shop.example.test/cart")


def _held(example: ContractExample) -> str | None:
    body = example.body
    element = Element.model_validate({"id": 7, **body["element"]}) if body["element"] is not None else None
    return browser_tool._classify(body["step"], url=body["url"], requested=body.get("requested"), element=element)


@pytest.mark.parametrize("name", sorted(_load("commit_steps").examples))
def test_the_commit_list_holds_what_the_fixture_says(name):
    example = _load("commit_steps").examples[name]
    assert example.kind == "classification"
    assert _held(example) == example.body["held"], name


def test_the_commit_list_covers_every_word_and_both_holds():
    held = [e.body["held"] for e in _load("commit_steps").examples.values()]
    assert {"commit", "position", None} <= set(held)
    names = " ".join(
        (e.body["element"] or {}).get("name", "") for e in _load("commit_steps").examples.values() if e.body["held"] == "commit"
    ).lower()
    for word in browser_tool.COMMIT_WORDS:
        assert word in names, word


# ── requests ───────────────────────────────────────────────────────────────


class _Recorder:
    """A host that records each request and answers with any fixture answer
    the client can read, so every client call completes. ``page`` names the
    page example it reads back."""

    def __init__(self, page: str = "a page with no check"):
        self.sent: list[tuple[str, str, dict | None]] = []
        self.page = page

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path.removeprefix("/browser")
        self.sent.append((request.method, path, body))
        answers = {
            ("POST", f"/sessions/{PROFILE}/control"): _load("control").examples["handed to the person"].body,
            ("GET", f"/sessions/{PROFILE}"): _load("control").examples["a session the agent holds"].body,
            ("POST", f"/sessions/{PROFILE}/tabs"): _load("tabs").examples["the opened tab"].body,
            ("GET", f"/sessions/{PROFILE}/tabs"): _load("tabs").examples["the open tabs"].body,
            ("POST", f"/sessions/{PROFILE}/tabs/t2/activate"): _load("tabs").examples["the activated tab"].body,
            ("POST", f"/sessions/{PROFILE}/sites"): _load("sites").examples["the noted site"].body,
            ("GET", f"/sessions/{PROFILE}/sites"): _load("sites").examples["the sites on record"].body,
            ("POST", "/saved-tabs"): _load("saved_tabs").examples["the saved tab"].body,
            ("GET", "/saved-tabs"): _load("saved_tabs").examples["the project's saved tabs"].body,
            ("GET", f"/sessions/{PROFILE}/page"): _load("page").examples[self.page].body,
            ("POST", f"/sessions/{PROFILE}/action"): _load("action").examples["the answer with the page after"].body,
        }
        answer = answers.get((request.method, path), {"ok": True})
        return httpx.Response(200, json=answer, headers={"x-image-width": "1024", "x-image-height": "640"})


async def _k():
    return "k"


def _client(recorder: _Recorder) -> BrowserClient:
    return BrowserClient(
        base_url=ROUTER, credential=_k, trust_env=True, first_call_timeout_s=45.0,
        http=httpx.AsyncClient(transport=httpx.MockTransport(recorder.handler)),
    )


async def _every_client_call(client: BrowserClient) -> None:
    await client.open_session(session_id=PROFILE, url="https://shop.example.test/")
    await client.page(session_id=PROFILE)
    await client.act(session_id=PROFILE, step={"action": "click", "element": 7, "include_page": True})
    await client.screenshot(session_id=PROFILE)
    await client.sites(session_id=PROFILE)
    await client.note_site(session_id=PROFILE, site="github.com", account_hint="ada")
    await client.control(session_id=PROFILE)
    await client.hand_to_person(session_id=PROFILE, reason="Please sign in to Example Shop")
    await client.tabs(session_id=PROFILE)
    await client.new_tab(session_id=PROFILE, url="https://docs.example.test/")
    await client.activate_tab(session_id=PROFILE, tab_id="t2")
    await client.close_tab(session_id=PROFILE, tab_id="t2")
    await client.saved_tabs(project=PROJECT)
    await client.save_tab(project=PROJECT, name="Q3 pipeline report", url="https://crm.example.test/reports/q3", notes="start here")
    await client.delete_saved_tab(project=PROJECT, name="Q3 pipeline report")
    await client.close_session(session_id=PROFILE)


#: The client call (or, for steps, the tool's step body) that sends each
#: request example. Every request example the hosts accept is here.
REQUESTS = {
    ("sessions", "open a session"): lambda c: c.open_session(session_id=PROFILE, url="https://shop.example.test/"),
    ("control", "hand to the person"): lambda c: c.hand_to_person(session_id=PROFILE, reason="Please sign in to Example Shop"),
    ("tabs", "open a tab"): lambda c: c.new_tab(session_id=PROFILE, url="https://docs.example.test/"),
    ("sites", "note a site"): lambda c: c.note_site(session_id=PROFILE, site="github.com", account_hint="ada"),
    ("saved_tabs", "save a tab"): lambda c: c.save_tab(
        project=PROJECT, name="Q3 pipeline report", url="https://crm.example.test/reports/q3", notes="start here",
    ),
}
#: The step the model writes for each action request example.
STEPS = {
    "click an element": {"action": "click", "element": 7},
    "type and submit": {"action": "type", "element": 1, "text": "MacBook Air", "submit": True},
    "click at a point on the screenshot": {"action": "click", "x": 412, "y": 230},
    "press Enter in an element": {"action": "press", "key": "Enter", "element": 2},
    "drag from an element to a point": {"action": "drag", "from": {"element": 5}, "to": {"x": 600, "y": 300}},
}
#: The step the model writes for each action request example the tool's gate
#: sends to the desktop app's browser once the person approves its card, on
#: the page with a frame.
APPROVED_IN_APP_STEPS = {
    "click at a point the person approved as one step that may land in a frame": {"action": "click", "x": 200, "y": 350},
}


async def _approved_in_app(step: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
    """The body the gate sends to the desktop app's browser when the person
    approves the step's card, on a page that holds still while the card
    waits: the gate reads it again after a card for a point in a frame."""
    async def approve(*args, **kwargs):
        return None

    monkeypatch.setattr(browser_tool, "ask_approval_card", approve)
    session = SimpleNamespace(_browser_config=BrowserConfig(base_url="http://127.0.0.1:51234", profile=PROFILE, token="k"))
    state = browser_tool._BrowserState(client=_client(_Recorder(page="a page with a frame")), session_id=PROFILE)
    page = Page.model_validate(_load("page").examples["a page with a frame"].body)
    return await browser_tool._gate(session, state=state, body=browser_tool._step_body({"step": step}), page=page, risky=False)


@pytest.mark.parametrize(("reason", "sent"), [
    ("a" * 299 + "\U0001F600", "a" * 299),
    ("\U0001F600" * 150, "\U0001F600" * 150),
    ("a" * 300 + "b", "a" * 300),
])
async def test_the_hand_over_reason_fits_the_hosts_utf16_limit(reason, sent):
    """The hosts count the reason in UTF-16 units and answer a longer one
    400, so an emoji counts 2 and the cut never splits one."""
    recorder = _Recorder()
    await _client(recorder).hand_to_person(session_id=PROFILE, reason=reason)
    body = recorder.sent[0][2]
    assert body["reason"] == sent
    assert len(body["reason"].encode("utf-16-le")) // 2 <= client_module.REASON_UNITS


def test_every_accepted_request_is_one_the_client_sends(monkeypatch):
    accepted = {(f, n) for f, n, e in _examples("request") if not e.refused}
    assert accepted == set(REQUESTS) | {("action", n) for n in [*STEPS, *APPROVED_IN_APP_STEPS]}
    for (file, name), call in REQUESTS.items():
        recorder = _Recorder()
        asyncio.run(call(_client(recorder)))
        assert recorder.sent[-1][2] == _load(file).examples[name].body, name
    for name, step in STEPS.items():
        assert browser_tool._step_body({"step": step}) == _load("action").examples[name].body, name
    for name, step in APPROVED_IN_APP_STEPS.items():
        assert asyncio.run(_approved_in_app(step, monkeypatch)) == _load("action").examples[name].body, name


def test_the_client_never_sends_a_refused_request():
    refused = [e.body for _, _, e in _examples("request") if e.refused]
    assert refused == [{"holder": "agent", "reason": ""}]
    recorder = _Recorder()
    asyncio.run(_every_client_call(_client(recorder)))
    assert len(recorder.sent) == 16
    sent = [body for _, _, body in recorder.sent]
    for body in refused:
        assert body not in sent
    assert not any(isinstance(b, dict) and b.get("holder") == "agent" for b in sent)


# ── errors ─────────────────────────────────────────────────────────────────

#: The reason the tool answers each host error with. Each refusal has its
#: own, so none reads as a bad credential unless it is one.
ERROR_REASONS = {
    "unauthorized": "browser_unauthorized",
    "unscoped": "browser_unscoped",
    "not_entitled": "browser_not_entitled",
    "person_only": "browser_person_only",
    "password_field": "browser_secret_field",
    "bad_request": "browser_bad_input",
    "tab_not_found": "browser_error",
    "tab_limit": "browser_error",
    "human_in_control": "browser_human_in_control",
    "stale_element": "browser_stale_element",
    "page_blocked": "browser_page_blocked",
    "page_busy": "browser_page_blocked",
    "daily_limit": "browser_daily_limit",
    "browser_unavailable": "browser_unreachable",
    "entitlements_unavailable": "browser_entitlements_unavailable",
    "out_of_scope": "browser_out_of_scope",
    "forbidden": "browser_forbidden",
    "at_capacity": "browser_at_capacity",
    "approval_required": "browser_step_not_approved",
    "too_many_saved_tabs": "browser_error",
    "browser_starting": "browser_unreachable",
    "too_many_requests": "browser_at_capacity",
}


def test_every_error_body_is_named_by_its_code_and_mapped():
    errors = _load("error").examples
    assert set(errors) == set(ERROR_REASONS)
    for name, example in errors.items():
        assert example.kind == "response" and example.status >= 400
        assert example.body["error"] == name and example.body["message"]


@pytest.mark.parametrize("code", sorted(ERROR_REASONS))
async def test_each_host_error_reaches_the_model_with_its_own_reason(code, monkeypatch):
    monkeypatch.setattr(client_module, "BUSY_RETRY_S", (0.0, 0.0, 0.0))
    example = _load("error").examples[code]
    client = BrowserClient(
        base_url=ROUTER, credential=_k, trust_env=True, first_call_timeout_s=45.0,
        http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(example.status, json=example.body))),
    )
    with pytest.raises(BrowserError) as raised:
        await client.page(session_id=PROFILE)
    assert (raised.value.status, raised.value.code, raised.value.message) == (example.status, code, example.body["message"])

    session = SimpleNamespace(
        _browser_config=BrowserConfig(base_url=ROUTER, profile=PROFILE),
        _browser_state=browser_tool._BrowserState(client=client, session_id=PROFILE),
    )
    result = await browser_tool.handle_browser(session, {"action": "page"})
    assert result.ok is False and result.reason == ERROR_REASONS[code]
    if code not in ("unauthorized",):
        assert "reconnect MindsHub" not in result.content


# ── the session line and the browser block ─────────────────────────────────


def test_the_session_line_is_what_the_cloud_turn_writes():
    from anton.cloud_turn.__main__ import stream_turn
    from anton.core.llm.provider import StreamBrowserSession

    (example,) = _load("session_event").examples.values()

    class _Session:
        async def turn_stream(self, user_input, **kwargs):
            yield StreamBrowserSession(session_id=example.body["session_id"])

        def close(self):
            pass

    lines: list[dict] = []
    request = json.dumps({"protocol_version": 1, "conversation_id": "c", "input": "hi"})
    asyncio.run(stream_turn(request, lines.append, session_builder=lambda r: _Session()))
    assert [line for line in lines if line.get("kind") == "browser"] == [example.body]


@pytest.mark.parametrize(
    "name,example", [(n, e) for _, n, e in _examples("turn_block")], ids=lambda v: v if isinstance(v, str) else "",
)
def test_turn_blocks_read_back_as_the_block_written(name, example):
    config = BrowserConfig.from_dict(example.body)
    if example.refused:
        assert config is None, name
        return
    assert config is not None and config.to_dict() == example.body


@pytest.mark.parametrize(
    "name,example", [(n, e) for _, n, e in _examples("constructor")], ids=lambda v: v if isinstance(v, str) else "",
)
def test_constructor_arguments_build_only_the_two_forms(name, example):
    if example.refused:
        with pytest.raises(ValidationError):
            BrowserConfig(**example.body)
        return
    config = BrowserConfig(**example.body)
    assert config.in_app and "launch-key-example" not in repr(config)
    assert BrowserConfig.from_dict(example.body) is None
    with pytest.raises(ValueError):
        config.to_dict()
