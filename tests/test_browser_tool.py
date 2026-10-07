"""The `browser` tool (ENG-3296): the HTTP calls it makes, the Jev step loop,
hand-overs, login memory and how it is wired into sessions and the cloud pod."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx2 as httpx
import pytest

from anton.core.browser.client import BrowserClient
from anton.core.browser.config import BrowserConfig
from anton.core.browser.page import format_page, shortlist, signature
from anton.core.browser.policy import parse_step, step_request
from anton.core.browser import tool as browser_tool
from anton.core.interaction.elicit import AskAnswer
from anton.core.llm import jev
from anton.core.llm.provider import StreamBrowserSession

BASE = "https://br-abc12345.4nton.ai"

PAGE = {
    "url": "https://shop.example/",
    "title": "Shop",
    "elements": [
        {"id": 1, "role": "searchbox", "name": "Search"},
        {"id": 2, "role": "button", "name": "Search"},
        {"id": 3, "role": "link", "name": "Today's deals", "offscreen": True},
        {"id": 4, "role": "button", "name": "Disabled thing", "disabled": True},
    ],
    "text": "Welcome to the shop",
}


class FakeInstance:
    """The browser service plus the worker's /_embed, answering over MockTransport."""

    def __init__(self, pages=None):
        self.calls: list[tuple[str, str, dict | None, str]] = []
        self.pages = list(pages or [PAGE])
        self.sites: list[dict] = []

    def page(self):
        return self.pages[0] if len(self.pages) == 1 else self.pages.pop(0)

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.calls.append((request.method, path, body, request.headers.get("authorization", "")))
        if path == "/sessions" and request.method == "POST":
            return httpx.Response(201, json={"session_id": body["session_id"], "url": "about:blank"})
        if path == "/_embed":
            return httpx.Response(200, json={"view_url": f"{BASE}/sessions/main/view?et=tok", "expires_at": 1999999999})
        if path.endswith("/page"):
            return httpx.Response(200, json=self.page())
        if path.endswith("/action"):
            if body.get("element") == 99:
                return httpx.Response(409, json={"error": "stale_element", "message": "element 99 is no longer on the page"})
            return httpx.Response(200, json={"ok": True, "url": PAGE["url"], "page": self.page()})
        if path.endswith("/sites") and request.method == "GET":
            return httpx.Response(200, json={"sites": self.sites})
        if path.endswith("/sites") and request.method == "POST":
            record = {"site": body["site"], "account_hint": body.get("account_hint", ""), "cookies_present": True}
            self.sites.append(record)
            return httpx.Response(200, json={"site": record})
        if path.endswith("/screenshot"):
            return httpx.Response(200, content=b"\xff\xd8jpeg")
        if request.method == "DELETE":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, json={"error": "not_found", "message": path})


class FakeElicitor:
    supported_kinds = ("choice",)
    answer_hint = ""
    timeout_s = None

    def __init__(self, answer="done"):
        self.answer = answer
        self.prompts: list[str] = []


class FakeCortex:
    mode = "autopilot"

    def __init__(self):
        self.encoded = []

    async def encode(self, engrams):
        self.encoded.extend(engrams)


def make_session(instance: FakeInstance, *, jev_mode="on", elicitor=None, mindshub=True):
    provider = SimpleNamespace(
        export_connection_info=lambda: SimpleNamespace(base_url="https://api.mindshub.ai/v1" if mindshub else "https://api.openai.com/v1"),
        current_api_key=lambda: _async("turn-key"),
    )
    tasks = []
    session = SimpleNamespace(
        _browser_config=BrowserConfig(base_url=BASE),
        _browser_state=None,
        _settings=SimpleNamespace(browser_jev=jev_mode, browser_jev_min_p=0.8, browser_max_steps=15, minds_api_key="user-key", verifier_jev_model="jev-test"),
        _llm=SimpleNamespace(coding_provider=provider),
        _cortex=FakeCortex(),
        _track_memory_write=tasks.append,
        elicitor=elicitor,
        events=[],
    )

    async def emit(event):
        session.events.append(event)

    session.emit = emit
    client = BrowserClient(BASE, lambda: browser_tool._credential(session), http=httpx.AsyncClient(transport=httpx.MockTransport(instance.handler)))
    session._browser_state = browser_tool._BrowserState(client=client, session_id="main")
    session.tasks = tasks
    return session


async def _async(value):
    return value


def jev_answers(action, action_p=0.95, target=None, target_p=0.95, needs_user=0.02, risky=0.02, input_key=None, signed_in=None):
    answers = {
        "next_action": {"choice": action, "probabilities": {action: action_p}},
        "target": {"choice": target or "none", "probabilities": {target or "none": target_p}},
        "needs_user": {"noul": needs_user},
        "risky": {"noul": risky},
    }
    if input_key:
        answers["input_key"] = {"choice": input_key, "probabilities": {input_key: 0.95}}
    if signed_in is not None:
        answers = {"signed_in": {"noul": signed_in}}
    return jev.JevDecision(answers=answers, ms=12)


@pytest.fixture
def fake_jev(monkeypatch):
    """Queue of Jev answers; each decide() call pops the next one."""
    queue: list = []
    calls: list[dict] = []

    async def decide(**kwargs):
        calls.append(kwargs)
        return queue.pop(0) if queue else jev.JevDecision(answers={}, error="exhausted")

    monkeypatch.setattr(jev, "decide", decide)
    return SimpleNamespace(queue=queue, calls=calls)


@pytest.fixture
def elicit_answers(monkeypatch):
    answers: list[str] = []
    prompts: list[str] = []

    async def elicit(session, question_id, request):
        prompts.append(request.prompt)
        value = answers.pop(0) if answers else "stop"
        return AskAnswer(status="answered", values=(value,))

    import anton.core.interaction.elicit as elicit_mod

    monkeypatch.setattr(elicit_mod, "elicit", elicit)
    return SimpleNamespace(answers=answers, prompts=prompts)


# ── page helpers ───────────────────────────────────────────────────────────


def test_format_page_is_one_line_per_element():
    text = format_page(PAGE)
    assert '[1] searchbox "Search"' in text
    assert "[3] link \"Today's deals\" (offscreen)" in text
    assert text.endswith("Welcome to the shop")


def test_shortlist_ranks_goal_words_and_drops_disabled():
    ranked = shortlist(PAGE["elements"], "open today's deals")
    assert ranked[0]["id"] == 3
    assert 4 not in [el["id"] for el in ranked]


def test_signature_changes_with_values():
    changed = {**PAGE, "elements": [{**PAGE["elements"][0], "value": "mac"}, *PAGE["elements"][1:]]}
    assert signature(PAGE) != signature(changed)


def test_step_request_asks_four_questions_plus_inputs():
    _, questions, candidates = step_request("search for macbook", PAGE, {"search": "MacBook Air"}, [])
    assert set(questions) == {"next_action", "target", "needs_user", "risky", "input_key"}
    assert "e1" in questions["target"]["criteria"] and "none" in questions["target"]["criteria"]
    assert len(candidates) == 3


def test_parse_step_reads_choices_and_tolerates_junk():
    step = parse_step(jev_answers("click", target="e2"))
    assert (step.action, step.target, step.target_p) == ("click", 2, 0.95)
    junk = parse_step(jev.JevDecision(answers={"next_action": {"choice": "fly", "probabilities": {"fly": 2}}}))
    assert junk.action == "" and junk.target is None
    assert parse_step(jev.JevDecision(answers={}, error="timeout")).error == "timeout"


# ── the tool ───────────────────────────────────────────────────────────────


async def test_open_emits_the_viewer_and_lists_sites(fake_jev):
    instance = FakeInstance()
    instance.sites = [{"site": "salesforce.com", "account_hint": "jorge@acme", "cookies_present": True},
                      {"site": "github.com", "account_hint": "", "cookies_present": False}]
    session = make_session(instance)
    result = await browser_tool.handle_browser(session, {"action": "open", "url": "shop.example"})
    assert result.ok is True
    assert "still signed in (cookies present): salesforce.com (jorge@acme)" in result.content
    assert "session looks expired: github.com" in result.content
    assert session.events == [StreamBrowserSession(session_id="main", view_url=f"{BASE}/sessions/main/view?et=tok", expires_at=1999999999)]
    method, path, body, auth = instance.calls[0]
    assert (method, path, body) == ("POST", "/sessions", {"session_id": "main", "url": "shop.example"})
    assert auth == "Bearer turn-key"

    # Opening again doesn't announce a second viewer.
    await browser_tool.handle_browser(session, {"action": "page"})
    assert len(session.events) == 1


async def test_credential_falls_back_to_the_users_key_off_mindshub():
    session = make_session(FakeInstance(), mindshub=False)
    assert await browser_tool._credential(session) == "user-key"


async def test_act_takes_one_step_and_returns_the_page():
    instance = FakeInstance()
    session = make_session(instance)
    result = await browser_tool.handle_browser(session, {"action": "act", "step": {"action": "click", "element": 2}})
    assert result.ok and result.content.startswith("Done: click [2].")
    assert ("POST", "/sessions/main/action", {"action": "click", "element": 2, "include_page": True}) == instance.calls[-1][:3]


async def test_act_rejects_bad_steps_and_reports_stale_elements():
    session = make_session(FakeInstance())
    bad = await browser_tool.handle_browser(session, {"action": "act", "step": {"action": "explode"}})
    assert bad.ok is False and bad.reason == "browser_bad_input"
    stale = await browser_tool.handle_browser(session, {"action": "act", "step": {"action": "click", "element": 99}})
    assert stale.ok is False and stale.reason == "browser_stale_element"


async def test_run_takes_confident_steps_with_inputs(fake_jev):
    typed = {**PAGE, "elements": [{**PAGE["elements"][0], "value": "MacBook Air"}, *PAGE["elements"][1:]]}
    results = {**PAGE, "url": "https://shop.example/s?q=mac", "elements": [{"id": 9, "role": "link", "name": "MacBook Air 13"}]}
    instance = FakeInstance(pages=[PAGE, typed, results])
    session = make_session(instance)
    fake_jev.queue += [
        jev_answers("type", target="e1", input_key="search"),
        jev_answers("click", target="e2"),
        jev_answers("done"),
    ]
    result = await browser_tool.handle_browser(session, {"action": "run", "goal": "search for a MacBook Air", "inputs": {"search": "MacBook Air"}})
    assert "Jev reports the goal is done" in result.content
    actions = [c[2] for c in instance.calls if c[1].endswith("/action")]
    assert actions == [
        {"action": "type", "element": 1, "text": "MacBook Air", "include_page": True},
        {"action": "click", "element": 2, "include_page": True},
    ]
    assert fake_jev.calls[0]["base_url"] == "https://api.mindshub.ai/v1"
    assert fake_jev.calls[0]["api_key"] == "turn-key"


async def test_run_hands_back_when_jev_is_unsure_or_needs_text(fake_jev):
    session = make_session(FakeInstance())
    fake_jev.queue.append(jev_answers("click", action_p=0.5, target="e2"))
    unsure = await browser_tool.handle_browser(session, {"action": "run", "goal": "do it"})
    assert "Not sure of the next step" in unsure.content

    fake_jev.queue.append(jev_answers("type", target="e1"))
    needs_text = await browser_tool.handle_browser(session, {"action": "run", "goal": "search"})
    assert "needs text to type from you" in needs_text.content

    fake_jev.queue.append(jev_answers("click", target="e2", target_p=0.4))
    which = await browser_tool.handle_browser(session, {"action": "run", "goal": "search"})
    assert "Not sure which element" in which.content


async def test_run_stops_at_sign_in_walls(fake_jev):
    session = make_session(FakeInstance())
    fake_jev.queue.append(jev_answers("click", target="e2", needs_user=0.93))
    result = await browser_tool.handle_browser(session, {"action": "run", "goal": "open my dashboard"})
    assert "Call action=hand_to_user" in result.content


async def test_run_asks_before_a_risky_step(fake_jev, elicit_answers):
    instance = FakeInstance()
    session = make_session(instance, elicitor=FakeElicitor())
    fake_jev.queue.append(jev_answers("click", target="e2", risky=0.9))
    elicit_answers.answers.append("stop")
    result = await browser_tool.handle_browser(session, {"action": "run", "goal": "buy it"})
    assert "didn't confirm" in result.content
    assert not [c for c in instance.calls if c[1].endswith("/action")]
    assert "Go ahead?" in elicit_answers.prompts[0]


async def test_run_stops_when_steps_change_nothing(fake_jev):
    session = make_session(FakeInstance())
    fake_jev.queue += [jev_answers("scroll_down")] * 3
    result = await browser_tool.handle_browser(session, {"action": "run", "goal": "find the footer"})
    assert "changed nothing" in result.content


async def test_run_without_jev_hands_the_page_to_the_model(fake_jev):
    session = make_session(FakeInstance(), mindshub=False)
    result = await browser_tool.handle_browser(session, {"action": "run", "goal": "anything"})
    assert "Jev is not available here" in result.content
    assert fake_jev.calls == []


async def test_shadow_mode_records_agreement_without_acting(fake_jev, monkeypatch):
    events = []
    monkeypatch.setattr(browser_tool, "_track", lambda session, event, **props: events.append((event, props)))
    instance = FakeInstance()
    session = make_session(instance, jev_mode="shadow")
    fake_jev.queue.append(jev_answers("click", target="e2"))
    result = await browser_tool.handle_browser(session, {"action": "run", "goal": "search"})
    assert result.content.startswith("Step-by-step mode")
    assert not [c for c in instance.calls if c[1].endswith("/action")]

    await browser_tool.handle_browser(session, {"action": "act", "step": {"action": "click", "element": 2}})
    assert events == [("browser_jev_shadow", {
        "jev_action": "click", "llm_action": "click", "agree_action": True, "agree_target": True,
        "p_action": "0.95", "p_target": "0.95",
    })]


async def test_off_mode_never_calls_jev(fake_jev):
    session = make_session(FakeInstance(), jev_mode="off")
    result = await browser_tool.handle_browser(session, {"action": "run", "goal": "search"})
    assert result.content.startswith("Step-by-step mode")
    assert fake_jev.calls == []


async def test_hand_to_user_then_notes_the_login(fake_jev, elicit_answers):
    signed_in_page = {**PAGE, "url": "https://acme.my.salesforce.com/home"}
    instance = FakeInstance(pages=[signed_in_page])
    session = make_session(instance, elicitor=FakeElicitor())
    elicit_answers.answers.append("done")
    fake_jev.queue.append(jev_answers("", signed_in=0.97))
    result = await browser_tool.handle_browser(session, {"action": "hand_to_user", "reason": "Please sign in to Salesforce"})
    assert "The user finished in the browser." in result.content
    assert "Noted: the user is signed in to acme.my.salesforce.com" in result.content
    assert instance.sites[0]["site"] == "acme.my.salesforce.com"
    for task in session.tasks:
        await task
    engram = session._cortex.encoded[0]
    assert (engram.kind, engram.scope, engram.topic) == ("profile", "global", "browser")
    assert "acme.my.salesforce.com" in engram.text


async def test_hand_to_user_without_an_elicitor_tells_the_model_to_wait():
    session = make_session(FakeInstance(), elicitor=None)
    result = await browser_tool.handle_browser(session, {"action": "hand_to_user", "reason": "Sign in"})
    assert "end your turn and wait" in result.content


async def test_note_login_keeps_hints_but_never_asks_for_secrets():
    instance = FakeInstance()
    session = make_session(instance)
    result = await browser_tool.handle_browser(session, {"action": "note_login", "site": "github.com", "account_hint": "jorge"})
    assert result.content.startswith("Noted: the user is signed in to github.com as jorge.")
    assert "password" not in json.dumps(instance.calls[-1][2])


async def test_screenshot_returns_an_image_block():
    session = make_session(FakeInstance())
    result = await browser_tool.handle_browser(session, {"action": "screenshot"})
    assert result.content[0]["type"] == "image"
    assert result.content[0]["source"]["media_type"] == "image/jpeg"


async def test_auth_failures_read_as_a_login_problem():
    def deny(request):
        return httpx.Response(403, json={"error": "Not authorized for this instance"})

    session = make_session(FakeInstance())
    session._browser_state.client = BrowserClient(BASE, lambda: _async("k"), http=httpx.AsyncClient(transport=httpx.MockTransport(deny)))
    result = await browser_tool.handle_browser(session, {"action": "page"})
    assert result.ok is False and result.reason == "browser_unauthorized"


# ── jev.decide ─────────────────────────────────────────────────────────────


async def test_decide_posts_questions_and_never_raises():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"risky": {"noul": 0.1}}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        decision = await jev.decide(base_url="https://api.mindshub.ai/v1", api_key="k", model="jev", state={"a": 1},
                                    questions={"risky": {"type": "noul"}}, timeout_s=2, client=client)
    assert decision.yes("risky") == 0.1 and decision.model == "jev-1.13.0"
    assert seen["questions"] == {"risky": {"type": "noul"}}

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
        failed = await jev.decide(base_url="https://x", api_key="k", model="jev", state={}, questions={}, timeout_s=2, client=client)
    assert failed.error == "http_500" and failed.answers == {}


# ── wiring ─────────────────────────────────────────────────────────────────


def test_browser_config_parses_only_https_instances():
    assert BrowserConfig.from_dict({"base_url": f"{BASE}/"}) == BrowserConfig(base_url=BASE, profile="main")
    assert BrowserConfig.from_dict({"base_url": "http://br-x.4nton.ai"}) is None
    assert BrowserConfig.from_dict({"base_url": BASE, "profile": "work"}).profile == "work"
    assert BrowserConfig.from_dict("nope") is None


def test_turn_request_carries_the_browser_block():
    from anton.cloud_turn.contract import TURN_PROTOCOL_VERSION, TurnRequestV1

    raw = {"protocol_version": TURN_PROTOCOL_VERSION, "conversation_id": "c", "input": "hi"}
    assert TurnRequestV1.from_json(json.dumps(raw)).browser is None
    with_browser = TurnRequestV1.from_json(json.dumps({**raw, "browser": {"base_url": BASE}}))
    assert with_browser.browser == {"base_url": BASE}
    assert TurnRequestV1.from_json(json.dumps({**raw, "browser": "junk"})).browser is None


def test_session_registers_the_tool_only_with_a_browser():
    from anton.core.session import _browser_from_settings

    assert _browser_from_settings(SimpleNamespace(browser_url="", browser_profile="main")) is None
    assert _browser_from_settings(SimpleNamespace(browser_url=BASE, browser_profile="main")) == BrowserConfig(base_url=BASE)
    assert _browser_from_settings(None) is None
