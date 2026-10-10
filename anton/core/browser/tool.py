"""The ``browser`` tool: drive the user's MindsHub browser, side by side with them.

The model states a goal; Jev (``/v1/decisions``) picks each click, scroll
and select on the way to it; the model writes any text and takes over
whenever Jev is unsure. The user watches in the viewer the host opens next
to the chat, and takes over for sign-ins, CAPTCHAs and confirmations.

``browser_jev`` (settings) decides how much Jev does:

- ``on``: ``run`` executes Jev's confident steps itself.
- ``shadow`` (default): ``run`` only reads the page. Jev's pick is recorded
  and compared with the step the model then takes, so we can measure
  agreement before letting Jev act.
- ``off``: no Jev calls at all.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from anton.core.browser.client import BrowserClient, BrowserError
from anton.core.browser.page import format_page, signature
from anton.core.browser.policy import (
    StepDecision,
    parse_step,
    signed_in_request,
    step_request,
)
from anton.core.tools.registry import ToolOutcome

if TYPE_CHECKING:
    from anton.core.session import ChatSession

logger = logging.getLogger(__name__)

#: A step Jev may take on its own needs at least this probability, the same
#: bar as the completion verifier's ``verifier_jev_min_p`` default.
DEFAULT_MIN_P = 0.8
#: Above this, a step is treated as committing something: ask first.
RISKY_P = 0.5
#: Steps one ``run`` may take before handing back to the model.
DEFAULT_MAX_STEPS = 15
#: A step that leaves the page unchanged this many times in a row ends the run.
MAX_STALLS = 2
SCROLL_PX = 700

ACTIONS = ("open", "run", "page", "act", "screenshot", "hand_to_user", "note_login", "close")
STEP_ACTIONS = ("click", "type", "press", "select", "scroll", "hover", "back", "forward", "reload", "wait", "navigate")


@dataclass
class _BrowserState:
    """Per-session browser state, kept on the ChatSession."""

    client: BrowserClient
    session_id: str
    opened: bool = False
    # Shadow mode: Jev's pick for the page the model is looking at, keyed by
    # that page's signature so a stale prediction is never compared.
    shadow: dict | None = None
    steps: list[str] = field(default_factory=list)


# ── session plumbing ───────────────────────────────────────────────────────


def _setting(session: "ChatSession", name: str, default):
    return getattr(getattr(session, "_settings", None), name, default)


def _mindshub_connection(session: "ChatSession"):
    """(base_url, provider) when the session's LLM is MindsHub, else (None, None)."""
    from anton.core.llm.endpoints import ENDPOINT_MINDSHUB, classify_base_url

    llm = getattr(session, "_llm", None)
    provider = getattr(llm, "coding_provider", None)
    if provider is None or not hasattr(provider, "export_connection_info"):
        return None, None
    base_url = provider.export_connection_info().base_url or ""
    if classify_base_url(base_url) != ENDPOINT_MINDSHUB or not hasattr(provider, "current_api_key"):
        return None, None
    return base_url, provider


async def _credential(session: "ChatSession") -> str:
    """The MindsHub credential the worker checks: the turn key in a cloud pod,
    the user's key or token on desktop and the CLI."""
    _, provider = _mindshub_connection(session)
    if provider is not None:
        try:
            key = await provider.current_api_key()
        except Exception:
            key = ""
        if key:
            return key
    return _setting(session, "minds_api_key", "") or ""


def _state(session: "ChatSession") -> _BrowserState:
    state = getattr(session, "_browser_state", None)
    if state is None:
        config = session._browser_config
        state = _BrowserState(
            client=BrowserClient(config.base_url, lambda: _credential(session)),
            session_id=config.profile,
        )
        session._browser_state = state
    return state


def _jev_mode(session: "ChatSession") -> str:
    mode = str(_setting(session, "browser_jev", "shadow")).lower()
    return mode if mode in ("on", "shadow", "off") else "shadow"


async def _jev(session: "ChatSession", state: dict, questions: dict):
    """One Jev decision, or None when Jev is off or the session isn't on MindsHub."""
    if _jev_mode(session) == "off":
        return None
    base_url, provider = _mindshub_connection(session)
    if provider is None:
        return None
    try:
        api_key = await provider.current_api_key()
    except Exception:
        return None
    if not api_key:
        return None
    from anton.core.llm import jev

    return await jev.decide(
        base_url=base_url,
        api_key=api_key,
        model=_setting(session, "verifier_jev_model", "jev-1.13.0"),
        state=state,
        questions=questions,
        timeout_s=float(_setting(session, "browser_jev_timeout_s", 4.0)),
    )


def _track(session: "ChatSession", event: str, **props) -> None:
    """Content-free analytics. Never raises."""
    try:
        from anton.analytics import send_event
        from anton.core.tools.tool_handlers import _ask_user_telemetry_settings

        settings = _ask_user_telemetry_settings(session)
        if settings:
            send_event(settings, event, **{k: str(v) for k, v in props.items()})
    except Exception:
        pass


async def _ask(session: "ChatSession", prompt: str, options: list[tuple[str, str]]) -> str | None:
    """Ask the user to pick one option; the value, or None when nobody can answer."""
    elicitor = getattr(session, "elicitor", None)
    if elicitor is None or "choice" not in getattr(elicitor, "supported_kinds", ()):
        return None
    import uuid

    from anton.core.interaction.elicit import AskOption, AskRequest, elicit

    request = AskRequest(
        prompt=prompt,
        kind="choice",
        timeout_s=getattr(elicitor, "timeout_s", None),
        options=tuple(AskOption(value=v, label=label, style="primary" if i == 0 else "default") for i, (v, label) in enumerate(options)),
        allow_custom=False,
    )
    answer = await elicit(session, f"browser:{uuid.uuid4().hex}", request)
    if answer.status != "answered" or not answer.values:
        return answer.status
    return answer.values[0]


# ── actions ────────────────────────────────────────────────────────────────


async def _ensure_open(session: "ChatSession", url: str | None = None) -> dict:
    """Open the profile's browser session once per conversation turn chain and
    tell the host to show it. Returns the session's info from the service."""
    state = _state(session)
    if state.opened:
        # The service relaunches Chromium on the same profile by itself if it
        # was reaped for idleness, so only a new URL needs a call here.
        if url:
            await state.client.act(state.session_id, {"action": "navigate", "url": url, "include_page": False})
        return {}
    info = await state.client.open_session(state.session_id, url)
    if not state.opened:
        state.opened = True
        try:
            embed = await state.client.embed(state.session_id)
            from anton.core.llm.provider import StreamBrowserSession

            await session.emit(StreamBrowserSession(
                session_id=state.session_id,
                view_url=embed.get("view_url", ""),
                expires_at=int(embed.get("expires_at") or 0),
            ))
        except BrowserError as exc:
            # The browser still works without a viewer; say so instead of failing.
            logger.warning("browser embed failed: %s", exc)
    return info


def _sites_line(sites: list[dict]) -> str:
    if not sites:
        return "Signed-in sites on record: none yet."
    live = [s for s in sites if s.get("cookies_present")]
    gone = [s for s in sites if not s.get("cookies_present")]

    def name(s: dict) -> str:
        return f"{s['site']} ({s['account_hint']})" if s.get("account_hint") else s["site"]

    parts = []
    if live:
        parts.append("still signed in (cookies present): " + ", ".join(name(s) for s in live))
    if gone:
        parts.append("session looks expired: " + ", ".join(name(s) for s in gone))
    return "Signed-in sites on record: " + "; ".join(parts) + "."


async def _do_open(session: "ChatSession", tc_input: dict) -> str:
    state = _state(session)
    await _ensure_open(session, tc_input.get("url") or None)
    page = await state.client.page(state.session_id)
    sites = await state.client.sites(state.session_id)
    return (
        "The browser is open and the user can see it beside the chat.\n"
        f"{_sites_line(sites)}\n\n{format_page(page)}"
    )


async def _do_page(session: "ChatSession", tc_input: dict) -> str:
    state = _state(session)
    await _ensure_open(session)
    page = await state.client.page(state.session_id)
    return format_page(page)


def _step_body(tc_input: dict) -> dict:
    step = tc_input.get("step")
    if not isinstance(step, dict) or step.get("action") not in STEP_ACTIONS:
        raise ValueError(f"act needs `step` with an action in {', '.join(STEP_ACTIONS)}")
    return {**step, "include_page": True}


async def _do_act(session: "ChatSession", tc_input: dict) -> str:
    state = _state(session)
    body = _step_body(tc_input)
    await _ensure_open(session)
    _record_shadow_agreement(session, state, body)
    result = await state.client.act(state.session_id, body)
    state.steps.append(_describe(body))
    page = result.get("page") or await state.client.page(state.session_id)
    return f"Done: {_describe(body)}.\n\n{format_page(page)}"


async def _do_screenshot(session: "ChatSession", tc_input: dict):
    state = _state(session)
    await _ensure_open(session)
    image = await state.client.screenshot(state.session_id, annotate=tc_input.get("annotate", True) is not False)
    return ToolOutcome(
        content=[
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(image).decode("ascii")}},
            {"type": "text", "text": "Screenshot of the browser. Red labels are element ids from the page list."},
        ],
        ok=True,
    )


async def _do_hand_to_user(session: "ChatSession", tc_input: dict) -> str:
    reason = (tc_input.get("reason") or "").strip() or "This step needs you."
    state = _state(session)
    await _ensure_open(session)
    answer = await _ask(
        session,
        f"{reason}\n\nUse the browser beside the chat. Choose Done when you're finished.",
        [("done", "Done"), ("stop", "Stop")],
    )
    if answer is None:
        return (
            "No one can answer in this conversation right now. Tell the user what they need to do in "
            "the browser (they can see it beside the chat), then end your turn and wait for them."
        )
    if answer != "done":
        return f"The user did not take over ({answer}). Do not retry the step; ask how they want to continue."
    page = await state.client.page(state.session_id)
    noted = await _maybe_note_login(session, page)
    lines = ["The user finished in the browser."]
    if noted:
        lines.append(noted)
    return "\n".join(lines) + f"\n\n{format_page(page)}"


async def _maybe_note_login(session: "ChatSession", page: dict) -> str:
    """After a hand-over, ask Jev whether the user is now signed in; record it if so."""
    state_, questions = signed_in_request(page)
    decision = await _jev(session, state_, questions)
    if decision is None or decision.error:
        return ""
    min_p = float(_setting(session, "browser_jev_min_p", DEFAULT_MIN_P))
    if decision.yes("signed_in") < min_p:
        return ""
    host = urlparse(page.get("url", "")).hostname or ""
    if not host:
        return ""
    return await _note_login(session, host, "")


async def _note_login(session: "ChatSession", site: str, account_hint: str) -> str:
    state = _state(session)
    record = await state.client.note_site(state.session_id, site, account_hint)
    site = record.get("site") or site
    hint = record.get("account_hint") or account_hint
    _remember_login(session, site, hint)
    who = f" as {hint}" if hint else ""
    return f"Noted: the user is signed in to {site}{who}. Logins stay in this browser between conversations."


def _remember_login(session: "ChatSession", site: str, hint: str) -> None:
    """A global profile memory, so later conversations know the system exists and
    that the browser holds the login. Never a credential."""
    cortex = getattr(session, "_cortex", None)
    if cortex is None or getattr(cortex, "mode", "off") == "off":
        return
    import asyncio

    from anton.core.memory.base import Engram

    who = f" as {hint}" if hint else ""
    engram = Engram(
        text=(
            f"The user is signed in to {site}{who} in their MindsHub browser. For work in {site}, use "
            "the browser tool instead of asking for credentials; if the session has expired, hand over "
            "so they can sign in again."
        ),
        kind="profile",
        scope="global",
        confidence="high",
        topic="browser",
        source="user",
    )

    async def _encode():
        try:
            await cortex.encode([engram])
        except Exception:
            pass

    track = getattr(session, "_track_memory_write", None)
    task = asyncio.create_task(_encode())
    if track is not None:
        track(task)


async def _do_note_login(session: "ChatSession", tc_input: dict) -> str:
    site = (tc_input.get("site") or "").strip()
    if not site:
        raise ValueError("note_login needs `site`, e.g. salesforce.com")
    await _ensure_open(session)
    return await _note_login(session, site, (tc_input.get("account_hint") or "").strip())


async def _do_close(session: "ChatSession", tc_input: dict) -> str:
    state = _state(session)
    await state.client.close_session(state.session_id)
    state.opened = False
    return "Closed the browser. The profile and its logins are kept for next time."


# ── run: the Jev step loop ─────────────────────────────────────────────────


def _describe(step: dict) -> str:
    action = step.get("action", "")
    if step.get("element") is not None:
        text = f" {step['text']!r}" if "text" in step else ""
        return f"{action} [{step['element']}]{text}"
    if action == "navigate":
        return f"navigate {step.get('url', '')}"
    if action == "scroll":
        return f"scroll {step.get('direction', 'down')}"
    if action == "press":
        return f"press {step.get('key', '')}"
    return action


def _record_shadow_agreement(session: "ChatSession", state: _BrowserState, body: dict) -> None:
    """Shadow mode: compare the model's step with what Jev picked for the same page."""
    shadow, state.shadow = state.shadow, None
    if not shadow:
        return
    jev_action = shadow["action"]
    llm_action = body.get("action", "")
    if llm_action == "scroll":
        llm_action = f"scroll_{body.get('direction', 'down')}"
    _track(
        session,
        "browser_jev_shadow",
        jev_action=jev_action,
        llm_action=llm_action,
        agree_action=jev_action == llm_action,
        agree_target=shadow["target"] is not None and shadow["target"] == body.get("element"),
        p_action=f"{shadow['action_p']:.2f}",
        p_target=f"{shadow['target_p']:.2f}",
    )


def _step_from(decision: StepDecision, inputs: dict, page: dict) -> dict | None:
    """The service action for a confident Jev step, or None when Jev can't take it alone."""
    if decision.action in ("scroll_down", "scroll_up"):
        return {"action": "scroll", "direction": decision.action.split("_")[1], "amount": SCROLL_PX}
    if decision.action == "back":
        return {"action": "back"}
    if decision.target is None:
        return None
    if decision.action == "click":
        return {"action": "click", "element": decision.target}
    if decision.action == "type":
        if decision.input_key is None or decision.input_key not in inputs:
            return None
        return {"action": "type", "element": decision.target, "text": str(inputs[decision.input_key])}
    if decision.action == "select":
        if decision.input_key is None or decision.input_key not in inputs:
            return None
        return {"action": "select", "element": decision.target, "value": str(inputs[decision.input_key])}
    return None


def _handback(reason: str, taken: list[str], page: dict) -> str:
    steps = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(taken)) or "  (none)"
    return f"{reason}\nSteps taken:\n{steps}\n\n{format_page(page)}"


async def _do_run(session: "ChatSession", tc_input: dict) -> str:
    goal = (tc_input.get("goal") or "").strip()
    if not goal:
        raise ValueError("run needs a `goal`")
    inputs = tc_input.get("inputs") if isinstance(tc_input.get("inputs"), dict) else {}
    state = _state(session)
    await _ensure_open(session, tc_input.get("url") or None)
    mode = _jev_mode(session)
    min_p = float(_setting(session, "browser_jev_min_p", DEFAULT_MIN_P))
    max_steps = int(tc_input.get("max_steps") or _setting(session, "browser_max_steps", DEFAULT_MAX_STEPS))
    page = await state.client.page(state.session_id)

    if mode != "on":
        if mode == "shadow":
            jev_state, questions, _ = step_request(goal, page, inputs, state.steps)
            decision = await _jev(session, jev_state, questions)
            if decision is not None and not decision.error:
                step = parse_step(decision)
                state.shadow = {"action": step.action, "action_p": step.action_p, "target": step.target, "target_p": step.target_p}
        return (
            "Step-by-step mode: read the page and take the next step with action=act, "
            "then repeat until the goal is done.\n\n" + format_page(page)
        )

    taken: list[str] = []
    stalls = 0
    for _ in range(max_steps):
        jev_state, questions, _ = step_request(goal, page, inputs, state.steps)
        decision = await _jev(session, jev_state, questions)
        if decision is None or decision.error:
            reason = "Jev is not available here" if decision is None else f"Jev failed ({decision.error})"
            return _handback(f"{reason}; continue step by step with action=act.", taken, page)
        step = parse_step(decision)
        _track(session, "browser_jev_step", action=step.action, p_action=f"{step.action_p:.2f}", ms=step.ms)

        if step.needs_user_p >= min_p or step.action == "needs_user":
            return _handback(
                "The page needs the user (sign-in, CAPTCHA, code or confirmation). "
                "Call action=hand_to_user with a one-line reason.",
                taken,
                page,
            )
        if step.action == "done" and step.action_p >= min_p:
            return _handback("Jev reports the goal is done. Check the page and answer.", taken, page)
        if step.action_p < min_p or step.action in ("unsure", "done", ""):
            return _handback("Not sure of the next step; take it with action=act.", taken, page)

        body = _step_from(step, inputs, page)
        if body is None:
            hint = f" on [{step.target}]" if step.target is not None else ""
            what = "text to type" if step.action in ("type", "select") else "a target"
            return _handback(f"The next step is {step.action}{hint}, but it needs {what} from you; use action=act.", taken, page)
        if body.get("element") is not None and step.target_p < min_p:
            return _handback(f"Not sure which element to {step.action}; take it with action=act.", taken, page)

        if step.risky_p >= RISKY_P and body["action"] in ("click", "type", "select"):
            answer = await _ask(
                session,
                f"Anton is about to {_describe(body)} on {page.get('title') or page.get('url', 'this page')}. "
                "This may commit something (a purchase, a message, a deletion or a form). Go ahead?",
                [("proceed", "Go ahead"), ("stop", "Stop")],
            )
            if answer != "proceed":
                return _handback(f"Paused before {_describe(body)}: the user didn't confirm it.", taken, page)

        before = signature(page)
        result = await state.client.act(state.session_id, {**body, "include_page": True})
        description = _describe(body)
        taken.append(description)
        state.steps.append(description)
        page = result.get("page") or await state.client.page(state.session_id)
        stalls = stalls + 1 if signature(page) == before else 0
        if stalls >= MAX_STALLS:
            return _handback("The last steps changed nothing on the page; take over with action=act.", taken, page)

    return _handback(f"Stopped after {max_steps} steps without finishing.", taken, page)


_HANDLERS = {
    "open": _do_open,
    "run": _do_run,
    "page": _do_page,
    "act": _do_act,
    "screenshot": _do_screenshot,
    "hand_to_user": _do_hand_to_user,
    "note_login": _do_note_login,
    "close": _do_close,
}


async def handle_browser(session: "ChatSession", tc_input: dict):
    action = tc_input.get("action")
    handler = _HANDLERS.get(action)
    if handler is None:
        return ToolOutcome(content=f"Unknown action {action!r}; use one of {', '.join(ACTIONS)}.", ok=False, reason="browser_bad_input")
    try:
        result = await handler(session, tc_input)
    except ValueError as exc:
        return ToolOutcome(content=f"Error: {exc}", ok=False, reason="browser_bad_input")
    except BrowserError as exc:
        if exc.code == "stale_element":
            return ToolOutcome(content=f"{exc.message}", ok=False, reason="browser_stale_element")
        if exc.status in (401, 403):
            return ToolOutcome(
                content=f"The browser refused this session's credentials ({exc.code}). The user may need to reconnect MindsHub.",
                ok=False, reason="browser_unauthorized",
            )
        if exc.status in (0, 502, 503):
            return ToolOutcome(
                content=f"The browser is not reachable right now ({exc.code}): {exc.message}. It may be starting; try again shortly.",
                ok=False, reason="browser_unreachable",
            )
        return ToolOutcome(content=f"Browser error {exc.status} {exc.code}: {exc.message}", ok=False, reason="browser_error")
    if isinstance(result, ToolOutcome):
        return result
    return ToolOutcome(content=result, ok=True)


BROWSER_TOOL_PROMPT = """\
BROWSER: you and the user share a real browser (Chromium, their own MindsHub instance). The user sees it beside this chat and can click and type in it at any time.
- Prefer a connector or API when one can do the job; use the browser for sites without one, or when the user asks.
- Only open sites the user named or that the task plainly needs. Never invent URLs.
- Start a task with action=run and a goal; give values to type in `inputs`. run takes the steps it is sure of and hands back with the page when it isn't.
- To act yourself, use action=act with a step: {"action":"click","element":17}, {"action":"type","element":4,"text":"…","submit":true}, {"action":"select","element":9,"value":"Blue"}, {"action":"scroll","direction":"down"}, {"action":"press","key":"Enter"}, {"action":"back"}, {"action":"navigate","url":"…"}. Element numbers come from the latest page.
- Never ask for, read out, or type a password, one-time code, or card number. For sign-ins, CAPTCHAs, two-factor codes and payments, call action=hand_to_user with a short reason and let the user do it in the browser.
- After the user signs in to a site, call action=note_login with the site (and an account hint they'd recognise, never a secret) so later conversations know the login is there.
- Ask before anything that commits on the user's behalf: buying, paying, sending, deleting, or submitting a form.
- Only report what the page actually shows. If a step fails or the page didn't change, say so.
- Use action=screenshot only when the page list can't show what you need (canvas, maps, charts, text in images).
"""

BROWSER_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": list(ACTIONS),
            "description": (
                "open: open the browser (optionally at url) and list signed-in sites. "
                "run: work toward a goal, taking the steps that are clear. "
                "page: read the current page. act: take one step. screenshot: see the page. "
                "hand_to_user: let the user take over (sign-in, CAPTCHA, payment). "
                "note_login: record that the user is signed in to a site. close: close the browser."
            ),
        },
        "url": {"type": "string", "description": "open / run: page to start on."},
        "goal": {"type": "string", "description": "run: what to achieve on the site, in one sentence."},
        "inputs": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "run: values run may type or select, by a short name, e.g. {\"search\": \"MacBook Air\"}. Never secrets.",
        },
        "step": {"type": "object", "description": "act: one step, e.g. {\"action\": \"click\", \"element\": 17}."},
        "reason": {"type": "string", "description": "hand_to_user: what the user needs to do, in one line."},
        "site": {"type": "string", "description": "note_login: the site, e.g. salesforce.com."},
        "account_hint": {"type": "string", "description": "note_login: which account, e.g. jorge@acme. Never a secret."},
        "annotate": {"type": "boolean", "description": "screenshot: draw element numbers (default true)."},
        "max_steps": {"type": "integer", "description": "run: step limit (default 15)."},
    },
    "required": ["action"],
}


def build_browser_tool():
    from anton.core.tools.tool_defs import ToolDef

    return ToolDef(
        name="browser",
        description=(
            "Use the browser you share with the user: open sites, read pages as numbered elements, "
            "click, type, select and scroll, and hand over to the user for sign-ins and confirmations."
        ),
        input_schema=BROWSER_TOOL_SCHEMA,
        handler=handle_browser,
        prompt=BROWSER_TOOL_PROMPT,
    )
