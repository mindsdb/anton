"""Jev as the browser's step policy.

One ``/v1/decisions`` call per step asks four things about the same page:
what kind of step comes next, which element it acts on, whether a person
has to act first, and whether the step would commit something hard to undo.
On a crowded page (more elements than one choice should carry, in several
parts) it asks which part of the page instead of which element, and a
second call picks the element within that part and asks the risk again.
The tool acts on a confident answer and hands anything else back to the
model, the way the completion verifier uses Jev (see session._jev_verdict).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from anton.core.browser.models import Element, Page
from anton.core.browser.page import SHORTLIST_SIZE, candidates, element_line, rank

ACTIONS = {
    "click": "Click a link, button, tab, checkbox or menu item to make progress toward the goal.",
    "type": "Type text into a field, such as a search box or a form input, to make progress.",
    "select": "Choose an option in a dropdown.",
    "scroll_down": "What the goal needs next is probably further down this page.",
    "scroll_up": "What the goal needs next is probably further up this page.",
    "back": "This page is a wrong turn for the goal; go back to the previous page.",
    "needs_user": (
        "A person has to act before the goal can continue: sign in, solve a CAPTCHA, enter a "
        "two-factor code, accept terms, or confirm a payment."
    ),
    "collect": (
        "This page shows information the goal needs, and the goal needs more from other pages: keep what "
        "this page shows, then go on."
    ),
    "done": "The goal is achieved, or all the information the goal asks for is visible or already kept.",
    "unsure": "None of the other options clearly fits.",
}

RISKY = {
    "type": "noul",
    "instructions": (
        "Would the most likely next step on this page commit something hard to undo on the "
        "user's behalf: a purchase or payment, sending a message or email, deleting data, "
        "or submitting a form?"
    ),
}

NO_TARGET = "none"
NO_INPUT = "none"


@dataclass(frozen=True)
class StepDecision:
    action: str = ""
    action_p: float = 0.0
    target: int | None = None
    target_p: float = 0.0
    input_key: str | None = None
    input_p: float = 0.0
    needs_user_p: float = 0.0
    risky_p: float = 0.0
    ms: int = 0
    error: str = ""


#: The most parts of a page Jev chooses among; the rest go under one "other parts".
MAX_REGIONS = 30
#: How many element names stand for a part of the page in its description.
REGION_SAMPLES = 6
REST_OF_PAGE = "the rest of the page"


@dataclass(frozen=True)
class StepRequest:
    """One step's question to Jev.

    ``regions`` is set on a crowded page (more elements than one choice
    should carry, in several parts): the request then asks which part of the
    page the step is in (``region``) instead of which element, and
    ``target_request`` asks for the element within the part Jev picked.
    """

    state: dict
    questions: dict
    candidates: list[Element]
    regions: dict[str, list[Element]] = field(default_factory=dict)


def _regions(ranked: list[Element]) -> dict[str, list[Element]]:
    """``ranked`` grouped by their containers, the part holding the best
    element first, as {"r1": [...], ...}; past MAX_REGIONS, the rest share one."""
    groups: dict[str, list[Element]] = {}
    for el in ranked:
        groups.setdefault(el.context, []).append(el)
    parts = list(groups.values())
    if len(parts) > MAX_REGIONS:
        parts = parts[: MAX_REGIONS - 1] + [[el for part in parts[MAX_REGIONS - 1:] for el in part]]
    return {f"r{i + 1}": part for i, part in enumerate(parts)}


def _region_line(part: list[Element]) -> str:
    contexts = {el.context for el in part}
    label = part[0].context if len(contexts) == 1 and part[0].context else (
        REST_OF_PAGE if contexts == {""} else "other parts of the page"
    )
    names = ", ".join(json.dumps(el.name, ensure_ascii=False) for el in part[:REGION_SAMPLES] if el.name)
    return f"{json.dumps(label, ensure_ascii=False)}: {len(part)} elements, such as {names}"


def step_request(
    *, goal: str, page: Page, inputs: dict | None, history: list[str], kept: bool = False,
) -> StepRequest:
    """The step's question: what kind of step, on which element (or, on a
    crowded page, in which part of it), whether a person must act, whether
    it commits something, and which input it types."""
    ranked = rank(candidates(page.elements), goal, inputs)
    regions = _regions(ranked) if len(ranked) > SHORTLIST_SIZE else {}
    if len(regions) < 2:
        regions = {}
    shown = ranked[:SHORTLIST_SIZE]
    state = {
        "goal": goal,
        "page": {
            "url": page.url,
            "title": page.title,
            "elements": [element_line(el) for el in shown],
            "text": page.text[:1500],
        },
        "steps_so_far": history[-8:],
    }
    if inputs:
        state["inputs"] = {str(k): str(v) for k, v in inputs.items()}
    questions: dict = {
        "next_action": {
            "type": "choice",
            "instructions": "Decide the single next browser step that best advances the goal on this page.",
            # A page already kept can't be kept again: the next step moves on.
            "criteria": {k: v for k, v in ACTIONS.items() if not (kept and k == "collect")},
        },
        "needs_user": {
            "type": "noul",
            "instructions": (
                "Does this page require the human user to act before the goal can continue, such as "
                "signing in, solving a CAPTCHA, entering a two-factor code, accepting terms, or "
                "confirming a payment?"
            ),
        },
        "risky": RISKY,
    }
    if regions:
        parts = {key: _region_line(part) for key, part in regions.items()}
        parts[NO_TARGET] = "The next step acts on no element."
        questions["region"] = {
            "type": "choice",
            "instructions": "Pick the part of the page that holds the element the next step should act on.",
            "criteria": parts,
        }
    else:
        questions["target"] = _target_question(shown)
    if inputs:
        keys = {str(k): f"The value provided as {k!r}." for k in inputs}
        keys[NO_INPUT] = "None of the provided values belongs in the field typed into next."
        questions["input_key"] = {
            "type": "choice",
            "instructions": "Which provided input value belongs in the field the next step types into?",
            "criteria": keys,
        }
    return StepRequest(state=state, questions=questions, candidates=shown, regions=regions)


def _target_question(elements: list[Element]) -> dict:
    targets = {f"e{el.id}": element_line(el) for el in elements}
    targets[NO_TARGET] = "No listed element is the right one to act on next."
    return {
        "type": "choice",
        "instructions": "Pick the element the next step should act on to advance the goal.",
        "criteria": targets,
    }


def target_request(
    *, goal: str, page: Page, action: str, part: list[Element], inputs: dict | None, history: list[str],
) -> StepRequest:
    """The second question on a crowded page: which element, within the part
    Jev picked, the ``action`` step acts on."""
    shown = rank(part, goal, inputs)[:SHORTLIST_SIZE]
    state = {
        "goal": goal,
        "next_step": action,
        "page": {"url": page.url, "title": page.title, "elements": [element_line(el) for el in shown]},
        "steps_so_far": history[-8:],
    }
    if inputs:
        state["inputs"] = {str(k): str(v) for k, v in inputs.items()}
    # Asked again now that the element is known: the first call rated it without one.
    return StepRequest(state=state, questions={"target": _target_question(shown), "risky": RISKY}, candidates=shown)


def parse_target(decision) -> tuple[int | None, float]:
    """The element a decision's ``target`` picked and its probability."""
    name, p = decision.choice("target")
    return (int(name[1:]) if name.startswith("e") and name[1:].isdigit() else None), p


def parse_step(decision) -> StepDecision:
    """A ``jev.JevDecision`` as a StepDecision; an error carries through unchanged."""
    if decision.error:
        return StepDecision(ms=decision.ms, error=decision.error)
    action, action_p = decision.choice("next_action")
    target, target_p = parse_target(decision)
    input_key, input_p = decision.choice("input_key")
    return StepDecision(
        action=action if action in ACTIONS else "",
        action_p=action_p,
        target=target,
        target_p=target_p,
        input_key=None if input_key in ("", NO_INPUT) else input_key,
        input_p=input_p,
        needs_user_p=decision.yes("needs_user"),
        risky_p=decision.yes("risky"),
        ms=decision.ms,
    )


def signed_in_request(page: Page) -> tuple[dict, dict]:
    """Whether the person is signed in, asked after they hand the browser back.
    The page's field values stay out: the question needs only its shape."""
    state = {
        "url": page.url,
        "title": page.title,
        "elements": [element_line(el, values=False) for el in page.elements[:80]],
        "text": page.text[:1500],
    }
    questions = {
        "signed_in": {
            "type": "noul",
            "instructions": (
                "Is a user signed in to this website on this page, shown by things like an account "
                "menu, an avatar, a user name or a sign-out link, rather than a sign-in form?"
            ),
        }
    }
    return state, questions
