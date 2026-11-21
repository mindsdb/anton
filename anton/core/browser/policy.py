"""Jev as the browser's step policy.

One ``/v1/decisions`` call per step asks four things about the same page:
what kind of step comes next, which element it acts on, whether a person
has to act first, and whether the step would commit something hard to undo.
The tool acts on a confident answer and hands anything else back to the
model, the way the completion verifier uses Jev (see session._jev_verdict).
"""

from __future__ import annotations

from dataclasses import dataclass

from anton.core.browser.models import Element, Page
from anton.core.browser.page import element_line, shortlist

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
    "done": "The goal is achieved, or the information the goal asks for is visible on this page now.",
    "unsure": "None of the other options clearly fits.",
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


def step_request(
    *, goal: str, page: Page, inputs: dict | None, history: list[str], redact: bool,
) -> tuple[dict, dict, list[Element]]:
    """(state, questions, shortlisted elements) for one step.

    ``redact`` leaves out what is in the page's fields and the values in
    ``inputs``, for a call that only measures how often Jev agrees with the
    model: the ``input_key`` question still lists the inputs by name.
    """
    candidates = shortlist(page.elements, goal, inputs)
    state = {
        "goal": goal,
        "page": {
            "url": page.url,
            "title": page.title,
            "elements": [element_line(el, values=not redact) for el in candidates],
            "text": page.text[:1500],
        },
        "steps_so_far": history[-8:],
    }
    if inputs and not redact:
        state["inputs"] = {str(k): str(v) for k, v in inputs.items()}

    targets = {f"e{el.id}": element_line(el, values=not redact) for el in candidates}
    targets[NO_TARGET] = "No listed element is the right one to act on next."
    questions: dict = {
        "next_action": {
            "type": "choice",
            "instructions": "Decide the single next browser step that best advances the goal on this page.",
            "criteria": ACTIONS,
        },
        "target": {
            "type": "choice",
            "instructions": "Pick the element the next step should act on to advance the goal.",
            "criteria": targets,
        },
        "needs_user": {
            "type": "noul",
            "instructions": (
                "Does this page require the human user to act before the goal can continue, such as "
                "signing in, solving a CAPTCHA, entering a two-factor code, accepting terms, or "
                "confirming a payment?"
            ),
        },
        "risky": {
            "type": "noul",
            "instructions": (
                "Would the most likely next step on this page commit something hard to undo on the "
                "user's behalf: a purchase or payment, sending a message or email, deleting data, "
                "or submitting a form?"
            ),
        },
    }
    if inputs:
        keys = {str(k): f"The value provided as {k!r}." for k in inputs}
        keys[NO_INPUT] = "None of the provided values belongs in the field typed into next."
        questions["input_key"] = {
            "type": "choice",
            "instructions": "Which provided input value belongs in the field the next step types into?",
            "criteria": keys,
        }
    return state, questions, candidates


def parse_step(decision) -> StepDecision:
    """A ``jev.JevDecision`` as a StepDecision; an error carries through unchanged."""
    if decision.error:
        return StepDecision(ms=decision.ms, error=decision.error)
    action, action_p = decision.choice("next_action")
    target_name, target_p = decision.choice("target")
    target = None
    if target_name.startswith("e") and target_name[1:].isdigit():
        target = int(target_name[1:])
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
