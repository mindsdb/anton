"""The browser API's answers as typed models.

Both hosts (the MindsHub browser router and the Cowork desktop app's own
browser) answer the same routes with these shapes. The JSON files in
``tests/fixtures/browser_contract`` pin them, and both hosts' tests keep
copies of those files. A field a host leaves out that has no default here
makes the answer unreadable, and the client reports that as ``bad_response``.
Fields a host adds that are not listed here are ignored.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
)


class _Answer(BaseModel):
    model_config = ConfigDict(frozen=True)


class Challenge(_Answer):
    """A frame on the page that asks the person to prove they're human.

    Hosts list only frames the person can see (a non-empty box in the
    viewport, not hidden), so an invisible reCAPTCHA or Turnstile on a
    sign-in form never triggers a hand-over.
    """

    #: The frame's host, e.g. ``challenges.cloudflare.com``.
    host: str
    #: ``cloudflare``, ``recaptcha``, ``hcaptcha``, or another word for a check
    #: the host recognized by its frame but has no name for.
    kind: str


class _ElementFields(_Answer):
    """What an element of the page and an element inside a frame share:
    everything but the number a step names it by."""

    role: str
    name: str
    value: str | None = None
    #: The host withheld this field's value because it is secret (a password,
    #: a one-time code, a card number). The agent must never type into it.
    value_hidden: bool = False
    type: str = ""
    checked: bool | None = None
    disabled: bool = False
    offscreen: bool = False
    options: list[str] = []
    #: The names of the containers around it, innermost last ("Select dates ›
    #: November 2026"): what tells apart elements that share a name.
    context: str = ""
    #: Inside an open dialog, which is what a step on that page is about.
    in_dialog: bool = False

    @property
    def secret(self) -> bool:
        """A field only the person may fill."""
        return self.type == "password" or self.value_hidden


class Element(_ElementFields):
    """One numbered element of a page, as the agent reads and acts on it."""

    id: int


class FrameElement(_ElementFields):
    """An element inside a child frame. It has no number, so a step reaches
    it only by position, and a focused type (``FocusedType``) the field a
    click there left the focus in. Hosts never send the value of an input,
    a textarea or an editor (a select keeps its value and options)."""


class FrameBox(_Answer):
    """A frame's content box in the latest screenshot's pixels."""

    x: int
    y: int
    width: int
    height: int


class PagePoint(_Answer):
    """A point in a document's content coordinates, in whole CSS pixels."""

    x: int
    y: int


class Frame(_Answer):
    """A rendered child frame of the page, or a frame inside a frame the host
    read, listed unread after them (``Page.frames``). Hosts read a frame's
    elements and text only when it is safe to (``read``): never a human
    check, a payment, consent or sign-in frame, or one that holds a secret
    field."""

    #: Origin and path for http and https, ``about:blank`` or ``about:srcdoc``,
    #: otherwise the scheme alone.
    url: str
    read: bool = False
    #: None when the host can't place the frame on the screenshot.
    box: FrameBox | None = None
    #: The frame element's top-left in its document: its top-left in the
    #: viewport plus the page's scroll plus the scroll offset of every
    #: scrolling element it sits in, so no scroll, the page's or an inner
    #: column's, moves it. Both hosts send it with every frame that has a box;
    #: None for a frame with no box, and from a host that predates it.
    at: PagePoint | None = None
    offscreen: bool = False
    elements: list[FrameElement] = []
    text: str = ""
    truncated: bool = False


class Scroll(_Answer):
    y: int
    height: int
    viewport: int


class Page(_Answer):
    """``GET /sessions/{id}/page``: the active tab as numbered elements and text."""

    url: str
    title: str
    elements: list[Element]
    text: str = ""
    truncated: bool = False
    scroll: Scroll | None = None
    #: An opaque token naming the focused node, the deepest one through the
    #: frames the host can read; when the focus is inside a frame it can't
    #: read (a payment frame, another site's frame inside another site's, a
    #: frame past its depth cap), the token names that frame's element. Equal
    #: tokens are the same node. The desktop app's browser sends it, and
    #: leaves it out only when nothing has the focus or it can't read the
    #: page at all; the router's pod, which reads the focus again when a step
    #: runs, leaves it out.
    focus: str | None = None
    #: Check frames the person can see; the router's pod always sends it (empty
    #: when there are none) and leaves out invisible checks. None from a host
    #: that doesn't say.
    challenges: list[Challenge] | None = None
    #: The page's rendered child frames, on-screen first, and after them a
    #: host may list, unread, the frames inside a frame it read. Their
    #: elements never carry a number, so nothing a step names by number is
    #: inside one. Both hosts always send it (empty when there are none);
    #: None from a host that predates it, which hasn't said where its frames
    #: are.
    frames: list[Frame] | None = None


class ActionResult(_Answer):
    """``POST /sessions/{id}/action``; ``page`` is the page after the step,
    when the request asked for it."""

    ok: bool
    url: str
    #: Whether the step changed what the page shows; None when the host can't
    #: tell (a frame, canvas or media element, a hover, a wait) or doesn't say.
    changed: bool | None = None
    page: Page | None = None


class Control(_Answer):
    """Who holds the browser.

    ``seq`` rises by one with every change. ``by`` says who made the change:
    the agent hands the browser to the person (``holder`` human, ``by``
    agent), and only the person hands it back, with Done in the browser
    (``holder`` agent, ``by`` person). A host refuses an API request for
    ``holder`` agent with 403 ``person_only``.
    """

    holder: Literal["agent", "human"]
    reason: str
    #: Epoch milliseconds of the last change.
    since: int
    seq: int
    by: Literal["agent", "person"]


class SessionInfo(_Answer):
    """``GET /sessions/{id}``, which answers even while the person holds control."""

    session_id: str
    control: Control
    #: The active tab's address; null before a page loads.
    url: str | None = None
    #: People watching it now: the router's viewers; None from a host that doesn't say.
    viewers: int | None = None
    #: The most tabs the agent may open for this task; None from a host that has no limit.
    max_tabs: int | None = None
    #: Tabs a recycled session noted, which reopen only when asked for.
    parked_tabs: list["ParkedTab"] = []


class ParkedTab(_Answer):
    url: str
    title: str = ""


class Tab(_Answer):
    id: str
    url: str
    title: str
    active: bool


class Site(_Answer):
    """A site the person signed in to in this browser, as the browser records it."""

    site: str
    account_hint: str = ""
    noted_at: str = ""
    cookies_present: bool = False
    cookies_expire_at: str | None = None


class SavedTab(_Answer):
    """A page saved to a project's tabs."""

    name: str
    url: str
    notes: str = ""
    saved_at: str = ""


class HeldStep(_Answer):
    """What a held step's approval will run, from the browser's own read of the
    page in the read that would run it: the hold, the step's action, the names
    of the elements that make it a commit (none for a hold by position) and
    the page address it is bound to."""

    kind: Literal["commit", "position"]
    action: str
    names: list[str] = []
    url: str


class FocusedType(BaseModel):
    """A type step into the field that has the focus, a frame's field
    included: no element and no point, since a frame's fields have no number.
    Both hosts type it only when the focus is on a text entry that isn't
    secret, and otherwise type nothing and refuse it (403 password_field, 409
    not_a_text_field, 409 focus_unreadable). It types at the caret, as keys
    do, and clears nothing first. ``submit`` presses Enter after the text; it
    is always sent, so both hosts read one shape."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    action: Literal["type"]
    focused: Literal[True]
    text: str
    submit: bool = False


class StepKey(BaseModel):
    """A step as the tool compares two of them: its own fields with their
    defaults dropped, and nothing the request adds (the page it asks back for,
    the point's scale, an approval's id), so a resend that names a default
    ("button": "left") is the same step."""

    model_config = ConfigDict(extra="allow")

    action: str
    button: str = "left"
    clicks: int = 1
    submit: bool = False
    clear: bool = False
    include_page: bool | None = Field(default=None, exclude=True)
    coords: str | None = Field(default=None, exclude=True)
    pending_id: str | None = Field(default=None, exclude=True)

    def key(self) -> str:
        return json.dumps(self.model_dump(exclude_defaults=True), sort_keys=True, default=str)


class ErrorBody(_Answer):
    """Every non-2xx answer: ``{"error": <code>, "message": <text>}``.

    ``approval_required`` adds the held step's ``pending_id``, the Unix time
    its approval lapses, ``expires_at``, and what the approval will run,
    ``held``."""

    error: str
    message: str = ""
    pending_id: str | None = None
    expires_at: int | None = None
    held: HeldStep | None = None

    @field_validator("held", mode="wrap")
    @classmethod
    def _unreadable_held_is_none(
        cls, value: object, handler: ValidatorFunctionWrapHandler
    ) -> HeldStep | None:
        # A held step stays held whatever its description says: an unreadable
        # one makes the card fall back to the tool's own read, not the
        # refusal unreadable.
        try:
            return handler(value)
        except ValidationError:
            return None


class TabsAnswer(_Answer):
    tabs: list[Tab]


class TabAnswer(_Answer):
    tab: Tab


class SitesAnswer(_Answer):
    sites: list[Site]


class SiteAnswer(_Answer):
    site: Site


class SavedTabsAnswer(_Answer):
    tabs: list[SavedTab]


class SavedTabAnswer(_Answer):
    tab: SavedTab
