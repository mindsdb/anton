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


class Element(_Answer):
    """One numbered element of a page, as the agent reads and acts on it."""

    id: int
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

    @property
    def secret(self) -> bool:
        """A field only the person may fill."""
        return self.type == "password" or self.value_hidden


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
    challenges: list[Challenge] = []


class ActionResult(_Answer):
    """``POST /sessions/{id}/action``; ``page`` is the page after the step,
    when the request asked for it."""

    ok: bool
    url: str
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
    #: People watching it now: the router's viewers; 0 from a host that doesn't say.
    viewers: int = 0
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
