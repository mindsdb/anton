"""Client for the render-check service: load a page the way the app preview does.

The service runs headless Chromium in its own container and loads the page
inside a sandboxed srcdoc frame under the preview's content security policy,
at a desktop and a phone width. A deployment that runs agent code in a
sandbox runs the service next to anton. anton sends the page's text and
reads back a verdict. Nothing runs in this process and nothing of this
process's environment is sent, so the check works where agent code runs in a
sandbox and no local browser may start.

The service's URL comes from `ANTON_RENDER_CHECK_URL` in this process's
environment, read at call time the way `html_lint` reads
`ANTON_HTML_LINT_BROWSER`. Unset, no caller asks the service anything.

Every way the service can fail to answer (not configured, unreachable, slow,
busy, an error status, a body that does not parse, a status its own findings
contradict) comes back as `RenderCheckUnavailable` with its reason, which
callers treat as an unverified page. Neither it nor a malformed answer ever
reads as a pass.
"""

from __future__ import annotations

import base64
import binascii
import errno
import logging
import os
import time
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple

import httpx2 as httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pydantic_core import PydanticCustomError

from anton.utils.datasources import scrub_output
from anton.utils.nofollow import read_bytes_nofollow

if TYPE_CHECKING:
    from anton.core.backends.sandbox import CellSandbox

_log = logging.getLogger(__name__)

RENDER_CHECK_ENV_VAR = "ANTON_RENDER_CHECK_URL"
# anton gives up on one check after this long. The service's own deadline
# for a whole check is 25 s, and after it the service still spends up to 3 s
# closing the page before it answers. The rest is room for a host short of
# CPU, so a slow page still comes back as a `timeout` finding before this
# fires.
TIMEOUT_SECONDS = 35.0
# A service busy with other checks answers 503 at once. anton asks once more
# after this wait, and only when at least `BUSY_RETRY_MIN_SECONDS` of the
# check's time limit would be left for the second ask. Never more than once.
BUSY_RETRY_SECONDS = 3.0
BUSY_RETRY_MIN_SECONDS = 5.0
# The largest page anton sends, counted in UTF-8 bytes, as on disk.
MAX_PAGE_BYTES = 8 * 1024 * 1024
# The service refuses a larger request body with 413; anton does not send
# one. JSON escaping doubles every quote, backslash and newline of the page,
# so a page under `MAX_PAGE_BYTES` with inline data needs up to twice its
# size, plus the rest of the request.
MAX_BODY_BYTES = 2 * MAX_PAGE_BYTES + 1024
# anton reads at most this much of an error answer, to tell a busy service
# from one that cannot check.
MAX_ERROR_BODY_BYTES = 4096
# anton stops reading an answer at this size and counts the page unverified.
# Room for the largest screenshot it keeps (`MAX_SCREENSHOT_BASE64_CHARS`)
# and many findings; a page that logs without end cannot make this process
# hold an answer of any size.
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
# At most this many distinct findings of one check reach the agent; the rest
# are counted. A page that logs one error per data row would otherwise put
# thousands of lines into one tool result, resent on every model call.
MAX_FINDINGS_SHOWN = 20
# A screenshot larger than this (base64 text) is not shown to the model. The
# service sends a JPEG at most 1024 px wide, far below it; the cap keeps a
# request with one screenshot well under the model route's 16 MB limit.
MAX_SCREENSHOT_BASE64_CHARS = 1024 * 1024

RenderStatus = Literal["passed", "failed", "unverified"]
FindingKind = Literal[
    "csp_violation",
    "separate_file",
    "console_error",
    "uncaught_exception",
    "blank_view",
    "overflow",
    "crashed",
    "timeout",
]
# Findings that fail a page whatever else the check saw or could not load.
_DEFINITE_KINDS = frozenset({"csp_violation", "separate_file"})
# Findings that fail a page even when the check ran out of time afterwards:
# the exception, error or crash happened whatever the check had left to do.
# A blank view or overflow measured on a slow host may still have been
# drawing, so a timeout leaves those unverified.
_FAILED_DESPITE_TIMEOUT_KINDS = frozenset({"uncaught_exception", "console_error", "crashed"})


class RenderCheckRequest(BaseModel):
    """`POST /v1/check` body: the entry file's text as it sits on disk.

    The service injects the preview shim and the `<base>` itself.
    """

    model_config = ConfigDict(extra="forbid")

    html: str
    screenshot: bool = True


class RenderFinding(BaseModel):
    """One problem seen in one viewport."""

    model_config = ConfigDict(extra="ignore")

    kind: FindingKind
    viewport: Literal["desktop", "phone"]
    detail: str


class UnreachableResource(BaseModel):
    """A URL the check could not load, so it cannot say how the page renders with it."""

    model_config = ConfigDict(extra="ignore")

    url: str
    reason: str


class RenderScreenshot(BaseModel):
    """The desktop viewport of the frame, base64-encoded."""

    model_config = ConfigDict(extra="ignore")

    media_type: Literal["image/jpeg", "image/png", "image/webp"]
    data: str

    @field_validator("data")
    @classmethod
    def _is_base64(cls, value: str) -> str:
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("screenshot data is not base64") from exc
        return value

    def image_block(self) -> dict:
        """The screenshot as an image content block, the shape `read_image` returns."""
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": self.media_type, "data": self.data},
        }


def _status_of(findings: Sequence[RenderFinding], *, unreachable: bool) -> RenderStatus:
    """The status the service gives a check with these findings, with or
    without resources it could not load: the rule `RenderCheckResult` lists."""
    kinds = {f.kind for f in findings}
    if kinds & _DEFINITE_KINDS:
        return "failed"
    if unreachable:
        return "unverified"
    if kinds & _FAILED_DESPITE_TIMEOUT_KINDS:
        return "failed"
    if "timeout" in kinds:
        return "unverified"
    if kinds:
        return "failed"
    return "passed"


# The pydantic error type of an answer whose status breaks the rule, so
# `check_html` can name that reason.
_STATUS_CONTRADICTS_FINDINGS = "status_contradicts_findings"


class RenderCheckResult(BaseModel):
    """The service's 200 answer. The service computes `status`:

    1. `failed` on any `csp_violation` or `separate_file` finding;
    2. otherwise `unverified` when `unreachable` is not empty;
    3. otherwise `failed` on any `uncaught_exception`, `console_error` or
       `crashed` finding, even beside a `timeout` one;
    4. otherwise `unverified` when a `timeout` finding exists, so a
       `blank_view` or `overflow` seen before the deadline stays unverified;
    5. otherwise `failed` on any other finding (`blank_view`, `overflow`);
    6. otherwise `passed`.

    An answer whose status breaks the rule does not validate, so a service
    that drifts from it reads as unverified, never as a pass.
    """

    model_config = ConfigDict(extra="ignore")

    status: RenderStatus
    findings: list[RenderFinding] = Field(default_factory=list)
    unreachable: list[UnreachableResource] = Field(default_factory=list)
    screenshot: RenderScreenshot | None = None
    policy: str = ""

    @model_validator(mode="after")
    def _status_follows_the_rule(self) -> "RenderCheckResult":
        expected = _status_of(self.findings, unreachable=bool(self.unreachable))
        if self.status != expected:
            raise PydanticCustomError(
                _STATUS_CONTRADICTS_FINDINGS,
                "status {status} contradicts the findings, which make it {expected}",
                {"status": self.status, "expected": expected},
            )
        return self

    @property
    def unverified_reason(self) -> str:
        """Why the check could not vouch for the page, in one line."""
        parts = [
            f"could not load {item.url} ({item.reason})" for item in self.unreachable[:3]
        ]
        if len(self.unreachable) > 3:
            parts.append(f"and {len(self.unreachable) - 3} more resources")
        parts += [
            f"the check ran out of time ({f.detail})" for f in self.findings if f.kind == "timeout"
        ][:1]
        return scrub_output("; ".join(parts) or "the render check could not verify the page")


class RenderCheckUnavailable(BaseModel):
    """No verdict from the service: the page counts as unverified."""

    model_config = ConfigDict(frozen=True)

    reason: str

    status: Literal["unverified"] = "unverified"

    @property
    def findings(self) -> list[RenderFinding]:
        return []

    @property
    def screenshot(self) -> None:
        return None

    @property
    def unverified_reason(self) -> str:
        return self.reason


RenderCheckOutcome = RenderCheckResult | RenderCheckUnavailable


class FindingGroup(NamedTuple):
    """One distinct finding and the viewports it was seen in."""

    kind: FindingKind
    detail: str
    # "desktop", "phone" or "desktop and phone".
    viewports: str

    @property
    def text(self) -> str:
        """`kind (viewports): detail`, the line a cell's result shows."""
        return f"{self.kind} ({self.viewports}): {self.detail}"


class GroupedFindings(NamedTuple):
    """The distinct findings to show, and how many more were left out."""

    shown: list[FindingGroup]
    more: int


def grouped_findings(findings: Sequence[RenderFinding]) -> GroupedFindings:
    """Each distinct finding once, in the order first seen, its detail scrubbed:
    the same problem in both viewports is one problem to fix. The first
    `MAX_FINDINGS_SHOWN` are shown and the rest counted."""
    seen: dict[tuple[str, str], list[str]] = {}
    for f in findings:
        seen.setdefault((f.kind, scrub_output(f.detail)), []).append(f.viewport)
    groups = [
        FindingGroup(kind=kind, detail=detail, viewports=" and ".join(dict.fromkeys(viewports)))
        for (kind, detail), viewports in seen.items()
    ]
    return GroupedFindings(shown=groups[:MAX_FINDINGS_SHOWN], more=max(len(groups) - MAX_FINDINGS_SHOWN, 0))


# Seen before the deadline, these may have drawn or settled given the time.
_MAY_STILL_DRAW_KINDS = frozenset({"blank_view", "overflow"})


class UnverifiedFindings(NamedTuple):
    """What an unverified check still found, worded to list under its
    `unverified_reason`."""

    # Introduces the lines: `before it ran out of time, the check found`
    # after a timeout, else `the check still found`.
    lead: str
    # One line per distinct finding, `kind (viewports): detail`.
    lines: list[str]
    # How many more distinct findings were left out.
    more: int


def unverified_findings(outcome: RenderCheckOutcome) -> UnverifiedFindings:
    """What an unverified check still found, grouped like `grouped_findings`.

    An exception beside a library the check could not load, say, or a
    `blank_view` or `overflow` seen before the deadline. After a timeout the
    lead says the findings came before it, and a blank view or overflow
    says the page may still have been drawing, which is why the service
    leaves those unverified (rule 4 in `RenderCheckResult`). No `timeout`
    finding, which `unverified_reason` already names. Empty for
    `RenderCheckUnavailable`, which holds no findings.
    """
    timed_out = any(f.kind == "timeout" for f in outcome.findings)
    grouped = grouped_findings([f for f in outcome.findings if f.kind != "timeout"])

    def line(group: FindingGroup) -> str:
        if timed_out and group.kind in _MAY_STILL_DRAW_KINDS:
            return f"{group.text} (it may still have been drawing)"
        return group.text

    return UnverifiedFindings(
        lead="before it ran out of time, the check found" if timed_out else "the check still found",
        lines=[line(group) for group in grouped.shown],
        more=grouped.more,
    )


def configured_url() -> str | None:
    """The service's base URL, or None when `ANTON_RENDER_CHECK_URL` is unset or blank."""
    value = os.environ.get(RENDER_CHECK_ENV_VAR, "").strip()
    return value.rstrip("/") or None


class _ServiceError(BaseModel):
    """The JSON body of the service's error answers."""

    model_config = ConfigDict(extra="ignore")

    detail: str = ""


class _Reply(NamedTuple):
    """One POST to the service: the body of its 200 answer, or why there is
    none and whether the service was busy with other checks."""

    content: bytes = b""
    unavailable: RenderCheckUnavailable | None = None
    busy: bool = False


def check_html(html: str, *, screenshot: bool = True, timeout: float = TIMEOUT_SECONDS) -> RenderCheckOutcome:
    """Ask the service to load `html`. Never raises.

    `timeout` is anton's limit for this call, a second ask included; a
    caller with a smaller budget left passes what remains. A service busy
    with other checks is asked once more after `BUSY_RETRY_SECONDS` when
    at least `BUSY_RETRY_MIN_SECONDS` of the limit would be left for the
    second ask. Its answer is the outcome, busy or not: there is no third.
    """
    base = configured_url()
    if base is None:
        return RenderCheckUnavailable(reason=f"no render-check service configured ({RENDER_CHECK_ENV_VAR} unset)")
    page_bytes = len(html.encode("utf-8"))
    if page_bytes > MAX_PAGE_BYTES:
        return RenderCheckUnavailable(
            reason=f"the page is {page_bytes // 1024} KB, over the render check's "
            f"{MAX_PAGE_BYTES // (1024 * 1024)} MB limit"
        )
    body = RenderCheckRequest(html=html, screenshot=screenshot).model_dump_json().encode("utf-8")
    if len(body) > MAX_BODY_BYTES:
        return RenderCheckUnavailable(
            reason=f"the page's request is {len(body) // 1024} KB once JSON-escaped, over the render "
            f"check's {MAX_BODY_BYTES // (1024 * 1024)} MB request limit"
        )
    started = time.monotonic()
    reply = _post(base, body=body, timeout=timeout)
    if reply.busy:
        left = timeout - (time.monotonic() - started) - BUSY_RETRY_SECONDS
        if left >= BUSY_RETRY_MIN_SECONDS:
            _log.info("render check at %s is busy; asking once more in %.0f s", base, BUSY_RETRY_SECONDS)
            time.sleep(BUSY_RETRY_SECONDS)
            reply = _post(base, body=body, timeout=left)
    if reply.unavailable is not None:
        return reply.unavailable
    try:
        result = RenderCheckResult.model_validate_json(reply.content)
    except ValidationError as exc:
        if any(error["type"] == _STATUS_CONTRADICTS_FINDINGS for error in exc.errors()):
            _log.warning("render check at %s answered a status its findings contradict: %s", base, exc)
            return RenderCheckUnavailable(
                reason="the render-check service answered a status its own findings contradict"
            )
        _log.warning("render check at %s answered a body that does not parse", base, exc_info=True)
        return RenderCheckUnavailable(reason="the render-check service answered something that does not parse")
    if result.screenshot is not None and len(result.screenshot.data) > MAX_SCREENSHOT_BASE64_CHARS:
        result = result.model_copy(update={"screenshot": None})
    return result


def _post(base: str, *, body: bytes, timeout: float) -> _Reply:
    """POST `body` to the service once. Never raises."""
    try:
        # trust_env=False: the service is a sibling container on an internal
        # network, never behind the egress proxy a deployment may set.
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            with client.stream(
                "POST", f"{base}/v1/check", content=body, headers={"Content-Type": "application/json"}
            ) as response:
                if response.status_code != 200:
                    return _Reply(
                        unavailable=RenderCheckUnavailable(reason=_status_reason(response.status_code)),
                        busy=response.status_code == 503 and _says_busy(response),
                    )
                content = bytearray()
                for chunk in response.iter_bytes():
                    content += chunk
                    if len(content) > MAX_RESPONSE_BYTES:
                        return _Reply(unavailable=RenderCheckUnavailable(
                            reason="the render check's answer is over "
                            f"{MAX_RESPONSE_BYTES // (1024 * 1024)} MB, more than anton reads"
                        ))
    except httpx.TimeoutException:
        return _Reply(unavailable=RenderCheckUnavailable(
            reason=f"the render check gave no answer within {timeout:.0f} s"
        ))
    except httpx.InvalidURL:
        # Not an `httpx.HTTPError`: a port that is not a number, an
        # unclosed IPv6 bracket.
        _log.warning("%s is not a valid URL: %r", RENDER_CHECK_ENV_VAR, base, exc_info=True)
        return _Reply(unavailable=RenderCheckUnavailable(reason=f"{RENDER_CHECK_ENV_VAR} is not a valid URL"))
    except httpx.HTTPError as exc:
        _log.warning("render check at %s could not be reached", base, exc_info=True)
        return _Reply(unavailable=RenderCheckUnavailable(
            reason=f"the render-check service could not be reached ({type(exc).__name__})"
        ))
    return _Reply(content=bytes(content))


def _says_busy(response: httpx.Response) -> bool:
    """Whether a 503 answer is the service's busy one: a check runs and the
    queue behind it is full, so asking again shortly may get a turn. Any
    other 503 (the service is not ready, the browser could not run) is
    not. Reads at most `MAX_ERROR_BODY_BYTES` of the answer."""
    content = bytearray()
    for chunk in response.iter_bytes():
        content += chunk
        if len(content) > MAX_ERROR_BODY_BYTES:
            return False
    try:
        return _ServiceError.model_validate_json(bytes(content)).detail.startswith("busy")
    except ValidationError:
        return False


def check_file(
    path: Path, *, cell_sandbox: CellSandbox | None, screenshot: bool = True, timeout: float = TIMEOUT_SECONDS
) -> RenderCheckOutcome:
    """Read the page at `path` and ask the service to load it. Never raises.

    This process reads the file outside the sandbox, so under a policy it
    reads only what the person's own cells may read. `allows_read` approves
    the page's resolved path, and `read_bytes_nofollow` then opens that path
    one folder at a time with O_NOFOLLOW, so the file read is the one
    approved: agent code that swaps a link in after the check makes the read
    fail instead of leading it out of the person's files. The read also
    refuses a FIFO, a device or a file with another hard link, and reads no
    more than the service accepts.
    """
    started = time.monotonic()
    try:
        # Inside the try: resolving a name that agent code is swapping can
        # itself fail.
        real = Path(os.path.realpath(path))
        if cell_sandbox is not None and not cell_sandbox.allows_read(real):
            return RenderCheckUnavailable(reason="the page is outside the files this conversation may read")
        data = read_bytes_nofollow(real.anchor, real, limit=MAX_PAGE_BYTES)
    except OSError as exc:
        if exc.errno == errno.EFBIG:
            return RenderCheckUnavailable(
                reason=f"the page is over the render check's {MAX_PAGE_BYTES // (1024 * 1024)} MB limit"
            )
        return RenderCheckUnavailable(reason=f"the page could not be read ({exc.strerror or type(exc).__name__})")
    remaining = timeout - (time.monotonic() - started)
    return check_html(data.decode("utf-8", errors="replace"), screenshot=screenshot, timeout=max(remaining, 1.0))


# Opens the caption beside every render screenshot in a tool result, so
# `drop_older_screenshots` can tell these images from one `read_image` returned.
SCREENSHOT_CAPTION = "[render check screenshot]"
OLDER_SCREENSHOT_PLACEHOLDER = (
    "[An older render-check screenshot was removed to keep the conversation small; "
    "the newest one is kept.]"
)
# Left by `drop_refused_screenshots`. `confirm_refused_screenshots` turns it
# into the note below once a request goes through without the screenshots.
DROPPED_SCREENSHOT_PLACEHOLDER = (
    "[This render-check screenshot was removed after the model refused a request "
    "that carried it. The findings above still apply.]"
)
REFUSED_SCREENSHOT_PLACEHOLDER = (
    "[The model refused a request with this render-check screenshot in it, so the "
    "screenshot was removed. The findings above still apply, and later checks in this "
    "conversation send their findings without a screenshot.]"
)
# Left by `drop_refused_tool_images` where a tool's image was.
# `confirm_refused_tool_images` turns it into the note below once a request
# goes through without the image.
DROPPED_TOOL_IMAGE_PLACEHOLDER = (
    "[The model provider refused a request that carried this image, so it was removed. "
    "Tell the user you could not view it.]"
)
REFUSED_TOOL_IMAGE_PLACEHOLDER = (
    "[The model refused a request with this image in it, so the image was removed. "
    "Tell the user you could not view it. read_image sends no more images in this conversation.]"
)


def screenshot_blocks(screenshot: RenderScreenshot, *, page: str) -> list[dict]:
    """The caption and the image, to follow a tool result's text."""
    return [
        {
            "type": "text",
            "text": f"{SCREENSHOT_CAPTION} {page} as the app preview shows it, at desktop width.",
        },
        screenshot.image_block(),
    ]


def _is_render_result(blocks: list) -> bool:
    return any(
        isinstance(b, dict) and b.get("type") == "text"
        and str(b.get("text", "")).startswith(SCREENSHOT_CAPTION)
        for b in blocks
    )


def _tool_results(messages: list[dict]) -> list[dict]:
    """The `tool_result` blocks in `messages` whose content is a list of blocks, oldest first."""
    found: list[dict] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            if isinstance(block.get("content"), list):
                found.append(block)
    return found


def _render_results(messages: list[dict]) -> list[dict]:
    """The `tool_result` blocks in `messages` that carry a render screenshot's caption, oldest first."""
    return [block for block in _tool_results(messages) if _is_render_result(block["content"])]


def _replace_blocks(sites: list[tuple[dict, int]], text: str) -> int:
    """Put a text block holding `text` at each site; return how many."""
    for block, index in sites:
        inner = list(block["content"])
        inner[index] = {"type": "text", "text": text}
        block["content"] = inner
    return len(sites)


def _sites(messages: list[dict], wanted: Callable[[dict], bool]) -> list[tuple[dict, int]]:
    """Each `wanted` block inside a render result in `messages`, as its
    `tool_result` block and its index inside it, oldest first."""
    return [
        (block, i)
        for block in _render_results(messages)
        for i, b in enumerate(block["content"]) if isinstance(b, dict) and wanted(b)
    ]


def _is_screenshot(block: dict) -> bool:
    return block.get("type") == "image"


def _is_note(text: str) -> Callable[[dict], bool]:
    return lambda block: block.get("type") == "text" and block.get("text") == text


def drop_older_screenshots(messages: list[dict]) -> int:
    """Keep only the newest render screenshot in `messages`; return how many
    older ones were replaced with a short placeholder.

    A screenshot lives in a `tool_result` block next to its caption. Every
    check can add one, and the whole history is sent on every model call, so
    without this they would pile up towards the model route's size limit.
    Edits the blocks in place, so a list that shares them (the turn's own
    messages) sees the same history.
    """
    return _replace_blocks(_sites(messages, _is_screenshot)[:-1], OLDER_SCREENSHOT_PLACEHOLDER)


def drop_refused_screenshots(messages: list[dict]) -> int:
    """Replace every render screenshot in `messages` with a note that it was
    removed after a refusal; return how many went.

    For a model that may take no image input. The caption and the findings
    beside each screenshot stay. The note claims no more than what
    happened: whether the screenshots caused the refusal is known only once
    a request goes through without them (`confirm_refused_screenshots`).
    Edits in place, like `drop_older_screenshots`.
    """
    return _replace_blocks(_sites(messages, _is_screenshot), DROPPED_SCREENSHOT_PLACEHOLDER)


def confirm_refused_screenshots(messages: list[dict]) -> int:
    """Turn each note `drop_refused_screenshots` left into the note that the
    model refused the screenshot and later checks come without one; return
    how many. For once a request without the screenshots went through.
    """
    return _replace_blocks(
        _sites(messages, _is_note(DROPPED_SCREENSHOT_PLACEHOLDER)), REFUSED_SCREENSHOT_PLACEHOLDER
    )


def screenshots_refused(messages: list[dict]) -> bool:
    """Whether `messages` hold a render screenshot the model refused earlier.

    A host that builds a new session for every turn (cowork-server) seeds it
    with the conversation's earlier tool results as
    `anton.cloud_turn.history_rows.split_turn_into_rows` cut them, which keeps
    their text blocks, the note `confirm_refused_screenshots` left included.
    So the new session learns from them that the model takes no image input.
    A note no request confirmed does not count.
    """
    return bool(_sites(messages, _is_note(REFUSED_SCREENSHOT_PLACEHOLDER)))


def drop_refused_tool_images(messages: list[dict]) -> int:
    """Replace every image a tool returned in `messages` (what `read_image`
    loaded, say) with a note that it was removed after a refusal; return how
    many went.

    The counterpart of `drop_refused_screenshots` for a model that may take
    no image input: a render result's screenshot is left to that function,
    and an image the person attached, which sits in their own message rather
    than in a tool result, to the session's repair of rejected content
    (`ChatSession._strip_image_blocks_from_history`). The note names no
    cause: the refusal may have had another
    (`confirm_refused_tool_images`). Both image shapes go, `image` and the
    OpenAI-style `image_url`. A second call finds nothing to drop. Edits in
    place, like `drop_older_screenshots`.
    """
    return _replace_blocks(
        _tool_image_sites(messages, wanted=lambda b: b.get("type") in ("image", "image_url")),
        DROPPED_TOOL_IMAGE_PLACEHOLDER,
    )


def confirm_refused_tool_images(messages: list[dict]) -> int:
    """Turn each note `drop_refused_tool_images` left into the note that the
    model refused the image and `read_image` sends no more; return how
    many. For once a request without the images went through.
    """
    return _replace_blocks(
        _tool_image_sites(messages, wanted=_is_note(DROPPED_TOOL_IMAGE_PLACEHOLDER)), REFUSED_TOOL_IMAGE_PLACEHOLDER
    )


def tool_images_refused(messages: list[dict]) -> bool:
    """Whether `messages` hold an image a tool returned that the model
    refused earlier, as `confirm_refused_tool_images` noted it.

    The counterpart of `screenshots_refused` for a host that builds a new
    session for every turn: the note survives in the tool result's text
    blocks, so the new session learns the model takes no image input. A
    note no request confirmed does not count.
    """
    return bool(_tool_image_sites(messages, wanted=_is_note(REFUSED_TOOL_IMAGE_PLACEHOLDER)))


def _tool_image_sites(messages: list[dict], *, wanted: Callable[[dict], bool]) -> list[tuple[dict, int]]:
    """Each `wanted` block inside a tool result in `messages` that is not a
    render result, as its `tool_result` block and its index inside it."""
    return [
        (block, i)
        for block in _tool_results(messages) if not _is_render_result(block["content"])
        for i, b in enumerate(block["content"]) if isinstance(b, dict) and wanted(b)
    ]


def _status_reason(status: int) -> str:
    if status == 503:
        return "the render-check service is busy or not ready (HTTP 503)"
    if status == 413:
        return "the page is over the render check's size limit (HTTP 413)"
    if status == 400:
        return "the render-check service rejected the request (HTTP 400)"
    return f"the render-check service answered HTTP {status}"
