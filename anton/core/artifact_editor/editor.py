"""Change a small artifact in one or two model calls, and prove it still runs.

A port of yolocoder's change loop (``internal/agent/agent.go``) and Mochi's
newer TypeScript copy of it (``packages/runtime/src/builder/agent.ts``),
rebuilt on anton's LLM client, :class:`ArtifactWorkspace` and the patch engine.

One tool-using conversation per edit:

1. The artifact's file list leads (what changes least, for the prefix cache);
   a small artifact is then preloaded whole as synthetic ``read_files``
   call/result pairs, code files first, so the first call can edit instead of
   asking for files; the request comes last.
2. The model reads (``read_files``, ``search``) and edits (``apply_diff``, a
   compact patch). Each tool has its own budget: reads 8, search 5,
   apply_diff 4, and the conversation at most 20 rounds. Every apply_diff
   result says how many edit calls are left.
3. A patch that only partly places writes the files that did place and sends
   back every failure at once, with the failed files as they now stand.
4. ``task_complete=true`` ends the turn without a "done" round trip. Any later
   tool call clears it; a failed call in the same batch cancels it.
5. Only when the model says it is done is the artifact checked (static, then a
   browser). A failed check goes back into the same conversation ("The edits
   applied, but the check failed. Fix it."), with the screenshot when the
   model takes images. "Unverified" is said plainly, never treated as passed.
6. When the edit budget runs out with edits that would not place, the files
   are asked for whole, one file per call: the last resort, never used when
   the model itself decided the task cannot be done.
7. Giving up (a second failed check, a crash, a cancellation) restores every
   file the edit touched to what it was before the edit.
8. With ``escalation=True`` (the quick-edit bar, ENG-4227) the model also has
   ``hand_to_agent``: a request that needs data or analysis these files do not
   have is handed back (status ``escalated``) and every file is restored.
   ``max_fix_rounds`` then bounds how many failed checks go back for repair.

Every rejected patch is written to the editor's JSONL record (``records``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from anton.core.artifact_editor import prompts
from anton.core.artifact_editor.checkers import CheckResult, Checker
from anton.core.artifact_editor.patch import PatchParseError, apply_patch, parse_patch
from anton.core.artifact_editor.records import EditRecord
from anton.core.artifact_editor.workspace import ArtifactWorkspace, Snapshot, WorkspaceError

if TYPE_CHECKING:
    from anton.core.llm.provider import LLMResponse, ToolCall

__all__ = ["ArtifactEditor", "EditUsage", "Outcome", "Progress"]

_log = logging.getLogger(__name__)

MAX_ROUNDS = 20
QUOTA = {prompts.READ_FILES: 8, prompts.SEARCH: 5, prompts.APPLY_DIFF: 4}
MAX_CUT_OFFS = 3
MAX_EMPTIES = 3
# Failed browser checks one edit may see (static failures are bounded by the
# edit budget instead). The second one ends the edit and restores the snapshot: "≤2 calls with a repair" is the target, and a page that fails
# twice is not getting fixed by a third guess.
MAX_CHECK_FAILURES = 2
# Transient model failures (a dropped stream, a 5xx) get one more try.
MODEL_TRIES = 2
RETRY_WAIT_S = 1.5
# A narrow search attaches the files it matched in when there are at most
# this many and they are small.
SEARCH_ATTACH_FILES = 4
SEARCH_ATTACH_FILE_CHARS = 32 * 1024
SEARCH_ATTACH_TOTAL_CHARS = 64 * 1024
# Words the model writes beside its tool calls stay in the conversation, up to this.
MAX_ASIDE_CHARS = 800
# The cut-off ladder: one step down per cut-off reply.
_EFFORT_LADDER = ("max", "xhigh", "high", "medium", "low")

Status = Literal["applied", "answered", "restored", "escalated"]


class Progress(Protocol):
    """How the editor reports. A Progress may also have ``stage(name, **info)``:
    "reading", "writing", "checking" or "fixing", called when the edit enters it."""

    def status(self, text: str) -> None: ...  # a transient activity line
    def log(self, text: str) -> None: ...  # a permanent line of the trail


class _NoProgress:
    def status(self, text: str) -> None:
        pass

    def log(self, text: str) -> None:
        pass


@dataclass
class EditUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def add(self, usage: Any) -> None:
        self.calls += 1
        if usage is None:
            return
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        self.cache_read_tokens += int(getattr(usage, "cache_read_tokens", 0) or 0)
        self.cache_creation_tokens += int(getattr(usage, "cache_creation_tokens", 0) or 0)


@dataclass
class Outcome:
    """What one edit amounted to, built from what happened, not from what the model said.

    ``status``:
    * ``applied`` — edits landed and the last check passed or was unverified
      (``check.status`` says which);
    * ``answered`` — the model changed nothing and replied (a question, or
      "this cannot be done by editing these files");
    * ``restored`` — the edit gave up and every file it touched is back as it
      was; ``error`` says why;
    * ``escalated`` — (only with ``escalation=True``) the model said the request
      needs more than these files; every file is back as it was and ``reply``
      is its reason.
    """

    status: Status
    reply: str
    files: list[str] = field(default_factory=list)
    check: CheckResult | None = None
    attempts: int = 0
    rewrote: bool = False
    usage: EditUsage = field(default_factory=EditUsage)
    error: str = ""
    seconds: float = 0.0
    # Failed checks that went back to the model for repair.
    fix_rounds: int = 0

    @property
    def ok(self) -> bool:
        return self.status not in ("restored", "escalated")

    @property
    def model_calls(self) -> int:
        return self.usage.calls


class _GiveUp(Exception):
    """The edit cannot finish; restore and report `str(self)`."""

    def __init__(self, message: str, check: CheckResult | None = None) -> None:
        super().__init__(message)
        self.check = check


class _OutOfRounds(Exception):
    pass


class _Escalate(Exception):
    """The model handed the request to the agent; restore and report `str(self)`."""


@dataclass
class _ToolResult:
    output: str
    ok: bool
    detail: list[str] = field(default_factory=list)


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "..."


def _seconds(started: float) -> str:
    spent = time.monotonic() - started
    return f"{spent * 1000:.0f}ms" if spent < 1 else f"{spent:.1f}s"


def decode_json_object(text: str) -> dict | None:
    """A JSON object from a reply, tolerating a fence or prose around it."""
    text = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except ValueError:
        try:
            value = json.loads(text[start:end + 1], strict=False)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


_OBSERVED_JSON = re.compile(
    r"<script\b(?=[^>]*\bid=[\"']cowork-observed-json[\"'])[^>]*>(.*?)</script\s*>", re.S | re.I)
MAX_SHAPE_CHARS = 2500


def _shape(value: Any, depth: int = 0) -> str:
    """A compact picture of a JSON value: keys and types, the first element of each list."""
    if isinstance(value, dict):
        if depth >= 4:
            return "{...}"
        items = list(value.items())
        parts = [f"{json.dumps(k)}: {_shape(v, depth + 1)}" for k, v in items[:12]]
        if len(items) > 12:
            parts.append(f"... {len(items) - 12} more keys")
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        return f"[{len(value)} x {_shape(value[0], depth + 1)}]"
    if isinstance(value, str):
        return json.dumps(value if len(value) <= 40 else value[:40] + "...")
    return json.dumps(value)


def observed_data_shape(page: str) -> str:
    """The shape of a generated page's embedded data (window.ARTIFACT_DATA), or "".

    The workspace shows that data as one placeholder line so a large page stays
    editable; without its shape the model cannot tell which keys the code
    should read, which is exactly what a data bug needs.
    """
    match = _OBSERVED_JSON.search(page)
    if not match:
        return ""
    try:
        value = json.loads(match.group(1))
    except ValueError:
        return ""
    text = _shape(value)
    return text if len(text) <= MAX_SHAPE_CHARS else text[:MAX_SHAPE_CHARS] + "..."


class ArtifactEditor:
    """Edit one artifact folder with one model conversation.

    ``llm`` is anton's :class:`LLMClient` (or anything with the same
    ``plan_stream`` / ``code_stream``); ``role`` picks which of its models
    edits. ``checker`` is asked only when the model says it is done.
    """

    def __init__(
        self,
        workspace: ArtifactWorkspace,
        llm: Any,
        checker: Checker,
        progress: Progress | None = None,
        *,
        role: Literal["planning", "coding"] = "planning",
        slug: str = "",
        name: str = "",
        images: bool = True,
        record: EditRecord | None = None,
        conversation_id: str = "",
        escalation: bool = False,
        max_fix_rounds: int | None = None,
    ) -> None:
        self.ws = workspace
        self.llm = llm
        self.checker = checker
        self.progress: Progress = progress or _NoProgress()
        self.role = role
        self.slug = slug or workspace.folder.name
        self.name = name
        self.images = images
        self.record = record
        self.conversation_id = conversation_id
        self.escalation = escalation
        # None: the default bounds (static failures by the edit budget, browser
        # failures by MAX_CHECK_FAILURES). A number: at most that many failed
        # checks of any channel go back for repair; the next one gives up.
        self.max_fix_rounds = max_fix_rounds
        self.system = prompts.INSTRUCTIONS + (prompts.QUICK_EDIT_INSTRUCTIONS if escalation else "")
        self.tools = [*prompts.TOOLS, prompts.HAND_TO_AGENT_TOOL] if escalation else prompts.TOOLS
        self._reset()

    def _reset(self) -> None:
        self.messages: list[dict] = []
        self.served: dict[str, str] = {}
        self.used: dict[str, int] = {}
        self.applied: list[str] = []
        self.attempted: list[str] = []
        self.unplaced: list[str] = []
        self.read_paths: list[str] = []
        self.last_failure = ""
        self.complete = False
        self.closing = ""
        self.cut_offs = 0
        self.empties = 0
        self.check_failures = 0
        self.fix_rounds = 0
        self._stage_now = ""
        self.last_check: CheckResult | None = None
        self.usage = EditUsage()
        self.effort: str | None = None
        self._snapshots: list[Snapshot] = []
        self._snapshotted: set[str] = set()
        self._restored = False

    # ── the run ──────────────────────────────────────────────────────

    async def edit(self, task: str) -> Outcome:
        """Make the change ``task`` asks for. Never raises except on cancellation.

        A cancelled edit restores the snapshot and re-raises the cancellation.
        """
        self._reset()
        started = time.monotonic()
        if self.record is None:
            self.record = EditRecord(self.ws.folder, task, slug=self.slug, conversation_id=self.conversation_id)
        try:
            outcome = await self._run(task)
        except asyncio.CancelledError:
            self._restore()
            self._record_run("cancelled", rewrote=False, error="cancelled", started=started)
            raise
        except _Escalate as exc:
            self._restore()
            outcome = Outcome("escalated", reply=str(exc), check=self.last_check)
        except _GiveUp as exc:
            self._restore()
            outcome = Outcome("restored", reply="", check=exc.check or self.last_check, error=str(exc))
        except Exception as exc:  # noqa: BLE001 — any crash leaves the folder as it was
            _log.warning("the artifact editor failed", exc_info=True)
            self._restore()
            outcome = Outcome("restored", reply="", check=self.last_check, error=f"{type(exc).__name__}: {exc}")
        outcome.attempts = self.used.get(prompts.APPLY_DIFF, 0)
        outcome.usage = self.usage
        outcome.seconds = time.monotonic() - started
        outcome.fix_rounds = self.fix_rounds
        if outcome.status in ("restored", "escalated"):
            outcome.files = []
        self._record_run(outcome.status, rewrote=outcome.rewrote, error=outcome.error, started=started,
                         outcome=outcome)
        return outcome

    def _record_run(self, status: str, *, rewrote: bool, error: str, started: float,
                    outcome: Outcome | None = None) -> None:
        if self.record is None:
            return
        check = outcome.check if outcome is not None else self.last_check
        self.record.run(
            status=status, model_calls=self.usage.calls, attempts=self.used.get(prompts.APPLY_DIFF, 0),
            rewrote=rewrote, files=list(self.applied), snapshot_paths=sorted(self._snapshotted),
            check_status=check.status if check is not None else None,
            check_channel=check.channel if check is not None else None,
            error=error, seconds=round(time.monotonic() - started, 2),
            usage={"input": self.usage.input_tokens, "output": self.usage.output_tokens,
                   "cache_read": self.usage.cache_read_tokens, "cache_creation": self.usage.cache_creation_tokens},
        )

    def _stage(self, name: str, **info: Any) -> None:
        """Tell a Progress that has ``stage`` which part of the edit this is (once per change)."""
        if name == self._stage_now and not info:
            return
        self._stage_now = name
        stage = getattr(self.progress, "stage", None)
        if callable(stage):
            try:
                stage(name, **info)
            except Exception:  # noqa: BLE001 — a reporting failure never fails the edit
                _log.debug("a progress stage callback failed", exc_info=True)

    async def _run(self, task: str) -> Outcome:
        self._stage("reading")
        self.progress.status("Looking over the artifact...")
        entries = self.ws.map()
        file_list = "\n".join(f"{e.path} ({e.bytes} bytes)" for e in entries) or "(no files)"
        intro = prompts.intro(kind=self.ws.kind, slug=self.slug, name=self.name, primary=self.ws.primary,
                              file_list=file_list)
        shapes = []
        for entry in entries:
            if entry.path.lower().endswith((".html", ".htm")):
                try:
                    raw = (self.ws.folder / entry.path).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                shape = observed_data_shape(raw)
                if shape:
                    shapes.append(f"{entry.path}: window.ARTIFACT_DATA = {shape}")
        if shapes:
            intro += ("\n\nEMBEDDED DATA (shown in the page as an anton:keep line; its shape, with the first item "
                      "of each list):\n" + "\n".join(shapes))
        preload = await asyncio.to_thread(self.ws.preload)
        if preload:
            self.messages.append({"role": "user", "content": intro})
            self._preload(list(preload))
            self.progress.log(f"read the whole artifact up front: {len(preload)} files")
            self._say(prompts.request(task, preloaded=True))
        else:
            primary = self.ws.primary
            self.messages.append({"role": "user", "content": intro})
            # The entry page first: nearly every edit is to it.
            if primary and any(e.path == primary for e in entries):
                self._preload([primary])
            self._say(prompts.request(task, preloaded=False))

        self._stage("writing")
        self.progress.status("Working out the change...")
        try:
            outcome = await self._work()
        except _OutOfRounds:
            if self.attempted:
                return await self._rewrite_instead(task)
            raise _GiveUp(f"the model did not finish after {MAX_ROUNDS} rounds") from None
        out_of_edits = (
            (not self.applied or self.unplaced) and self.used.get(prompts.APPLY_DIFF, 0) >= QUOTA[prompts.APPLY_DIFF]
        )
        if self.attempted and out_of_edits:
            return await self._rewrite_instead(task)
        return outcome

    # ── the conversation ─────────────────────────────────────────────

    def _preload(self, paths: list[str]) -> None:
        """Files as read_files results the model can see it asked for."""
        for i in range(0, len(paths), prompts.MAX_READ_PATHS):
            chunk = paths[i:i + prompts.MAX_READ_PATHS]
            call_id = f"preload_{i // prompts.MAX_READ_PATHS + 1}"
            self.messages.append({"role": "assistant", "content": [{
                "type": "tool_use", "id": call_id, "name": prompts.READ_FILES,
                "input": {"paths": chunk, "matching": ""},
            }]})
            self.messages.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": call_id, "content": self._read_files(chunk),
            }]})
            self.read_paths += chunk

    def _say(self, text: str, *, image: tuple[str, str] | None = None) -> None:
        """Add TEXT (and an image) to the conversation as the user's, joining a trailing user message."""
        blocks: list[dict] = [{"type": "text", "text": text}]
        if image is not None and self.images:
            data, media_type = image
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
        last = self.messages[-1] if self.messages else None
        if last is not None and last["role"] == "user":
            content = last["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            last["content"] = [*content, *blocks]
            return
        self.messages.append({"role": "user", "content": blocks if len(blocks) > 1 else text})

    def _drop_images(self) -> bool:
        dropped = False
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, list):
                kept = [b for b in content if not (isinstance(b, dict) and b.get("type") == "image")]
                if len(kept) != len(content):
                    message["content"] = kept
                    dropped = True
        return dropped

    async def _work(self) -> Outcome:
        from anton.core.llm.provider import ensure_replayable_tool_call

        for _round in range(MAX_ROUNDS):
            started = time.monotonic()
            response, budget = await self._create(system=self.system, tools=self.tools)
            self.progress.log(f"thought · {_seconds(started)}")
            calls = [ensure_replayable_tool_call(tc) for tc in (response.tool_calls or [])]
            cut = self._cut_off(response, budget)
            intact = [tc for tc in calls if not (tc.parse_error or tc.repaired)]

            if intact or not cut:
                self.cut_offs = 0
            else:
                self.cut_offs += 1
                if self.cut_offs > MAX_CUT_OFFS:
                    message = f"the model ran out of output room {self.cut_offs} times in a row without a reply"
                    raise _GiveUp(message)
                self._lower_effort()

            if not calls:
                reply = (response.content or "").strip()
                if cut:
                    # Partial text is not a terminal answer, even when nonempty.
                    self.complete, self.closing = False, ""
                    self.progress.log("reply was cut off, asking again...")
                    self._assistant(reply, [])
                    self._say(prompts.CUT_OFF_NO_CALL)
                    continue
                if not reply:
                    self.empties += 1
                    if self.empties <= MAX_EMPTIES:
                        self.progress.log("empty reply, nudging...")
                        self._say(prompts.EMPTY_REPLY)
                        continue
                    if not self.applied:
                        raise _GiveUp("the model kept replying with nothing")
                    result = await self._check()
                    if result.status == "failed":
                        raise _GiveUp("the model stopped answering, and the check failed", result)
                    return self._outcome("Updated " + ", ".join(self.applied) + ".", result)
                self.empties = 0
                self._assistant(reply, [])
                if self.applied:
                    result = await self._check()
                    if result.status == "failed":
                        self._check_failed(result)
                        continue
                    return self._outcome(reply, result)
                return Outcome("answered", reply=reply)

            self.empties = 0
            aside = (response.content or "").strip()
            self._assistant(aside[:MAX_ASIDE_CHARS], calls)
            results: list[dict] = []
            batch_ok = True
            for tc in calls:
                await asyncio.sleep(0)  # a cancellation lands between tools
                result = await self._run_tool(tc)
                batch_ok = batch_ok and result.ok
                for line in result.detail:
                    self.progress.log("  " + line)
                results.append({"type": "tool_result", "tool_use_id": tc.id, "content": result.output})
            self.messages.append({"role": "user", "content": results})

            if cut:
                self.complete, self.closing = False, ""
                self.progress.log("completion deferred: the reply was cut off after its tool calls")
                self._say(prompts.CUT_OFF_AFTER_CALLS)
                continue
            if not batch_ok:
                # The model declared completion before seeing this batch's results.
                self.complete, self.closing = False, ""
            if self.complete:
                result = await self._check()
                if result.status == "failed":
                    self.complete, self.closing = False, ""
                    self._check_failed(result)
                    continue
                return self._outcome(self.closing or "Updated " + ", ".join(self.applied) + ".", result)
        raise _OutOfRounds()

    def _assistant(self, text: str, calls: Sequence[ToolCall]) -> None:
        blocks: list[dict] = []
        if text:
            blocks.append({"type": "text", "text": text})
        for tc in calls:
            blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input or {}})
        if blocks:
            self.messages.append({"role": "assistant", "content": blocks})

    def _cut_off(self, response: LLMResponse, budget: int) -> bool:
        from anton.core.llm.structured import looks_truncated

        return looks_truncated(response, budget)

    def _provider(self):
        name = "_coding_provider" if self.role == "coding" else "_planning_provider"
        return getattr(self.llm, name, None)

    def _lower_effort(self) -> None:
        """One step down the effort ladder: a model that spent its room thinking thinks the same way again."""
        current = self.effort
        if current is None:
            effective = getattr(type(self.llm), "_effective_effort", None)
            provider = self._provider()
            if callable(effective) and provider is not None:
                try:
                    current = effective(provider)
                except Exception:  # noqa: BLE001
                    current = None
        if current in _EFFORT_LADDER and current != _EFFORT_LADDER[-1]:
            self.effort = _EFFORT_LADDER[_EFFORT_LADDER.index(current) + 1]
            self.progress.log(f"asking again with reasoning effort {self.effort}")

    async def _create(self, *, system: str, tools: list[dict] | None, max_tokens: int | None = None,
                      messages: list[dict] | None = None) -> tuple[LLMResponse, int]:
        """One streamed model call (the gateway cuts non-streamed calls at ~110 s)."""
        from anton.core.llm.effort import client_stream_budget
        from anton.core.llm.provider import (
            ModelCallTimeoutError,
            ProviderOverloadedError,
            TransientProviderError,
            is_image_refusal,
        )

        budget = max_tokens or client_stream_budget(self.llm, self.role)
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._stream_once(system=system, tools=tools, budget=budget,
                                                   messages=messages if messages is not None else self.messages)
                break
            except (TransientProviderError, ProviderOverloadedError, ModelCallTimeoutError):
                if attempt >= MODEL_TRIES:
                    raise
                self.progress.log("the model provider hiccuped; trying again")
                await asyncio.sleep(RETRY_WAIT_S)
            except Exception as exc:
                if self.images and messages is None and is_image_refusal(exc) and self._drop_images():
                    # The model takes no images: the findings stand without the screenshot.
                    self.images = False
                    self.progress.log("the model refused the screenshot; sending the findings without it")
                    continue
                raise
        self.usage.add(response.usage)
        return response, budget

    async def _stream_once(self, *, system: str, tools: list[dict] | None, budget: int,
                           messages: list[dict]) -> LLMResponse:
        from anton.core.llm.client import ModelRoute, asked_model, model_route
        from anton.core.llm.provider import LLMResponse, StreamComplete

        stream_fn = self.llm.code_stream if self.role == "coding" else self.llm.plan_stream
        route = None
        if self.effort is not None:
            model = asked_model(self.llm, self.role)
            if model:
                route = ModelRoute(model=model, effort=self.effort)
        final: LLMResponse | None = None
        # The route is entered in this coroutine, around the whole consumption,
        # never inside a generator.
        with model_route(route):
            stream = stream_fn(system=system, messages=list(messages), tools=tools, max_tokens=budget)
            try:
                async for event in stream:
                    if isinstance(event, StreamComplete):
                        final = event.response
            finally:
                closer = getattr(stream, "aclose", None)
                if closer is not None:
                    await closer()
        if final is None:
            return LLMResponse(content="", tool_calls=[], stop_reason=None)
        return final

    # ── tools ────────────────────────────────────────────────────────

    async def _run_tool(self, tc: ToolCall) -> _ToolResult:
        # A later read, rejected edit or refused call cannot inherit an earlier edit's "done".
        self.complete, self.closing = False, ""
        name = tc.name or ""
        if self.escalation and name == prompts.HAND_TO_AGENT:
            args = tc.input if isinstance(tc.input, dict) else {}
            reason = " ".join(str(args.get("reason") or "").split())[:400]
            self.progress.log("handed to the agent: " + (reason or "no reason given"))
            raise _Escalate(reason or "This needs more than editing the artifact's files.")
        quota = QUOTA.get(name)
        if quota is None:
            return _ToolResult(f"ERROR: unknown tool {name}", ok=False)
        if tc.parse_error or tc.repaired:
            self.used[name] = self.used.get(name, 0) + (1 if name == prompts.APPLY_DIFF else 0)
            return _ToolResult(
                "ERROR: this call's arguments were cut off or are not valid JSON, so nothing was done. Make the "
                "call again, complete; keep it shorter if you ran out of room." + (
                    prompts.budget_note(self._edits_left()) if name == prompts.APPLY_DIFF else ""),
                ok=False, detail=["the call's arguments were cut off"],
            )
        if self.used.get(name, 0) >= quota:
            return _ToolResult(
                f"ERROR: {name} has been used {quota} times, which is the limit for one turn. Work with what you "
                "already have, and say what you were unable to do.",
                ok=False, detail=[f"out of {name} calls for this turn"],
            )
        self.used[name] = self.used.get(name, 0) + 1
        args = tc.input if isinstance(tc.input, dict) else {}
        try:
            if name == prompts.READ_FILES:
                self.progress.status("Reading the files...")
                return await asyncio.to_thread(self._read_tool, args)
            if name == prompts.SEARCH:
                self.progress.status("Searching the artifact...")
                return await asyncio.to_thread(self._search_tool, args)
            self.progress.status("Applying the edit...")
            return await asyncio.to_thread(self._apply_diff, args)
        except WorkspaceError as exc:
            return _ToolResult(f"ERROR: {exc}", ok=False)

    def _edits_left(self) -> int:
        return QUOTA[prompts.APPLY_DIFF] - self.used.get(prompts.APPLY_DIFF, 0)

    def _view(self, path: str) -> str | None:
        try:
            return self.ws.read(path)
        except WorkspaceError:
            return None

    def _already_shown(self, path: str) -> bool:
        shown = self.served.get(path)
        return shown is not None and self._view(path) == shown

    def _read_files(self, paths: Sequence[str]) -> str:
        """Files already shown and unchanged come back as a sentence, not another copy."""
        out: list[str] = []
        for path in paths:
            if self._already_shown(path):
                out.append(f"--- {path} ---\n{prompts.UNCHANGED}\n")
                continue
            try:
                text = self.ws.read(path)
            except WorkspaceError as exc:
                out.append(f"--- {path} ---\n(cannot be read: {exc})\n")
                continue
            if text is None:
                out.append(f"--- {path} ---\n(no such file)\n")
                continue
            self.served[path] = text
            out.append(f"--- {path} ---\n{text}\n" if text.endswith("\n") or not text else f"--- {path} ---\n{text}\n\n")
        return "".join(out)

    def _read_tool(self, args: dict) -> _ToolResult:
        asked = [str(p).strip() for p in (args.get("paths") or []) if str(p).strip()][: prompts.MAX_READ_PATHS]
        pattern = str(args.get("matching") or "").strip()
        note = ""
        if pattern:
            hits = self.ws.search(pattern)
            matched = list(dict.fromkeys(hit.path for hit in hits))
            if not matched:
                note = f"Nothing matched {json.dumps(pattern)}.\n\n"
            else:
                extra = [p for p in matched if p not in asked and not self._already_shown(p)]
                if len(extra) > max(0, prompts.MAX_READ_PATHS - len(asked)):
                    found = "\n".join(f"{h.path}:{h.line}:{h.text}" for h in hits)
                    note = (f"That pattern matched {len(matched)} files, too many to read at once. Here is where it "
                            f"matched; ask again for the ones you want.\n\n{found}\n\n")
                else:
                    asked += extra
                    if extra:
                        note = f"Also read, for matching {json.dumps(pattern)}: {', '.join(extra)}\n\n"
        if not asked:
            return _ToolResult(note + "Nothing to read: no paths were given and nothing matched.", ok=True)
        self.read_paths += asked
        return _ToolResult(note + self._read_files(asked), ok=True)

    def _search_tool(self, args: dict) -> _ToolResult:
        query = str(args.get("query") or "")
        hits = self.ws.search(query)
        lines = [f"{h.path}:{h.line}:{h.text}" for h in hits]
        out = "\n".join(lines) if lines else "No matches."
        if self.ws.search_note:
            out += f"\n({self.ws.search_note})"
        fresh = [p for p in dict.fromkeys(h.path for h in hits) if not self._already_shown(p)]
        if fresh and len(fresh) <= SEARCH_ATTACH_FILES:
            kept, total = [], 0
            for path in fresh:
                text = self._view(path)
                if text is None or len(text) > SEARCH_ATTACH_FILE_CHARS or total + len(text) > SEARCH_ATTACH_TOTAL_CHARS:
                    continue
                total += len(text)
                kept.append(path)
            if kept:
                n = len(kept)
                out += (f"\n\nThose matches are in {n} {'file' if n == 1 else 'files'}, so here "
                        f"{'it is' if n == 1 else 'they are'} in full. Do not call read_files for "
                        f"{'it' if n == 1 else 'them'}.\n\n" + self._read_files(kept))
                self.read_paths += kept
        return _ToolResult(out, ok=True)

    def _exists(self, path: str) -> bool:
        try:
            return self.ws.read(path) is not None
        except WorkspaceError:
            return True

    def _snapshot(self, paths: Sequence[str]) -> None:
        """Keep what PATHS hold before the edit first writes them (the presentation copy too)."""
        from anton.core.artifacts.internal_files import PRESENTATION_PROGRAM_FILENAME

        wanted = [p for p in [*paths, PRESENTATION_PROGRAM_FILENAME] if p not in self._snapshotted]
        if not wanted:
            return
        self._snapshots.append(self.ws.snapshot(wanted))
        self._snapshotted.update(wanted)

    def _restore(self) -> None:
        if self._restored or not self._snapshots:
            self._restored = True
            return
        self._restored = True
        for snapshot in reversed(self._snapshots):
            try:
                snapshot.restore()
            except Exception:  # noqa: BLE001 — keep restoring the rest
                _log.error("could not restore %s after a failed edit", snapshot.paths, exc_info=True)
        self.progress.log("put every file back as it was before this edit")

    def _write(self, files: dict[str, str]) -> None:
        self._snapshot(list(files))
        self.ws.write(files)

    def _landed(self, paths: Sequence[str]) -> None:
        for path in paths:
            if path not in self.applied:
                self.applied.append(path)
        self.unplaced = [p for p in self.unplaced if p not in paths]

    def _attempt(self, paths: Sequence[str]) -> None:
        for path in paths:
            if path and path not in self.attempted:
                self.attempted.append(path)

    def _stale(self, paths: Sequence[str]) -> str:
        stale = [p for p in dict.fromkeys(paths) if p and not self._already_shown(p) and self._view(p) is not None]
        if not stale:
            return ""
        return prompts.STALE_HEADER + self._read_files(stale)

    def _reject(self, patch: str, paths: list[str], *, reasons: list[dict], detail: str, placed: list[str]) -> None:
        self.last_failure = detail
        if self.record is not None:
            self.record.patch_rejected(attempt=self.used.get(prompts.APPLY_DIFF, 0), patch=patch, paths=paths,
                                       reasons=reasons, detail=detail, placed=placed)

    def _parse(self, patch: str):
        """parse_patch, after two repairs for header slips measured on real models.

        * ``@@static/index.html`` (a hunk separator glued to a file name the
          artifact has) is read as the file header ``@static/index.html``.
        * A patch with no file header at all (or only bare ``@@`` separators)
          on an html-app goes to its one page: there is nothing to guess.
        """
        known = {entry.path for entry in self.ws.map()}
        lines = patch.split("\n")
        for index, line in enumerate(lines):
            if line.startswith("@@") and line[2:].strip() in known:
                lines[index] = "@" + line[2:].strip()
        patch = "\n".join(lines)
        try:
            return parse_patch(patch)
        except PatchParseError as exc:
            headerless = not any(line.startswith("@") and not line.startswith("@@") for line in lines)
            if (self.ws.kind == "html-app" and self.ws.primary and headerless
                    and ("before any file header" in str(exc) or "no file headers" in str(exc))):
                return parse_patch(f"@{self.ws.primary}\n{patch}")
            raise

    def _apply_diff(self, args: dict) -> _ToolResult:
        budget = prompts.budget_note(self._edits_left())
        if not isinstance(args.get("task_complete"), bool):
            return _ToolResult(
                "ERROR: task_complete must be a boolean: true only when this edit completes the entire request, "
                "false while work remains. Nothing was changed." + budget, ok=False)
        patch = str(args.get("patch") or "")
        if not patch.strip():
            return _ToolResult("ERROR: no patch was given." + budget, ok=False, detail=["no patch was given"])
        try:
            edits = self._parse(patch)
        except PatchParseError as exc:
            self._reject(patch, [], reasons=[{"kind": "parse", "message": str(exc)}], detail=str(exc), placed=[])
            return _ToolResult(f"The patch could not be read; nothing was changed.{budget}\n\n{exc}", ok=False,
                               detail=[f"unreadable patch: {_clip(str(exc), 100)}"])
        paths = list(dict.fromkeys(e.path for e in edits))
        if not edits:
            return _ToolResult("ERROR: the patch names no file to change." + budget, ok=False)
        refused = []
        for edit in edits:
            reason = self.ws.refuse_reason(edit.path, create=not self._exists(edit.path))
            if reason:
                refused.append(reason)
        if refused:
            detail = "\n".join(dict.fromkeys(refused))
            self._attempt(paths)
            self._reject(patch, paths, reasons=[{"kind": "refused", "message": r} for r in refused], detail=detail,
                         placed=[])
            return _ToolResult(f"The patch was refused; nothing was changed.{budget}\n\n{detail}", ok=False,
                               detail=["refused: " + _clip(refused[0], 100)])
        unreadable: list[str] = []

        def read(path: str) -> str | None:
            try:
                return self.ws.read(path)
            except WorkspaceError as exc:
                unreadable.append(str(exc))
                return None

        result = apply_patch(edits, read=read)
        if unreadable:
            detail = "\n".join(unreadable)
            self._reject(patch, paths, reasons=[{"kind": "unreadable", "message": u} for u in unreadable],
                         detail=detail, placed=[])
            return _ToolResult(f"The patch could not be applied; nothing was changed.{budget}\n\n{detail}", ok=False)
        written = dict(result.written)
        if written:
            try:
                self._write(written)
            except WorkspaceError as exc:
                self._attempt(paths)
                self._reject(patch, paths, reasons=[{"kind": "write_refused", "message": str(exc)}],
                             detail=str(exc), placed=[])
                return _ToolResult(
                    f"The patch placed, but saving it was refused; nothing was changed.{budget}\n\n{exc}",
                    ok=False, detail=["save refused: " + _clip(str(exc), 100)])
            self._landed(list(written))
        if result.ok:
            note = str(args.get("response_comment_for_user") or "").strip()
            if note:
                self.closing = note
            self.complete = bool(args["task_complete"])
            names = ", ".join(written)
            them = "it" if len(written) == 1 else "them"
            return _ToolResult(
                f"Applied. {names} now {'contains' if len(written) == 1 else 'contain'} every edit in this patch; "
                f"there is no need to read {them} back.{budget}", ok=True,
                detail=[f"edited {names}"] + ([f"repairs: {', '.join(result.repairs)}"] if result.repairs else []))
        # Something did not place (or the patch changes nothing).
        failed = list(result.failed)
        self._attempt(failed or paths)
        for path in failed:
            if path not in self.unplaced:
                self.unplaced.append(path)
        feedback = result.feedback(tool=prompts.APPLY_DIFF)
        reasons = [{"path": d.path, "hunk": d.hunk_index, "kind": d.kind, "message": d.message}
                   for group in result.failed.values() for d in group]
        self._reject(patch, paths, reasons=reasons or [{"kind": "noop"}], detail=feedback, placed=list(written))
        stale = self._stale(failed)
        if written:
            return _ToolResult(
                f"Part of the patch applied. Written now, with every edit for them in place: {', '.join(written)}. "
                f"Do not resend or re-read those files. Not applied: {', '.join(failed)}. Resend only the edits for "
                f"those files, fixed as below, all in one apply_diff call.{budget}\n\n{feedback}{stale}",
                ok=False, detail=[f"applied {', '.join(written)}", f"not placed: {', '.join(failed)}"])
        return _ToolResult(
            f"The patch did not apply. Nothing was changed by this call; earlier accepted edits were not rolled "
            f"back.{budget}\n\n{feedback}{stale}",
            ok=False, detail=[("not placed: " + ", ".join(failed)) if failed else "the patch changes nothing"])

    # ── the check ────────────────────────────────────────────────────

    async def _check(self) -> CheckResult:
        self._stage("checking")
        self.progress.status("Checking the page...")
        started = time.monotonic()
        try:
            result = await self.checker.check(self.ws, list(self.applied))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a checker crash is "nobody could verify", never "passed"
            _log.warning("the artifact check crashed", exc_info=True)
            result = CheckResult("unverified", "none", note=f"the check could not run ({type(exc).__name__})")
        self.last_check = result
        if self.record is not None:
            self.record.check(channel=result.channel, status=result.status, ms=result.ms,
                              findings=[vars(f) for f in result.findings], note=result.note)
        if result.status == "passed":
            self.progress.log(f"check passed ({result.channel}) · {_seconds(started)}")
        elif result.status == "unverified":
            self.progress.log(f"not verified in a browser: {_clip(result.note, 160)}")
        else:
            self.progress.log(f"check failed ({result.channel}), back to it · {_seconds(started)}")
            for finding in result.findings[:3]:
                self.progress.log("  " + _clip(f"{finding.kind}: {finding.message}", 140))
        return result

    def _check_failed(self, result: CheckResult) -> None:
        # A static failure (a script that does not parse) costs milliseconds to
        # find and is bounded by the edit budget; only browser failures count
        # toward giving up.
        if result.channel != "static":
            self.check_failures += 1
        elif self.used.get(prompts.APPLY_DIFF, 0) >= QUOTA[prompts.APPLY_DIFF]:
            raise _GiveUp("the edits still do not pass the static check, and the edit budget is spent", result)
        if self.max_fix_rounds is not None:
            if self.fix_rounds >= self.max_fix_rounds:
                raise _GiveUp(f"the check still failed after {self.fix_rounds} fix "
                              f"round{'s' if self.fix_rounds != 1 else ''}", result)
        elif self.check_failures >= MAX_CHECK_FAILURES:
            raise _GiveUp(f"the check failed {self.check_failures} times", result)
        self.fix_rounds += 1
        self._stage("fixing", round=self.fix_rounds, findings=len(result.findings))
        image = (result.screenshot_b64, result.screenshot_media_type) if result.screenshot_b64 else None
        self._say(prompts.CHECK_FAILED + result.describe(), image=image)

    def _outcome(self, reply: str, check: CheckResult | None, *, rewrote: bool = False) -> Outcome:
        return Outcome("applied", reply=reply.strip(), files=list(self.applied), check=check, rewrote=rewrote)

    # ── the last resort ──────────────────────────────────────────────

    async def _rewrite_instead(self, task: str) -> Outcome:
        """No edit would place, so the files are asked for whole, one per call."""
        from anton.core.llm.effort import WRITE_BUDGET_BY_EFFORT, client_stream_budget

        named = self.unplaced or self.attempted
        targets = [p for p in dict.fromkeys(named or self.read_paths) if self._view(p) is not None
                   and not self.ws.refuse_reason(p, create=False)]
        if not targets:
            raise _GiveUp("no edit would apply and there was no file to rewrite")
        evidence = self.last_failure or "Your edits could not be placed."
        self._stage("writing")
        self.progress.status("Rewriting the file instead...")
        summary = ""
        budget = client_stream_budget(self.llm, self.role, by_effort=WRITE_BUDGET_BY_EFFORT)
        for path in targets:
            current = self._view(path) or ""
            started = time.monotonic()
            prompt = (f"TASK:\n{task}\n\nFILE TO REWRITE:\n{path}\n\nITS CURRENT CONTENTS:\n{current}\n\n"
                      f"WHY THE DIFF FAILED:\n{evidence}")
            response, used_budget = await self._create(
                system=prompts.REWRITE_INSTRUCTIONS, tools=None, max_tokens=budget,
                messages=[{"role": "user", "content": prompt}],
            )
            if self._cut_off(response, used_budget):
                raise _GiveUp(f"the whole-file rewrite of {path} was cut off before it finished")
            parsed = decode_json_object(response.content or "")
            content = parsed.get("content") if parsed else None
            if not isinstance(content, str) or not content:
                raise _GiveUp(f"the model returned no content for {path}")
            if not content.strip() and current.strip():
                raise _GiveUp(f"refusing to empty {path}: the model returned no content for it")
            try:
                await asyncio.to_thread(self._write, {path: content})
            except WorkspaceError as exc:
                raise _GiveUp(f"the rewrite of {path} was refused: {exc}") from exc
            self._landed([path])
            self.progress.log(f"wrote {path} whole · {_seconds(started)}")
            summary = summary or str(parsed.get("summary") or "")
        result = await self._check()
        if result.status == "failed":
            raise _GiveUp(f"rewrote {', '.join(targets)}, but the check failed", result)
        return self._outcome(summary or f"Rewrote {', '.join(targets)}.", result, rewrote=True)
