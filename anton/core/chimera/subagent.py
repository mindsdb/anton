"""The sidekick's bounded tool-call loop.

Generalised from generate_artifact's `_run_loop`
(`anton/core/tools/generate_artifact/engine.py`): one model call a round on
the sidekick role, the round's tool calls run one at a time, the results go
back with the rounds left, and a `finish` call ends it. What it adds:

  - the messages persist across delegations (`SidekickState`), so a follow-up
    delegation within the reuse window continues the same conversation and
    its prompt cache;
  - the tools are a private `ToolRegistry` built from the session's own,
    restricted to what the sidekick may use (`tools.sidekick_tool_defs`);
  - Jev watches it (`ChimeraPolicy`), and can hand the work back to the main
    agent;
  - it compacts its own messages under context pressure, and only then may
    move up the upgrade ladder.

Protocol is Anthropic tool-use / tool-result blocks, as in `_run_loop`.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from anton.core.llm.client import ModelRoute, model_route
from anton.core.llm.provider import damaged_tool_call_result
from anton.core.llm.tracing import get_trace_context, reset_trace_context, set_trace_context
from anton.core.tools.generate_artifact import sub_tools
from anton.core.tools.generate_artifact.spend import WIND_DOWN_ROUNDS, SpendGuard
from anton.core.tools.generate_artifact.streaming import _call_with_stream_retry
from anton.core.tools.progress import ToolProgress
from anton.core.tools.registry import ToolOutcome, ToolRegistry
from anton.utils.datasources import scrub_credentials

from .report import SidekickReport, str_list

if TYPE_CHECKING:
    from anton.core.session import ChatSession

    from .policy import ChimeraPolicy, Verdict
    from .state import SidekickState

logger = logging.getLogger(__name__)

# The same for every delegation, so the sidekick's prompt cache survives from
# one to the next: the brief rides in the first user message instead.
SIDEKICK_SYSTEM_PROMPT = """\
You are the sidekick: a careful, fast assistant the main agent hands bounded pieces of work to. \
The main agent talks to the user; you never do. Nobody reads your text replies as they \
happen: the main agent reads only the report your `finish` call carries.

How to work:
- Do exactly what the brief asks, nothing more. When something outside the brief looks \
worth doing, put it in `open_questions` instead of doing it.
- Check your work before you report it: run it, read it back, count it. Put what you \
checked in `evidence` (a number, a file path, a cell's output) so the main agent can \
verify it quickly.
- When the brief is ambiguous, needs a decision only the user or the main agent can \
make, or keeps failing for a reason you cannot fix, stop and call `finish` with status \
`needs_main` and say exactly what is needed.
- Never invent data, results or file contents. If you could not get something, say so.
- Each reply costs a round and rounds are limited; batch independent tool calls in one \
reply when you can.

End every task with one `finish` call, also when you could not complete it."""

FINISH_TOOL = {
    "name": "finish",
    "description": (
        "End this task and hand your report to the main agent. Call it once, when the "
        "done-when condition is met or when you cannot get further."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["done", "needs_main"],
                "description": (
                    "`done` when the done-when condition is met; `needs_main` when the "
                    "main agent has to decide, ask the user, or take over."
                ),
            },
            "summary": {
                "type": "string",
                "description": "What you did and found, in a few sentences.",
            },
            "outputs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "What you produced: file paths, artifact slugs, scratchpad variables, values.",
            },
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "How you checked the result, concretely enough for the main agent to re-check.",
            },
            "open_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Decisions or input the main agent still owes, and anything you left undone.",
            },
        },
        "required": ["status", "summary"],
    },
}

_NO_TOOL_CALL_MSG = (
    "Your reply called no tool, and nobody reads text replies. Continue with a tool "
    "call, or call `finish` with your report."
)
_WIND_DOWN_MSG = (
    "The turn's spend limit is reached. Stop working now and call `finish` with what "
    "you have; list what is left undone in `open_questions`."
)
_FINISH_NOT_DONE_MSG = (
    "Not accepted: an automatic check found no evidence yet that the done-when "
    "condition is met. Check the result against it and call `finish` again; if it "
    "cannot be met, call `finish` with status `needs_main` and say why."
)
_FINISH_TAKEN_BACK_MSG = "Recorded. The main agent is taking this task back."
_STUCK_MSG = (
    "[Automatic check: you look stuck, repeating steps without progress. Change "
    "approach now, or call `finish` with status `needs_main` and say what blocks you.]"
)
_OFF_SCOPE_MSG = (
    "[Automatic check: your last steps look outside the brief. Go back to what the "
    "brief asks; note anything else in `open_questions` instead of doing it.]"
)
_COMPACT_SYSTEM = (
    "Summarize this sidekick's work log so it can continue without it. Keep: what it "
    "has done and found, file paths, variable and scratchpad names, values and errors "
    "that matter, and what is left. Plain text, at most 300 words."
)
_COMPACTED_NOTE = (
    "Your earlier rounds on this task were compacted into the notes above. Continue "
    "from them."
)

# Context for a Jev check: enough recent rounds to judge progress, small enough
# for a 2-second call.
_CHECK_TRANSCRIPT = {"max_convo": 6, "max_tool": 12, "tool_cap": 300, "text_cap": 1200, "final_cap": 2000}
_COMPACT_TRANSCRIPT = {"max_convo": 20, "max_tool": 40, "tool_cap": 600, "text_cap": 2000, "final_cap": 4000}


@dataclass
class _Outcome:
    status: str
    summary: str
    outputs: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)


def _transcript(messages: list[dict], caps: dict) -> str:
    # The completion verifier's renderer: text-only, every cut marked.
    # Imported here because session.py imports this package.
    from anton.core.session import _render_verify_transcript

    return _render_verify_transcript(messages, **caps)


def _scrubbed(content):
    """``content`` with credentials removed, as the main loop does for every
    tool result; image blocks carry none."""
    if isinstance(content, list):
        return [
            {**b, "text": scrub_credentials(b.get("text", ""))}
            if isinstance(b, dict) and b.get("type") == "text" else b
            for b in content
        ]
    return scrub_credentials(content if isinstance(content, str) else str(content))


@contextlib.contextmanager
def _traced(fields: dict[str, str]) -> Iterator[None]:
    """Add ``fields`` to the Langfuse metadata of the calls made inside, when
    a turn is traced. Around one await only, like `model_route`."""
    ctx = get_trace_context()
    if ctx is None:
        yield
        return
    token = set_trace_context(replace(ctx, metadata={**(ctx.metadata or {}), **fields}))
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            reset_trace_context(token)


class SubAgent:
    """One delegation. Build it, `await run()`, then read ``messages`` (what
    the next delegation may continue) and ``state`` (updated signals)."""

    def __init__(
        self,
        *,
        session: "ChatSession",
        brief: str,
        done_when: str,
        context: str,
        registry: ToolRegistry,
        state: "SidekickState",
        policy: "ChimeraPolicy",
        max_rounds: int,
        delegation_id: str = "",
        on_progress: Callable[[ToolProgress], None] | None = None,
        system: str = SIDEKICK_SYSTEM_PROMPT,
    ) -> None:
        self.session = session
        self.brief = brief
        self.done_when = done_when
        self.context = context
        self.registry = registry
        self.state = state
        self.policy = policy
        self.max_rounds = max_rounds
        self.delegation_id = delegation_id
        self.system = system
        self._on_progress = on_progress
        llm = session._llm
        self._base_model: str = llm.sidekick_model
        self._effort: str | None = getattr(llm.sidekick_provider, "reasoning_effort", None)
        # An upgrade a previous delegation earned carries over; any other saved
        # model is ignored, so a changed sidekick setting takes effect.
        self.model: str = state.model if state.model in policy.config.upgrade_ladder else self._base_model
        self.messages: list[dict] = state.resume()
        self.rounds = 0
        # Tokens of the sidekick's own calls. The handler adds what its tools
        # and Jev spent when the turn's books can tell.
        self.tokens = 0
        self.compactions = 0
        self._error_streak = 0
        self._finish_refused = False
        self._stuck_nudged = False

    # -- helpers ------------------------------------------------------------

    def _progress(self, text: str, kind: str = "step") -> None:
        if self._on_progress is not None:
            self._on_progress(ToolProgress(text, kind=kind))

    def _kickoff(self, *, notes: str = "") -> str:
        """The task as the first user message: brief, done-when, the main
        agent's context, and earlier work when there are no messages to
        continue."""
        parts = [f"## Brief\n{self.brief}", f"## Done when\n{self.done_when}"]
        if self.context:
            parts.append(f"## Context from the main agent\n{self.context}")
        earlier = notes or (self.state.summary if not self.messages else "")
        if earlier:
            parts.append(f"## Your earlier work in this conversation\n{earlier}")
        parts.append(f"You have {self.max_rounds} rounds for this task.")
        return "\n\n".join(parts)

    def _append_user(self, blocks: list[dict]) -> None:
        """Add a user message, merged into the last one when that is a user
        message too (a continued delegation starts right after the previous
        one's `finish` result)."""
        if self.messages and self.messages[-1].get("role") == "user":
            last = self.messages[-1]
            content = last.get("content")
            existing = content if isinstance(content, list) else [{"type": "text", "text": str(content or "")}]
            self.messages[-1] = {**last, "content": existing + blocks}
            return
        self.messages.append({"role": "user", "content": blocks})

    def _route(self) -> ModelRoute:
        """The sidekick's model, for calls that run on other roles' providers."""
        return ModelRoute(model=self.model, effort=self._effort)

    def _round_note(self, round_idx: int) -> dict:
        rounds_left = self.max_rounds - round_idx - 1
        note = f"[{rounds_left} round(s) left in this task"
        if 0 < rounds_left <= WIND_DOWN_ROUNDS:
            note += " — wrap up NOW and call `finish`"
        return {"type": "text", "text": note + "]"}

    def _trace_fields(self, round_idx: int) -> dict[str, str]:
        return {
            "delegation_id": self.delegation_id,
            "sidekick_id": self.state.sidekick_id,
            "sidekick_round": str(round_idx + 1),
        }

    # -- the loop -------------------------------------------------------------

    async def run(self) -> SidekickReport:
        """Run the delegation to a report. Raises only on cancellation."""
        try:
            outcome = await self._loop()
        except Exception as exc:  # the main agent gets a report, never a crash
            logger.warning("chimera: the sidekick loop failed", exc_info=True)
            outcome = _Outcome(
                status="failed",
                summary=f"The sidekick stopped on an error: {type(exc).__name__}: {exc}"[:500],
            )
        # A round that died between its call and its results leaves tool
        # calls nothing answers, which the next delegation could not send.
        while self.messages and self.messages[-1].get("role") == "assistant":
            self.messages.pop()
        return SidekickReport(
            status=outcome.status,  # type: ignore[arg-type]
            summary=outcome.summary,
            outputs=outcome.outputs,
            evidence=outcome.evidence,
            open_questions=outcome.open_questions,
            rounds=self.rounds,
            tokens=self.tokens,
        )

    async def _loop(self) -> _Outcome:
        self._append_user([{"type": "text", "text": self._kickoff()}])
        spend = SpendGuard(self.session)
        closing_rounds_left = WIND_DOWN_ROUNDS
        wind_down_announced = False
        last_text = ""

        for round_idx in range(self.max_rounds):
            if spend.should_wind_down():
                if closing_rounds_left <= 0:
                    return self._out_of_budget("The turn's spend limit was reached.", last_text)
                closing_rounds_left -= 1
                if not wind_down_announced:
                    self._append_user([{"type": "text", "text": _WIND_DOWN_MSG}])
                    wind_down_announced = True
            self.rounds = round_idx + 1
            self._progress(f"Sidekick round {round_idx + 1}/{self.max_rounds}")
            response = await self._call(round_idx)
            last_text = (response.content or "").strip() or last_text
            self.messages.append({"role": "assistant", "content": sub_tools.assistant_blocks(response)})

            if not response.tool_calls:
                self._append_user([{"type": "text", "text": _NO_TOOL_CALL_MSG}, self._round_note(round_idx)])
                continue

            # Every tool_use needs its tool_result in the next message, finish
            # included, whatever this round decides; the slots keep the order.
            results: list[dict | None] = []
            finish = None
            for tc in response.tool_calls:
                damaged = damaged_tool_call_result(tc)
                if damaged is not None:
                    results.append(damaged)
                    self._error_streak += 1
                elif tc.name == "finish":
                    if finish is None:
                        finish = (len(results), tc)
                        results.append(None)
                    else:
                        results.append(sub_tools.tool_result(tc.id, "Error: call `finish` once."))
                else:
                    content, ok = await self._run_tool(tc)
                    results.append(sub_tools.tool_result(tc.id, content))
                    self._error_streak = self._error_streak + 1 if ok is False else 0

            trigger = self.policy.trigger(
                round_idx=round_idx, error_streak=self._error_streak, finishing=finish is not None,
            )
            verdict = await self._check(trigger, round_idx) if trigger else None

            if verdict is not None and verdict.take_back:
                self.state.signals += 1
                self._progress("Sidekick is handing the task back to the main agent")
                outcome = self._from_finish(finish[1]) if finish is not None else None
                self._answer(results, finish, _FINISH_TAKEN_BACK_MSG)
                self._append_user([r for r in results if r is not None])
                return _Outcome(
                    status="needs_main",
                    summary=(outcome.summary if outcome else last_text[:1500])
                    or "An automatic check judged that this task needs the main agent.",
                    outputs=outcome.outputs if outcome else [],
                    evidence=outcome.evidence if outcome else [],
                    open_questions=(outcome.open_questions if outcome else [])
                    or ["An automatic check judged that this needs the main agent's decision or the user's input."],
                )

            if finish is not None:
                if verdict is not None and verdict.not_done and not self._finish_refused:
                    self._finish_refused = True
                    self._answer(results, finish, _FINISH_NOT_DONE_MSG)
                else:
                    self._answer(results, finish, "ok")
                    self._append_user([r for r in results if r is not None])
                    return self._from_finish(finish[1])

            notes: list[dict] = []
            if verdict is not None and verdict.stuck:
                self.state.signals += 1
                if self._stuck_nudged:
                    self._append_user([r for r in results if r is not None])
                    return _Outcome(
                        status="needs_main",
                        summary=last_text[:1500] or "The sidekick got stuck and could not make progress.",
                        open_questions=["The sidekick got stuck twice on this task; it needs another approach."],
                    )
                self._stuck_nudged = True
                notes.append({"type": "text", "text": _STUCK_MSG})
            if verdict is not None and verdict.off_scope:
                notes.append({"type": "text", "text": _OFF_SCOPE_MSG})
            self._append_user([r for r in results if r is not None] + notes + [self._round_note(round_idx)])

            if response.usage.context_pressure > self.session._context_pressure_threshold:
                await self._compact()

        return self._out_of_budget(f"The round budget ({self.max_rounds}) ran out before `finish`.", last_text)

    async def _call(self, round_idx: int):
        """One sidekick round, retried once on a dropped stream. On a model of
        its own after an upgrade, the sidekick role is routed to it."""
        route = self._route() if self.model != self._base_model else None
        with model_route(route), _traced(self._trace_fields(round_idx)):
            response, _ = await _call_with_stream_retry(
                self.session._llm.sidekick_stream,
                system=self.system,
                messages=self.messages,
                tools=self.registry.dump() + [FINISH_TOOL],
                max_tokens=None,
                default_cap=None,
            )
        usage = response.usage
        self.tokens += (
            usage.input_tokens + usage.output_tokens + usage.cache_read_tokens + usage.cache_creation_tokens
        )
        return response

    async def _run_tool(self, tc) -> tuple[str | list[dict], bool | None]:
        """Dispatch one call through the private registry: its content and the
        handler's own verdict. A tool the sidekick may not use is refused
        without running; a raise becomes a failed result, as in the main loop."""
        names = [t.name for t in self.registry.get_tool_defs()]
        if tc.name not in names:
            return (
                f"Error: `{tc.name}` is not available to you. Your tools: "
                f"{', '.join(names + ['finish'])}. If the task needs `{tc.name}`, call "
                "`finish` with status `needs_main` and say so."
            ), False
        self._progress(f"Sidekick: {tc.name}")
        # generate_artifact builds on the sidekick's model, not the agent's.
        # The route is set around each step of the handler, never across a
        # yield, and the pipeline's task inherits it when it starts.
        route = self._route() if tc.name == "generate_artifact" else None
        result = None
        stream = self.registry.dispatch_tool_stream(self.session, tc.name, tc.input or {})
        try:
            while True:
                try:
                    with model_route(route):
                        item = await anext(stream)
                except StopAsyncIteration:
                    break
                if isinstance(item, ToolProgress):
                    self._progress(f"Sidekick · {item.text}" if item.kind == "step" else item.text, item.kind)
                else:
                    result = item
        except Exception as exc:
            return f"Tool '{tc.name}' failed: {exc}", False
        finally:
            await stream.aclose()
        if isinstance(result, ToolOutcome):
            return _scrubbed(result.content), result.ok
        return _scrubbed(result), None

    async def _check(self, trigger: str, round_idx: int) -> "Verdict":
        verdict = await self.policy.check(
            trigger=trigger,
            brief=self.brief,
            done_when=self.done_when,
            transcript=_transcript(self.messages, _CHECK_TRANSCRIPT),
            metadata=self._trace_fields(round_idx),
        )
        if trigger != "finish":
            self._error_streak = 0
        return verdict

    async def _compact(self) -> None:
        """Replace the messages with the task and a summary of the work so
        far. A failed summary leaves them as they are. Only here may the
        sidekick move up the upgrade ladder."""
        try:
            response = await self.session._llm.summarize(
                system=_COMPACT_SYSTEM,
                messages=[{"role": "user", "content": _transcript(self.messages, _COMPACT_TRANSCRIPT)}],
            )
        except Exception:
            logger.warning("chimera: compacting the sidekick's messages failed", exc_info=True)
            return
        summary = (response.content or "").strip()
        if not summary:
            return
        self.compactions += 1
        self.state.summary = summary
        self.messages = [{"role": "user", "content": [
            {"type": "text", "text": self._kickoff(notes=summary) + "\n\n" + _COMPACTED_NOTE},
        ]}]
        self._progress("Sidekick compacted its notes")
        upgrade = self.policy.next_model(self.model, self.state.signals)
        if upgrade is not None:
            logger.info("chimera: sidekick %s moves up from %s to %s", self.state.sidekick_id, self.model, upgrade)
            self.model = upgrade
            self.state.signals = 0
            self._progress(f"Sidekick moved up to {upgrade}")

    # -- outcomes -------------------------------------------------------------

    @staticmethod
    def _answer(results: list[dict | None], finish, content: str) -> None:
        if finish is not None:
            index, tc = finish
            results[index] = sub_tools.tool_result(tc.id, content)

    @staticmethod
    def _from_finish(tc) -> _Outcome:
        inp = tc.input or {}
        status = "needs_main" if inp.get("status") == "needs_main" else "done"
        return _Outcome(
            status=status,
            summary=str(inp.get("summary") or "").strip()[:4000] or "(no summary)",
            outputs=str_list(inp.get("outputs")),
            evidence=str_list(inp.get("evidence")),
            open_questions=str_list(inp.get("open_questions")),
        )

    @staticmethod
    def _out_of_budget(reason: str, last_text: str) -> _Outcome:
        summary = reason + (f" Its last note: {last_text[:1500]}" if last_text else "")
        return _Outcome(
            status="budget_exhausted",
            summary=summary,
            open_questions=["The sidekick did not finish; check what it left and do the rest."],
        )
