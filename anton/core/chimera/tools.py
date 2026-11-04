"""The `delegate` tool: the main agent hands bounded work to the sidekick.

Anton runs tool calls one at a time, and a host may discard the session when
the turn ends, so a delegation lives entirely inside its own tool call: the
loop runs in a task this handler starts, its progress streams out as
`ToolProgress` (the way generate_artifact's does), and the task has finished
or been cancelled before the handler returns.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING

from anton.core.tools.progress import ToolProgress
from anton.core.tools.registry import ToolOutcome, ToolRegistry
from anton.core.tools.tool_defs import ToolDef

from .policy import ChimeraPolicy
from .report import SidekickReport
from .state import SidekickState
from .subagent import SubAgent

if TYPE_CHECKING:
    from anton.core.session import ChatSession

logger = logging.getLogger(__name__)

# anton's own tools the sidekick may use. `generate_artifact` runs on the
# sidekick's model (`SubAgent._run_tool`).
SIDEKICK_ALLOWLIST = frozenset({
    "scratchpad",
    "read_text_file",
    "read_image",
    "list_artifacts",
    "recall",
    "create_artifact",
    "update_artifact",
    "generate_artifact",
})
# Never the sidekick's, even when a host marks one sidekick-safe: each talks to
# the user, changes what persists beyond the task, or delegates again.
# `connect_new_datasource` and `publish_or_preview` are anton's own names for
# `connect_datasource` and `publish`.
SIDEKICK_NEVER = frozenset({
    "ask_user",
    "approve_held_write",
    "publish",
    "publish_or_preview",
    "memorize",
    "request_credentials",
    "connect_datasource",
    "connect_new_datasource",
    "label_connection",
    "create_skill_draft",
    "launch_backend",
    "delegate",
})

# How long a cancelled delegation may take to stop before the tool call returns
# anyway. A cell or a pipeline step ends on cancellation within moments.
_CANCEL_GRACE_S = 10.0

DELEGATE_PROMPT = """\
## Delegating to your sidekick
You have a sidekick: a faster, cheaper model you hand bounded work to with the `delegate` tool. \
It works in its own conversation with its own scratchpad access and reports back; it never \
talks to the user.

Delegate work that is bounded, mechanical and checkable: pulling or transforming data in the \
scratchpad, reading files and extracting from them, building or updating an artifact from a \
settled spec, running a fixed list of checks. Keep for yourself anything ambiguous, the plan \
and its decisions, anything the user sees or answers, and the final review.

Write the brief so it can be done without asking: the goal, the inputs (paths, scratchpad and \
variable names, data sources), the constraints, and a `done_when` the sidekick can check. Pass \
in `context` what it cannot find on its own.

Always review the report before you rely on it: it is the sidekick's claim, not a fact. \
Spot-check its evidence against `done_when`. On `needs_main` or `budget_exhausted`, decide \
yourself what happens next. A follow-up delegation within a few minutes continues the same \
sidekick conversation; pass `fresh: true` for unrelated work."""


def sidekick_tool_defs(session: "ChatSession") -> list[ToolDef]:
    """The session's tools the sidekick may use, in the session's order and
    in the session's own variants (a scratchpad tool carrying its package
    list, the artifact tools this deployment offers)."""
    return [
        tool for tool in session.tool_registry.get_tool_defs()
        if tool.name not in SIDEKICK_NEVER
        and (tool.name in SIDEKICK_ALLOWLIST or getattr(tool, "sidekick_safe", False) is True)
    ]


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _role_tokens(session: "ChatSession") -> dict[str, int] | None:
    """Tokens per role the turn has booked so far, or None outside a turn."""
    books = getattr(session, "_turn_cost", None)
    by_role = getattr(books, "by_role", None)
    if not isinstance(by_role, dict):
        return None
    return {role: cost.tokens for role, cost in by_role.items()}


def _state_summary(previous: str, brief: str, report: SidekickReport) -> str:
    """What a restarted sidekick reads about its earlier work: this
    delegation's outcome, after a clipped tail of the earlier summary."""
    lines = [f"Task: {brief[:500]}", f"Result ({report.status}): {report.summary[:1500]}"]
    if report.outputs:
        lines.append("Outputs: " + "; ".join(report.outputs)[:1000])
    if report.open_questions:
        lines.append("Left open: " + "; ".join(report.open_questions)[:1000])
    current = "\n".join(lines)
    if not previous:
        return current
    earlier = previous if len(previous) <= 1500 else "[...] " + previous[-1500:]
    return f"{earlier}\n\n{current}"


async def _run_then_close(agent: SubAgent, queue: "asyncio.Queue[ToolProgress | None]") -> SidekickReport:
    """Run the delegation, then close the progress channel exactly once, so
    the handler's drain ends on a crash or a cancellation too."""
    try:
        return await agent.run()
    finally:
        queue.put_nowait(None)


async def handle_delegate(session: "ChatSession", tc_input: dict):
    """Run one delegation and return the sidekick's report as JSON.

    An async generator: a delegation takes minutes, and the step lines it
    yields are what keep it from reading as a hang.
    """
    brief = _text(tc_input.get("brief"))
    done_when = _text(tc_input.get("done_when"))
    if not brief:
        yield ToolOutcome(content="Error: `brief` is required: what the sidekick should do.", ok=False,
                          reason="delegate_invalid_input")
        return
    if not done_when:
        yield ToolOutcome(content="Error: `done_when` is required: how the sidekick can tell it is done.",
                          ok=False, reason="delegate_invalid_input")
        return
    chimera = getattr(session, "_chimera", None)
    llm = session._llm
    if chimera is None or not chimera.enabled or getattr(llm, "has_sidekick", False) is not True:
        yield ToolOutcome(content="Error: delegation is not available in this session.", ok=False,
                          reason="delegate_unavailable")
        return

    available = sidekick_tool_defs(session)
    wanted = tc_input.get("tools")
    if wanted:
        names = {tool.name for tool in available}
        asked = [str(name) for name in wanted] if isinstance(wanted, list) else [str(wanted)]
        unknown = sorted(set(asked) - names)
        if unknown:
            yield ToolOutcome(
                content=(
                    f"Error: the sidekick cannot use {', '.join(unknown)}. Its tools here: "
                    f"{', '.join(sorted(names)) or 'none'}. Leave `tools` out to give it all of them."
                ),
                ok=False,
                reason="delegate_invalid_input",
            )
            return
        available = [tool for tool in available if tool.name in asked]
    registry = ToolRegistry()
    for tool in available:
        registry.register_tool(tool)

    state: SidekickState | None = getattr(session, "_sidekick_state", None)
    if state is None or tc_input.get("fresh") is True:
        state = SidekickState.new(llm.sidekick_model)
    policy = getattr(session, "_chimera_policy", None) or ChimeraPolicy(
        chimera, decide=getattr(session, "_jev_decide", None),
    )
    delegation_id = uuid.uuid4().hex[:12]
    checks_before = len(policy.checks)
    tokens_before = _role_tokens(session)

    queue: asyncio.Queue[ToolProgress | None] = asyncio.Queue()
    agent = SubAgent(
        session=session,
        brief=brief,
        done_when=done_when,
        context=_text(tc_input.get("context")),
        registry=registry,
        state=state,
        policy=policy,
        max_rounds=chimera.rounds_for(tc_input.get("max_rounds")),
        delegation_id=delegation_id,
        on_progress=queue.put_nowait,
    )
    task = asyncio.create_task(_run_then_close(agent, queue))
    try:
        while (item := await queue.get()) is not None:
            yield item
        report = await task
    finally:
        # A no-op once the task is done. Reached with it running when the
        # consumer closes this generator (a Stop, a cancelled turn): the
        # sidekick must not keep working with nobody to report to.
        if not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=_CANCEL_GRACE_S)

    tokens_after = _role_tokens(session)
    by_role: dict[str, int] = {}
    if tokens_before is not None and tokens_after is not None:
        by_role = {
            role: spent for role, tokens in tokens_after.items()
            if (spent := tokens - tokens_before.get(role, 0)) > 0
        }
        report.tokens = sum(by_role.values()) or report.tokens

    state.record(
        messages=agent.messages,
        summary=_state_summary(state.summary, brief, report),
        report_status=report.status,
        model=agent.model,
    )
    session._sidekick_state = state
    checks = policy.checks[checks_before:]
    _record_delegation(session, report, checks)
    logger.info(
        "chimera delegation=%s sidekick=%s model=%s status=%s rounds=%d tokens=%d compactions=%d "
        "by_role=%s jev=%s",
        delegation_id, state.sidekick_id, agent.model, report.status, report.rounds, report.tokens,
        agent.compactions,
        ",".join(f"{role}={tokens}" for role, tokens in sorted(by_role.items())) or "-",
        ";".join(
            f"{c['trigger']}:" + (c["error"] or ",".join(
                f"{name}={c[name]:.2f}" for name in ("take_back", "stuck", "off_scope", "done")
                if c.get(name) is not None
            ))
            for c in checks
        ) or "-",
    )
    yield ToolOutcome(
        content=report.to_tool_content(),
        ok=report.status != "failed",
        reason="sidekick_failed" if report.status == "failed" else "",
    )


def _record_delegation(session: "ChatSession", report: SidekickReport, checks: list[dict]) -> None:
    """Count the delegation on the turn's books, which the turn event carries."""
    try:
        books = getattr(session, "_turn_cost", None)
        if books is None or not hasattr(books, "sidekick_delegations"):
            return
        books.sidekick_delegations += 1
        books.sidekick_rounds += report.rounds
        books.sidekick_last_status = report.status
        books.sidekick_jev_checks += len(checks)
        books.sidekick_handbacks += int(report.status in ("needs_main", "budget_exhausted"))
    except Exception:  # pragma: no cover - bookkeeping must not break the tool
        logger.debug("chimera: delegation bookkeeping failed", exc_info=True)


DELEGATE_TOOL = ToolDef(
    name="delegate",
    description=(
        "Hand a bounded, checkable piece of work to your sidekick, a faster model that works "
        "in its own conversation with scratchpad, file and artifact tools, and get back its "
        "report: status (done | needs_main | budget_exhausted | failed), summary, outputs, "
        "evidence and open questions. The sidekick cannot ask the user anything. Review the "
        "report before relying on it."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "brief": {
                "type": "string",
                "description": (
                    "The task, complete enough to do without asking: the goal, the inputs "
                    "(paths, scratchpad and variable names, data sources) and the constraints."
                ),
            },
            "done_when": {
                "type": "string",
                "description": "A condition the sidekick can check to know it is done.",
            },
            "context": {
                "type": "string",
                "description": "What the sidekick needs from this conversation that it cannot find itself.",
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Limit the sidekick to these of its tools. Leave out to give it all of "
                    "them (scratchpad, read_text_file, read_image, list_artifacts, recall, "
                    "create_artifact, update_artifact, generate_artifact, where available here)."
                ),
            },
            "max_rounds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 40,
                "description": "Rounds the sidekick may take; leave out for the default.",
            },
            "fresh": {
                "type": "boolean",
                "description": "Start a new sidekick conversation instead of continuing the last one.",
            },
        },
        "required": ["brief", "done_when"],
    },
    handler=handle_delegate,
    prompt=DELEGATE_PROMPT,
)
