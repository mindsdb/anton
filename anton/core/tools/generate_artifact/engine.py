"""Async tool-call loop(s) that drive the inner generation LLM.

For html-app: single loop.
For fullstack-stateless-app and fullstack-stateful-app:
  1. One-shot planning call → OpenAPI specification (JSON, kept in memory).
  2. asyncio.gather → backend loop + frontend loop in parallel.

The sub-generator reaches real data itself through the `scratchpad` sub-tool,
guided by the free-form `## Data` section of the brief (which names the
scratchpads/cells the main agent already used). The engine no longer fabricates
test data or pre-pickles variables.

The loop protocol is Anthropic tool-use / tool-result blocks, which both
providers Anton ships (AnthropicProvider, OpenAIProvider) accept on input.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from anton.core.artifacts.internal_files import PRD_FILENAME

from . import sub_tools
from .state import SPEC_MAX_TOKENS, SPEC_MAX_TOKENS_RETRY
from .prompts import (
    build_api_spec_prompt,
    build_backend_kickoff,
    build_backend_system_prompt,
    build_frontend_kickoff,
    build_frontend_system_prompt,
    build_subagent_system_prompt,
    build_user_kickoff,
)

if TYPE_CHECKING:
    from anton.chat_session import ChatSession
    from anton.core.llm.provider import LLMResponse


# Higher than the old 12 because the sub-generator also spends rounds on
# scratchpad calls (pulling/rebuilding data) on top of writing files, and higher
# than 16 because a file is now built in chunks (`mode="a"`) rather than in one
# call — head, sections, scripts and closing tags each cost a round unless the
# model batches them. One shared cap: it also gates `fetch_data_sample`, and a
# runaway there is bounded by DATA_LOOP_MAX anyway.
MAX_ROUNDS = 20


# Two distinct rejections, because the causes differ and the model must react
# differently. Both wordings point at the only working way out — chunked writing
# (see _WRITE_DISCIPLINE in prompts.py and the design spec, 3.1).
#
# CRITICAL in _TRUNCATED_MSG: state that the EARLIER calls in the same reply did
# land. Truncation is streaming — only the last block is incomplete, the rest
# arrived in full. Without saying so, the model re-sends the chunks it already
# wrote and mode="a" duplicates them in the file.
_TRUNCATED_MSG = (
    "Error: THIS tool call was cut off by the output limit and wrote nothing. "
    "The earlier tool calls in this same reply DID take effect — do NOT re-send "
    "them, or `mode=\"a\"` will duplicate their content. Continue from where the "
    "file now ends, appending with `mode=\"a\"`. Your next chunk must be at most "
    "4,000 characters of `content`; split the remaining work into as many "
    "chunks as it takes."
)

_NO_CONTENT_MSG = (
    "Error: `content` was not delivered, so nothing was written. Re-emit this "
    "chunk with a non-empty `content` of at most 4,000 characters, appending "
    "with `mode=\"a\"` if the file already has earlier chunks."
)


def _unwrap_outcome(result):
    """Flatten a handler result down to `tool_result`-ready content.

    Since ENG-696 some handlers return a `ToolOutcome` (content + the
    handler's own ok/reason verdict) instead of a bare string —
    `handle_scratchpad`'s exec path is the one this loop hits. The verdict
    drives the outer agent's error streak, which this sub-generator does
    not participate in, so only `content` is meaningful here; passing the
    dataclass straight into a tool_result block would ship its repr to the
    model.
    """
    from anton.core.tools.registry import ToolOutcome

    return result.content if isinstance(result, ToolOutcome) else result


async def _drain_stream(events) -> "LLMResponse":
    """Consume a `plan_stream()`/`code_stream()` iterator, discarding the
    token-level deltas, and return the final assembled response.

    Used in place of the one-shot `plan()`/`code()` calls. This pipeline runs
    headless — its own progress surface is the step-level `ToolProgress`
    protocol, not per-token UX — so nothing needs the intermediate
    `StreamTextDelta`/`StreamToolUse*` events, only the terminal
    `StreamComplete`. The reason to stream at all is transport, not UX: a
    non-streaming call sends no bytes over the wire until the whole response
    is ready, and `api.mindshub.ai` sits behind Cloudflare, which kills a
    connection that has been silent for ~100s with a 524 — a real failure on
    long spec/code generations. Streaming keeps bytes flowing continuously so
    the proxy never observes silence, regardless of how long the full
    generation takes.
    """
    from anton.core.llm.provider import StreamComplete

    result = None
    async for event in events:
        if isinstance(event, StreamComplete):
            result = event.response
    if result is None:
        raise RuntimeError("LLM stream ended without a StreamComplete event")
    return result


def _output_token_cap(session) -> int | None:
    """The client's effective output cap, or None when it cannot be read.

    Reads the public `max_tokens` property (`anton/core/llm/client.py:151`,
    added by ENG-1042 for exactly this comparison). In tests the session is an
    `AsyncMock` where the attribute exists and is truthy but is not a number,
    so the type check is mandatory: without it the detection would fire on
    every test.
    """
    cap = getattr(getattr(session, "_llm", None), "max_tokens", None)
    return cap if isinstance(cap, int) and cap > 0 else None


def _response_is_truncated(response, cap: int | None) -> bool:
    """The reply hit the output cap, so its last tool call was not delivered.

    Same semantics as the shared `looks_truncated` (`llm/structured.py`), which
    honours both `stop_reason` and the token count — the gateway reports
    `stop_reason` correctly since 2026-08-03. Kept local because mock sessions
    make type strictness mandatory here: `usage.output_tokens` on an AsyncMock
    is a truthy Mock, and comparing it against the cap would flag every test
    round as truncated.
    """
    if getattr(response, "stop_reason", None) in ("length", "max_tokens"):
        return True
    if not cap:
        return False
    used = getattr(getattr(response, "usage", None), "output_tokens", None)
    return isinstance(used, int) and used >= cap


_SPEC_COMPACT_NUDGE = (
    "\n\nIMPORTANT: your previous answer was cut off by the output limit "
    "before it reached the end, so it was discarded. Write the whole "
    "specification again, and make it fit: no preamble, no restating the "
    "brief or the PRD back at me, no worked examples, short lines. A complete "
    "structure matters far more than depth in any one section."
)


async def _plan_whole_document(
    session: "ChatSession",
    *,
    system: str,
    user: str,
    node_label: str,
    trace=None,
    on_retry=None,
) -> tuple[str, str | None]:
    """One planning call for a whole specification, retried once with more room.

    Returns ``(body, error)``; exactly one of the two is set.

    A spec is produced by a single call, so — unlike the file-writing loop —
    there is no chunk boundary to append at and a cut answer cannot be
    continued. It has to be re-asked, and the re-ask must CHANGE the call or it
    dies identically (measured for the main loop's own recovery,
    ``session._recover_truncated_stream``): the budget goes up AND the model is
    told to write compactly.

    A truncated document is never returned. Both generators consume the spec as
    their requirements, and `_spec_context` hands it to them verbatim, so half a
    spec means half a system built with nothing anywhere reporting that
    something was lost — which is the actual damage ENG-1116 describes, not the
    missing length.

    Truncation is detected with the shared `looks_truncated`, which also honours
    ``stop_reason`` — the gateway reports it correctly since 2026-08-03, and a
    token count alone cannot see a cut that stopped just under the cap.
    """
    from anton.core.llm.structured import looks_truncated

    response = None
    extra = ""
    for attempt, budget in enumerate((SPEC_MAX_TOKENS, SPEC_MAX_TOKENS_RETRY)):
        if attempt > 0 and on_retry is not None:
            on_retry()
        messages = [{"role": "user", "content": user + extra}]
        response = await _drain_stream(session._llm.plan_stream(
            system=system, messages=messages, max_tokens=budget,
        ))
        if trace is not None:
            trace.llm_call(
                node=node_label, method="plan", system=system,
                messages=messages, response=response, attempt=attempt,
            )
        if not looks_truncated(response, budget):
            return (response.content or "").strip(), None
        extra = _SPEC_COMPACT_NUDGE

    used = getattr(getattr(response, "usage", None), "output_tokens", None)
    return "", (
        f"{node_label}: the specification hit the output limit "
        f"({used} tokens against a {SPEC_MAX_TOKENS_RETRY} budget) and was "
        "still incomplete after a retry asking for a compact version."
    )


def _load_prd(state) -> None:
    """Load the confirmed PRD from the artifact folder, when there is one.

    Read here rather than accepted as a tool parameter: the handler already
    resolves the artifact folder from `slug`, so the file needs no addressing
    and the LLM-facing schema stays at `slug` + `context`. The calling agent
    cannot forget to pass it, mis-transcribe it, or paraphrase it — the
    document the user accepted is the one that arrives.

    Every failure mode degrades to "no PRD" instead of stopping the run:
    an agent may legitimately skip the PRD step, artifacts created before
    ENG-969 have no `prd.md`, and a file that cannot be read must not cost a
    generation that `context` alone can still complete. Which of the two
    modes ran is recorded, so a wrong-looking artifact can be traced back to
    the requirements it was actually built from.
    """
    path = state.artifact_path / PRD_FILENAME
    try:
        if not path.is_file():
            state.record("read_prd", "skipped", f"no {PRD_FILENAME} in the artifact folder")
            return
        body = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        state.record("read_prd", "error", f"{PRD_FILENAME} could not be read: {exc}")
        return
    if not body:
        state.record("read_prd", "skipped", f"{PRD_FILENAME} is empty")
        return
    state.prd = body
    state.record("read_prd", "done", f"{len(body)} chars from {PRD_FILENAME}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def generate(
    *,
    session: "ChatSession",
    artifact_type: str,
    artifact_path: Path,
    context: str,
    slug: str,
    primary: str | None = None,
    progress: "asyncio.Queue[str | None] | None" = None,
) -> dict | str:
    """Drive the artifact-generation FSM to populate ``artifact_path``.

    Returns a result dict (with a step ``trace``) on success, or a single
    error string naming the node where the machine stopped.

    ``progress``, when given, receives one user-facing line per step start
    (see ``GenState.step_started``). Optional so the non-streaming callers —
    ``bench_generate.py``, tests — need no channel to drain.
    """
    from .orchestrator import run
    from .state import GenState
    from .debug_trace import make_trace

    if artifact_type not in (
        "html-app", "fullstack-stateless-app", "fullstack-stateful-app"
    ):
        return f"Error: unsupported artifact type: {artifact_type!r}"

    trace = make_trace()
    trace.run_start(
        slug=slug,
        artifact_type=artifact_type,
        artifact_path=artifact_path,
        brief=context,
        is_fullstack=artifact_type != "html-app",
    )

    state = GenState(
        session=session,
        artifact_type=artifact_type,
        artifact_path=artifact_path,
        slug=slug,
        brief=context,
        is_fullstack=artifact_type != "html-app",
        primary=primary,
        trace_log=trace,
        progress=progress,
    )
    _load_prd(state)
    result = await run(state)
    if isinstance(result, str):
        trace.run_result(ok=False, error=result)
    else:
        trace.run_result(ok=True, result=result)
    return result


# ---------------------------------------------------------------------------
# Pre-generation steps
# ---------------------------------------------------------------------------


async def _generate_api_spec(
    session: "ChatSession",
    context: str,
    *,
    stateless: bool = False,
    trace=None,
    node_label: str = "make_api_spec",
    on_retry=None,
) -> str:
    """One-shot planning call → OpenAPI specification (JSON).

    The model is asked for an OpenAPI document as JSON. We validate the
    response by parsing it with ``json.loads``; if parsing succeeds the spec
    is considered valid and the (normalized) JSON string is returned.
    """
    system, user = build_api_spec_prompt(context, stateless=stateless)
    body, error = await _plan_whole_document(
        session, system=system, user=user, node_label=node_label, trace=trace,
        on_retry=on_retry,
    )
    if error is not None:
        # Already worded in terms of the output limit. Without this branch the
        # cut JSON would fall through to `json.loads` below and be reported as
        # "not valid JSON", pointing at the wrong cause entirely.
        return f"Error: {error}"
    spec = _strip_code_fence(body)
    if not spec:
        return "Error: API spec generation returned empty response."
    try:
        parsed = json.loads(spec)
    except json.JSONDecodeError as exc:
        return f"Error: API spec is not valid JSON: {exc}"
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def _strip_code_fence(text: str) -> str:
    """Strip a leading/trailing markdown code fence if present.

    Models often wrap JSON in ```json ... ``` despite being asked for raw JSON.
    """
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    lines = lines[1:]  # drop opening ```json / ``` line
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Generic bounded tool-call loop
# ---------------------------------------------------------------------------


async def _run_loop(
    *,
    session: "ChatSession",
    system: str,
    kickoff: str,
    artifact_path: Path,
    node_label: str,
    attempt: int | None = None,
    trace=None,
    step_injections: list[tuple[str, str]] | None = None,
    require_files: bool = True,
) -> dict | str:
    """Run one bounded sub-agent tool-call loop.

    ``step_injections`` is an optional list of ``(trigger_filename, message)``
    pairs. When a ``write_file`` call successfully writes ``trigger_filename``,
    ``message`` is appended to the tool-result content so the model receives
    the next-step instruction in the same turn. Each trigger fires at most once.

    Returns a result dict ``{files_written, rounds_used, summary,
    scratchpad_execs}`` on success, or a plain error string on failure.
    ``scratchpad_execs`` records every scratchpad ``exec`` the loop made —
    ``{name, code, output}`` per call — so callers can hand the exact
    data-access code that ran to later FSM steps.
    """
    tools = sub_tools.tool_schemas()
    cap = _output_token_cap(session)
    messages: list[dict] = [{"role": "user", "content": kickoff}]

    files_written: list[str] = []
    scratchpad_execs: list[dict] = []
    finished_summary: str | None = None
    injected: set[str] = set()

    for round_idx in range(MAX_ROUNDS):
        # First round: use the planning model for highest-quality initial generation.
        # Subsequent rounds (retries, read_file refinements) use the coding model.
        llm_call = session._llm.plan_stream if round_idx == 0 else session._llm.code_stream
        response = await _drain_stream(llm_call(
            system=system,
            messages=messages,
            tools=tools,
        ))

        if trace is not None:
            trace.llm_call(
                node=node_label,
                method="plan" if round_idx == 0 else "code",
                system=system,
                messages=messages,
                response=response,
                attempt=attempt,
                round=round_idx,
            )

        # Truncation is streaming: only the LAST block of the reply is
        # incomplete, the earlier ones arrived in full. Rejecting them all would
        # throw away finished work every round, and if the model consistently
        # spends its whole output budget (measured: output_tokens at the cap in
        # roughly 20 of 32 rounds) nothing would ever be written across all
        # rounds. That would be the same failure this work fixes, only with a
        # clearer error text.
        truncated_tc_id = (
            response.tool_calls[-1].id
            if _response_is_truncated(response, cap) and response.tool_calls
            else None
        )

        if not response.tool_calls:
            tail = (response.content or "").strip()
            return (
                f"generator stopped without writing files "
                f"(round {round_idx + 1}/{MAX_ROUNDS}). "
                f"Last output: {tail[:300]!r}"
            )

        assistant_blocks: list[dict] = []
        if response.content:
            assistant_blocks.append({"type": "text", "text": response.content})
        for tc in response.tool_calls:
            assistant_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                }
            )
        messages.append({"role": "assistant", "content": assistant_blocks})

        result_blocks: list[dict] = []
        for tc in response.tool_calls:
            if tc.parse_error:
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": (
                            "Error: malformed tool input — re-emit with valid "
                            f"JSON. ({tc.parse_error})"
                        ),
                    }
                )
                continue

            name = tc.name
            inp = tc.input or {}

            if name == "finish":
                summary = str(inp.get("summary") or "").strip()
                finished_summary = summary or "(no summary)"
                result_blocks.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": "ok"}
                )
            elif name == "write_file":
                # `inp.get("content")` without a "" default: a missing key must
                # be distinguishable from a deliberately empty string, or the
                # error text is guessing. Both forms are rejected either way, but
                # with different messages — their causes differ.
                content = inp.get("content")
                if tc.id == truncated_tc_id:
                    result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": _TRUNCATED_MSG,
                        }
                    )
                    continue
                if not content:
                    result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": _NO_CONTENT_MSG,
                        }
                    )
                    continue
                res = sub_tools.write_file(
                    artifact_path,
                    inp.get("path", ""),
                    content,
                    mode=inp.get("mode", "w"),
                )
                msg = res["message"]
                if res.get("ok"):
                    written = res["written"]
                    if written not in files_written:
                        files_written.append(written)
                    if trace is not None:
                        trace.file_written(node=node_label, path=written)
                    for trigger, inject_msg in (step_injections or []):
                        if written == trigger and inject_msg not in injected:
                            injected.add(inject_msg)
                            msg = f"{msg}\n\n{inject_msg}"
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": msg,
                    }
                )
            elif name == "read_file":
                res = sub_tools.read_file(
                    artifact_path, inp.get("path", ""),
                    full=bool(inp.get("full", False)),
                )
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": res["message"],
                    }
                )
            elif name == "scratchpad":
                # Full scratchpad access: the sub-generator pulls or rebuilds
                # the data described in the brief's `## Data` section. Lazy
                # import avoids a tool_handlers <-> generate_artifact cycle.
                from anton.core.tools.tool_handlers import handle_scratchpad

                content = _unwrap_outcome(await handle_scratchpad(session, inp))
                if inp.get("action") == "exec":
                    scratchpad_execs.append(
                        {
                            "name": str(inp.get("name") or ""),
                            "code": str(inp.get("code") or ""),
                            "output": content if isinstance(content, str) else str(content),
                        }
                    )
                if trace is not None:
                    trace.scratchpad(node=node_label, input=inp, output=content)
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": content,
                    }
                )
            else:
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": (
                            f"Error: unknown sub-tool `{name}`. "
                            "Use write_file, read_file, or finish."
                        ),
                    }
                )

        # Round accounting rides on the same user message as the tool results.
        # The model has no other way to see the budget, and without it the
        # measured failure mode is spending the last rounds on self-checks and
        # dying at the cap with a finished file (live run 2026-08-27). Appended
        # as a trailing text block: both providers accept text after
        # tool_result blocks, and appending never invalidates the prefix cache.
        rounds_left = MAX_ROUNDS - round_idx - 1
        note = f"[{rounds_left} round(s) left in this task"
        if 0 < rounds_left <= 5:
            note += (
                " — wrap up NOW: close any open tags and call `finish`. "
                "Do not spend the remaining rounds on checks"
            )
        note += "]"
        messages.append(
            {"role": "user", "content": result_blocks + [{"type": "text", "text": note}]}
        )

        if finished_summary is not None:
            break
    else:
        # Budget exhausted without `finish`. A missing `finish` call is not
        # evidence the files are bad — when the loop DID write files, hand
        # them to the caller and let the verifier judge them (live run
        # 2026-08-27: a complete 48 KB page was deleted and regenerated
        # because the model burned its last rounds self-checking). The
        # `finished` flag tells the caller how the loop ended.
        if not files_written:
            return (
                f"generator exceeded round budget ({MAX_ROUNDS}) without "
                "writing any files."
            )
        return {
            "files_written": files_written,
            "rounds_used": MAX_ROUNDS,
            "summary": f"(round budget {MAX_ROUNDS} exhausted before finish was called)",
            "scratchpad_execs": scratchpad_execs,
            "finished": False,
        }

    if require_files and not files_written:
        return "generator finished without writing any files."

    return {
        "files_written": files_written,
        "rounds_used": round_idx + 1,
        "summary": finished_summary,
        "scratchpad_execs": scratchpad_execs,
        "finished": True,
    }
