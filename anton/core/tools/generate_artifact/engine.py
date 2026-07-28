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

from . import sub_tools
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
    "file now ends, appending with `mode=\"a\"` and keeping each call's "
    "`content` well under ~6 KB."
)

_NO_CONTENT_MSG = (
    "Error: `content` was not delivered, so nothing was written. Re-emit this "
    "chunk with a non-empty `content`, well under ~6 KB, appending with "
    "`mode=\"a\"` if the file already has earlier chunks."
)


def _output_token_cap(session) -> int | None:
    """The client's effective output cap, or None when it cannot be read.

    `LLMClient` exposes no public accessor — only the private `_max_tokens`
    (`anton/core/llm/client.py:58`). In tests the session is an `AsyncMock` where
    the attribute exists and is truthy but is not a number, so the type check is
    mandatory: without it the detection would fire on every test.
    """
    cap = getattr(getattr(session, "_llm", None), "_max_tokens", None)
    return cap if isinstance(cap, int) and cap > 0 else None


def _response_is_truncated(response, cap: int | None) -> bool:
    """The reply hit the output cap, so its last tool call was not delivered.

    `stop_reason` cannot be relied on: the gateway reports `'stop'` on truncation
    too (stage-1a measurement, findings 2-3). `output_tokens` is sometimes None —
    then the signal is unknown and we do NOT flag: a false rejection is worse than
    a miss.
    """
    if not cap:
        return False
    used = getattr(getattr(response, "usage", None), "output_tokens", None)
    return isinstance(used, int) and used >= cap


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
) -> dict | str:
    """Drive the artifact-generation FSM to populate ``artifact_path``.

    Returns a result dict (with a step ``trace``) on success, or a single
    error string naming the node where the machine stopped.
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
    )
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
) -> str:
    """One-shot planning call → OpenAPI specification (JSON).

    The model is asked for an OpenAPI document as JSON. We validate the
    response by parsing it with ``json.loads``; if parsing succeeds the spec
    is considered valid and the (normalized) JSON string is returned.
    """
    system, user = build_api_spec_prompt(context, stateless=stateless)
    response = await session._llm.plan(
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if trace is not None:
        trace.llm_call(
            node=node_label,
            method="plan",
            system=system,
            messages=[{"role": "user", "content": user}],
            response=response,
        )
    spec = _strip_code_fence((response.content or "").strip())
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
        llm_call = session._llm.plan if round_idx == 0 else session._llm.code
        response = await llm_call(
            system=system,
            messages=messages,
            tools=tools,
        )

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
                res = sub_tools.read_file(artifact_path, inp.get("path", ""))
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

                content = await handle_scratchpad(session, inp)
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

        messages.append({"role": "user", "content": result_blocks})

        if finished_summary is not None:
            break
    else:
        return (
            f"generator exceeded round budget ({MAX_ROUNDS}) after writing "
            f"{len(files_written)} file(s): {files_written}."
        )

    if require_files and not files_written:
        return "generator finished without writing any files."

    return {
        "files_written": files_written,
        "rounds_used": round_idx + 1,
        "summary": finished_summary,
        "scratchpad_execs": scratchpad_execs,
    }
