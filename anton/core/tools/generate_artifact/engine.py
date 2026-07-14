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


# Higher than the old 12 because the sub-generator now also spends rounds on
# scratchpad calls (pulling/rebuilding data) on top of writing files.
MAX_ROUNDS = 16


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
                res = sub_tools.write_file(
                    artifact_path,
                    inp.get("path", ""),
                    inp.get("content", ""),
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
