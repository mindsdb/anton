"""Async tool-call loop(s) that drive the inner generation LLM.

For html-app: single loop.
For fullstack-stateless-app and fullstack-stateful-app:
  1. One-shot planning call → OpenAPI specification (JSON, kept in memory).
  2. asyncio.gather → backend loop + frontend loop in parallel.

The caller is responsible for providing real data context: a `### Sample`
subsection inside `## Data` in the brief and/or `data_refs` pointing at
scratchpad variables. The engine no longer fabricates test data.

The loop protocol is Anthropic tool-use / tool-result blocks, which both
providers Anton ships (AnthropicProvider, OpenAIProvider) accept on input.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from . import sub_tools
from .data_resolver import resolve_refs
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


MAX_ROUNDS = 12


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def generate(
    *,
    session: "ChatSession",
    artifact_type: str,
    artifact_path: Path,
    context: str,
    data_refs: list[dict],
) -> dict | str:
    """Drive the inner LLM(s) to populate ``artifact_path``.

    Returns either a result dict (on success) or a single error string.
    """
    sidecar_dir = artifact_path / "_gen_data"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    resolved = await resolve_refs(session, data_refs, sidecar_dir)
    if isinstance(resolved, str):
        return resolved

    # --- html-app: single generation loop, no changes --------------------
    if artifact_type == "html-app":
        return await _run_loop(
            session=session,
            system=build_subagent_system_prompt("html-app", artifact_path),
            kickoff=build_user_kickoff(context, resolved),
            artifact_path=artifact_path,
        )

    # --- fullstack types: spec → parallel backend + frontend --------------
    if artifact_type not in ("fullstack-stateless-app", "fullstack-stateful-app"):
        return f"Error: unsupported artifact type: {artifact_type!r}"

    api_spec_or_err = await _generate_api_spec(session, context, resolved)
    if api_spec_or_err.startswith("Error:"):
        return api_spec_or_err
    api_spec = api_spec_or_err

    stateless = artifact_type == "fullstack-stateless-app"

    backend_result, frontend_result = await asyncio.gather(
        _run_loop(
            session=session,
            system=build_backend_system_prompt(artifact_path, stateless=stateless),
            kickoff=build_backend_kickoff(context, resolved, api_spec),
            artifact_path=artifact_path,
            # Two-step backend generation: write backend.py first so that
            # requirements.txt can be based on its actual imports.
            step_injections=[
                (
                    "backend.py",
                    "backend.py written. Now write requirements.txt listing "
                    "EVERY package imported in backend.py (one per line, no "
                    "extras). Then call finish.",
                ),
            ],
        ),
        _run_loop(
            session=session,
            system=build_frontend_system_prompt(artifact_path),
            kickoff=build_frontend_kickoff(context, resolved, api_spec),
            artifact_path=artifact_path,
        ),
    )

    if isinstance(backend_result, str):
        return f"Backend generation failed: {backend_result}"
    if isinstance(frontend_result, str):
        return f"Frontend generation failed: {frontend_result}"

    return {
        "files_written": (
            backend_result["files_written"] + frontend_result["files_written"]
        ),
        "rounds_used": max(
            backend_result["rounds_used"], frontend_result["rounds_used"]
        ),
        "summary": (
            f"backend: {backend_result['summary']} | "
            f"frontend: {frontend_result['summary']}"
        ),
    }


# ---------------------------------------------------------------------------
# Pre-generation steps
# ---------------------------------------------------------------------------


async def _generate_api_spec(
    session: "ChatSession",
    context: str,
    data_summaries: list[dict],
) -> str:
    """One-shot planning call → OpenAPI specification (JSON).

    The model is asked for an OpenAPI document as JSON. We validate the
    response by parsing it with ``json.loads``; if parsing succeeds the spec
    is considered valid and the (normalized) JSON string is returned.
    """
    system, user = build_api_spec_prompt(context, data_summaries)
    response = await session._llm.plan(
        system=system,
        messages=[{"role": "user", "content": user}],
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
    step_injections: list[tuple[str, str]] | None = None,
) -> dict | str:
    """Run one bounded sub-agent tool-call loop.

    ``step_injections`` is an optional list of ``(trigger_filename, message)``
    pairs. When a ``write_file`` call successfully writes ``trigger_filename``,
    ``message`` is appended to the tool-result content so the model receives
    the next-step instruction in the same turn. Each trigger fires at most once.

    Returns a result dict ``{files_written, rounds_used, summary}`` on
    success, or a plain error string on failure.
    """
    tools = sub_tools.tool_schemas()
    messages: list[dict] = [{"role": "user", "content": kickoff}]

    files_written: list[str] = []
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

    if not files_written:
        return "generator finished without writing any files."

    return {
        "files_written": files_written,
        "rounds_used": round_idx + 1,
        "summary": finished_summary,
    }
