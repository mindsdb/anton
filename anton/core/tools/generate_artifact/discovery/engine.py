"""Phase A of the artifact pipeline: a bounded ReAct loop that determines
the artifact type, gathers/verifies data, and asks clarifying questions.

Shape mirrors generate_artifact/engine.py's `_run_loop` — same tool-call
protocol (Anthropic-style tool_use / tool_result blocks), same round budget
idea — but with a different, smaller tool set and a control tool
(`finish_gathering`) instead of `finish`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from anton.core.artifacts.models import ARTIFACT_TYPES

from .. import sub_tools as protocol
from . import prompts, sub_tools
from .notes import render_gathering_notes, string_list
from .state import PrdState, gathering_question_budget

if TYPE_CHECKING:
    from anton.chat_session import ChatSession


# Same order of magnitude as generate_artifact's MAX_ROUNDS=20: the loop also
# spends rounds on scratchpad/web_search/web_fetch calls, not just on
# ask_user, so a much smaller cap would cut off legitimate data-gathering.
MAX_ROUNDS = 20


def _web_tools() -> dict[str, tuple]:
    """`name -> (handler, input field recorded in `web_calls`)` for the two web
    sub-tools. Both forward to the main agent's fallback handlers."""
    from anton.core.tools.web_tools import (
        handle_web_fetch_fallback,
        handle_web_search_fallback,
    )

    return {
        "web_search": (handle_web_search_fallback, "query"),
        "web_fetch": (handle_web_fetch_fallback, "url"),
    }


def _exec_ran(outcome) -> bool:
    """Whether a scratchpad exec result describes a cell that executed.

    False for the single-scratchpad guard's challenge (`ok=True`, guidance
    rather than a failure, hence the `reason`) and for a cell the runtime
    reported as failed: neither is working data-access code.
    """
    if getattr(outcome, "reason", "") == "new_scratchpad_challenged":
        return False
    return getattr(outcome, "ok", None) is not False


async def run_gathering_loop(state: "PrdState") -> None:
    """Run phase 1 to completion (or until MAX_ROUNDS is exhausted).

    Mutates `state` in place:
      - `state.messages` gains this call's kickoff/continue message plus
        every round's assistant/tool_result blocks — re-entrant: a second
        call (from the orchestrator's `back_to_gathering` branch) appends
        rather than resetting, so phase 2's exchange survives.
      - `state.qa_log` gains one entry per `ask_user` call actually asked.
      - `state.final_artifact_type` / `state.gathering_notes` are set by
        `finish_gathering`. Left as `""` / the model's last text if the
        loop exhausts `MAX_ROUNDS`, or the model stops without calling
        `finish_gathering` — both are the best-effort case the caller
        (orchestrator.run) falls back on.
    """
    budget = gathering_question_budget(state.session)
    # One array for the whole shared-prefix region; availability per step is
    # decided in code by `sub_tools.rejection_for`, not by dropping entries.
    tools = state.pipeline_tools
    system = state.pipeline_system
    step = sub_tools.STEP_GATHERING

    if not state.messages:
        state.messages.append({
            "role": "user",
            "content": prompts.build_call_kickoff(state)
            + "\n\n"
            + prompts.step_message(step, state),
        })
    else:
        state.messages.append({
            "role": "user",
            "content": prompts.step_message(
                step, state, extra=prompts.GATHERING_CONTINUE
            ),
        })

    state.step_started(sub_tools.STEP_GATHERING)
    questions_asked = 0

    for round_idx in range(MAX_ROUNDS):
        if state.winding_down():
            # Stop collecting and let the caller write down what we have.
            # `gathering_complete` stays False, which is also condition 1 of
            # the emergency data loop — so a continuation automatically
            # re-checks the data this run never finished vouching for.
            state.record("gathering", "stopped_over_budget", f"after round {round_idx}")
            state.gathering_complete = False
            return
        method = "plan" if round_idx == 0 else "code"
        llm_call = state.session._llm.plan if round_idx == 0 else state.session._llm.code
        await sub_tools.signal_thinking(state.session)
        response = await llm_call(system=system, messages=state.messages, tools=tools)
        state.trace_log.llm_call(
            node="gathering", method=method, system=system,
            messages=state.messages, response=response, round=round_idx,
        )

        if not response.tool_calls:
            state.gathering_notes = (response.content or "").strip()
            # Recorded in `messages` too, not just `gathering_notes` — phase
            # 2 (`draft_brief`, `write_prd`) reads only `state.messages`, so
            # without this the model's entire best-effort summary would be
            # silently dropped and draft_brief would work from the bare
            # kickoff alone.
            if state.gathering_notes:
                state.messages.append({"role": "assistant", "content": state.gathering_notes})
            state.trace_log.node("gathering", "done", detail=state.gathering_notes[:200])
            return

        state.messages.append(
            {"role": "assistant", "content": protocol.assistant_blocks(response)}
        )

        result_blocks: list[dict] = []
        finished = False
        for tc in response.tool_calls:
            if tc.parse_error:
                result_blocks.append(protocol.malformed_input_result(tc))
                continue

            name = tc.name
            inp = tc.input or {}

            # Availability is enforced here, not by editing the tool array:
            # the array is part of the cached prefix and has to stay
            # byte-identical across every call in phases A-D.
            reason = sub_tools.rejection_for(
                step, name, questions_left=budget - questions_asked
            )
            if reason is not None:
                state.trace_log.tool_rejected(node=step, tool=name, reason=reason)
                result_blocks.append(protocol.tool_result(tc.id, reason))
                continue

            if name == "finish_gathering":
                claimed_type = str(inp.get("artifact_type") or "")
                # The schema's `enum` (see sub_tools.FINISH_GATHERING_SCHEMA)
                # is a hint, not an enforced constraint — a model can still
                # emit a type outside ARTIFACT_TYPES. Left unchecked, that
                # string reaches `write_prd`'s `ArtifactStore.update(type=...)`
                # unvalidated until much later, where it raises ValueError
                # and crashes the whole run instead of just
                # falling back to the type that was already known-good.
                state.final_artifact_type = (
                    claimed_type if claimed_type in ARTIFACT_TYPES else state.artifact_type
                )
                state.gathering_notes = render_gathering_notes(inp)
                state.assumptions = string_list(inp.get("assumptions"))
                state.open_points = string_list(inp.get("open_points"))
                state.declared_sources = string_list(inp.get("data_sources"))
                # On the hot path the gathering loop had the data tools and
                # used them, so a source it declares is one it worked with.
                # `web_calls` counts as much as `scratchpad_execs`: a source
                # that IS a web page is verified by having been fetched, and
                # demanding a scratchpad cell for it would send every
                # web-sourced request through the emergency data loop to
                # re-download an article that was already read. The unverified
                # list is what that loop reads, and after a correction it is
                # filled by `redraw_brief`, not here.
                state.unverified_sources = (
                    []
                    if (state.scratchpad_execs or state.web_calls)
                    else list(state.declared_sources)
                )
                state.gathering_complete = True
                result_blocks.append(protocol.tool_result(tc.id, "ok"))
                finished = True
            elif name == "ask_user":
                # The exhausted-budget case is handled by the gate above, so
                # reaching here means there is a question left to spend.
                outcome = await sub_tools.dispatch_ask_user(state.session, inp)
                questions_asked += 1
                state.record_qa(outcome["question"], outcome["answer_summary"])
                state.trace_log.node(
                    "ask_user", outcome["status"],
                    detail=f"{outcome['question']} -> {outcome['answer_summary']}",
                )
                result_blocks.append(protocol.tool_result(tc.id, outcome["tool_result"]))
            elif name == "scratchpad":
                from anton.core.tools.tool_handlers import handle_scratchpad

                outcome = await handle_scratchpad(state.session, inp)
                content = protocol.unwrap_outcome(outcome)
                state.trace_log.scratchpad(node="scratchpad", input=inp, output=content)
                # Raw material for `notes.render_exec_notes`: the working
                # data-access code is what phase E needs, and it must not
                # depend on the model mentioning it in a summary. Only a cell
                # that RAN counts: the twenty-third live run recorded an exec
                # the single-scratchpad guard had refused, and both generators
                # received its code with the refusal text as its "Output".
                if inp.get("action") == "exec" and inp.get("code") and _exec_ran(outcome):
                    state.scratchpad_execs.append({
                        "name": inp.get("name"),
                        "code": inp.get("code"),
                        "output": content if isinstance(content, str) else str(content),
                    })
                result_blocks.append(protocol.tool_result(tc.id, content))
            elif name in ("web_search", "web_fetch"):
                handler, field = _web_tools()[name]
                content = await handler(state.session, inp)
                state.trace_log.scratchpad(node=name, input=inp, output=content)
                # Raw material for `notes.render_web_notes`; the field the
                # call was made with (`query` or `url`) is what the notes cite.
                call = {"kind": name, "query": "", "url": "", "title": ""}
                call[field] = str(inp.get(field) or "")
                call["excerpt"] = content if isinstance(content, str) else str(content)
                state.web_calls.append(call)
                result_blocks.append(protocol.tool_result(tc.id, content))
            else:
                result_blocks.append(
                    protocol.tool_result(tc.id, f"Error: unknown sub-tool `{name}`.")
                )

        state.messages.append({"role": "user", "content": result_blocks})
        if finished:
            state.trace_log.node(
                "gathering", "done",
                detail=f"finish_gathering: type={state.final_artifact_type}",
            )
            return

    # MAX_ROUNDS exhausted without finish_gathering — best-effort; caller
    # checks state.final_artifact_type (still "") to detect this case.
    state.trace_log.node("gathering", "fail", detail="MAX_ROUNDS exhausted without finish_gathering")
