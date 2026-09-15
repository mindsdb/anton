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

from . import prompts, sub_tools
from .state import PrdState, gathering_question_budget

if TYPE_CHECKING:
    from anton.chat_session import ChatSession


# Same order of magnitude as generate_artifact's MAX_ROUNDS=20: the loop also
# spends rounds on scratchpad/web_search/web_fetch calls, not just on
# ask_user, so a much smaller cap would cut off legitimate data-gathering.
MAX_ROUNDS = 20


def _unwrap_outcome(result):
    """Flatten a handler result down to `tool_result`-ready content.

    Same reason as generate_artifact/engine.py's twin: since ENG-696
    `handle_scratchpad`'s exec path returns a `ToolOutcome` rather than a
    bare string, and only its `content` belongs in a tool_result block —
    the ok/reason verdict feeds the outer agent's error streak, which this
    loop is not part of.
    """
    from anton.core.tools.registry import ToolOutcome

    return result.content if isinstance(result, ToolOutcome) else result


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

        assistant_blocks: list[dict] = []
        if response.content:
            assistant_blocks.append({"type": "text", "text": response.content})
        for tc in response.tool_calls:
            assistant_blocks.append(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
            )
        state.messages.append({"role": "assistant", "content": assistant_blocks})

        result_blocks: list[dict] = []
        finished = False
        for tc in response.tool_calls:
            if tc.parse_error:
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": (
                            "Error: malformed tool input — re-emit with "
                            f"valid JSON. ({tc.parse_error})"
                        ),
                    }
                )
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
                result_blocks.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": reason}
                )
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
                state.gathering_notes = str(inp.get("notes") or inp.get("summary") or "")
                raw_sources = inp.get("data_sources")
                state.declared_sources = (
                    [str(s) for s in raw_sources] if isinstance(raw_sources, list) else []
                )
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
                result_blocks.append({"type": "tool_result", "tool_use_id": tc.id, "content": "ok"})
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
                result_blocks.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": outcome["tool_result"]}
                )
            elif name == "scratchpad":
                from anton.core.tools.tool_handlers import handle_scratchpad

                content = _unwrap_outcome(await handle_scratchpad(state.session, inp))
                state.trace_log.scratchpad(node="scratchpad", input=inp, output=content)
                # Raw material for `notes.render_exec_notes`: the working
                # data-access code is what phase E needs, and it must not
                # depend on the model mentioning it in a summary.
                if inp.get("action") == "exec" and inp.get("code"):
                    state.scratchpad_execs.append({
                        "name": inp.get("name"),
                        "code": inp.get("code"),
                        "output": content if isinstance(content, str) else str(content),
                    })
                result_blocks.append({"type": "tool_result", "tool_use_id": tc.id, "content": content})
            elif name == "web_search":
                from anton.core.tools.web_tools import handle_web_search_fallback

                content = await handle_web_search_fallback(state.session, inp)
                state.trace_log.scratchpad(node="web_search", input=inp, output=content)
                state.web_calls.append({
                    "kind": "web_search",
                    "query": str(inp.get("query") or ""),
                    "url": "",
                    "title": "",
                    "excerpt": content if isinstance(content, str) else str(content),
                })
                result_blocks.append({"type": "tool_result", "tool_use_id": tc.id, "content": content})
            elif name == "web_fetch":
                from anton.core.tools.web_tools import handle_web_fetch_fallback

                content = await handle_web_fetch_fallback(state.session, inp)
                state.trace_log.scratchpad(node="web_fetch", input=inp, output=content)
                state.web_calls.append({
                    "kind": "web_fetch",
                    "query": "",
                    "url": str(inp.get("url") or ""),
                    "title": "",
                    "excerpt": content if isinstance(content, str) else str(content),
                })
                result_blocks.append({"type": "tool_result", "tool_use_id": tc.id, "content": content})
            else:
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": f"Error: unknown sub-tool `{name}`.",
                    }
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
