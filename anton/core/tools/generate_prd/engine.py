"""Phase 1 of generate_prd: a bounded ReAct loop that determines the
artifact type, gathers/verifies data, and asks clarifying questions.

Shape mirrors generate_artifact/engine.py's `_run_loop` — same tool-call
protocol (Anthropic-style tool_use / tool_result blocks), same round budget
idea — but with a different, smaller tool set and a control tool
(`finish_gathering`) instead of `finish`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import sub_tools
from .prompts import (
    build_gathering_continue_message,
    build_gathering_kickoff,
    build_gathering_system_prompt,
)
from .state import PrdState, gathering_question_budget

if TYPE_CHECKING:
    from anton.chat_session import ChatSession


# Same order of magnitude as generate_artifact's MAX_ROUNDS=20: the loop also
# spends rounds on scratchpad/web_search/web_fetch calls, not just on
# ask_user, so a much smaller cap would cut off legitimate data-gathering.
MAX_ROUNDS = 20


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
    tools = sub_tools.tool_schemas(include_ask_user=budget > 0)

    if not state.messages:
        state.messages.append({"role": "user", "content": build_gathering_kickoff(state)})
    else:
        state.messages.append({"role": "user", "content": build_gathering_continue_message()})
    system = build_gathering_system_prompt(state)

    questions_asked = 0

    for round_idx in range(MAX_ROUNDS):
        llm_call = state.session._llm.plan if round_idx == 0 else state.session._llm.code
        response = await llm_call(system=system, messages=state.messages, tools=tools)

        if not response.tool_calls:
            state.gathering_notes = (response.content or "").strip()
            # Recorded in `messages` too, not just `gathering_notes` — phase
            # 2 (`draft_brief`, `write_prd`) reads only `state.messages`, so
            # without this the model's entire best-effort summary would be
            # silently dropped and draft_brief would work from the bare
            # kickoff alone.
            if state.gathering_notes:
                state.messages.append({"role": "assistant", "content": state.gathering_notes})
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

            if name == "finish_gathering":
                state.final_artifact_type = str(inp.get("artifact_type") or state.artifact_type)
                state.gathering_notes = str(inp.get("notes") or inp.get("summary") or "")
                result_blocks.append({"type": "tool_result", "tool_use_id": tc.id, "content": "ok"})
                finished = True
            elif name == "ask_user":
                if questions_asked >= budget:
                    result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": (
                                "Question limit reached for this turn; "
                                "proceed on a stated assumption instead of "
                                "asking again."
                            ),
                        }
                    )
                    continue
                outcome = await sub_tools.dispatch_ask_user(state.session, inp)
                questions_asked += 1
                state.record_qa(outcome["question"], outcome["answer_summary"])
                result_blocks.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": outcome["tool_result"]}
                )
            elif name == "scratchpad":
                from anton.core.tools.tool_handlers import handle_scratchpad

                content = await handle_scratchpad(state.session, inp)
                result_blocks.append({"type": "tool_result", "tool_use_id": tc.id, "content": content})
            elif name == "web_search":
                from anton.core.tools.web_tools import handle_web_search_fallback

                content = await handle_web_search_fallback(state.session, inp)
                result_blocks.append({"type": "tool_result", "tool_use_id": tc.id, "content": content})
            elif name == "web_fetch":
                from anton.core.tools.web_tools import handle_web_fetch_fallback

                content = await handle_web_fetch_fallback(state.session, inp)
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
            return

    # MAX_ROUNDS exhausted without finish_gathering — best-effort; caller
    # checks state.final_artifact_type (still "") to detect this case.
