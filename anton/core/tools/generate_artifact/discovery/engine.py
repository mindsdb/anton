"""Phase A of the artifact pipeline: a bounded ReAct loop that determines
the artifact type, gathers/verifies data, and asks clarifying questions.

Shape mirrors generate_artifact/engine.py's `_run_loop` — same tool-call
protocol (Anthropic-style tool_use / tool_result blocks), same round budget
idea — but with a different, smaller tool set and a control tool
(`finish_gathering`) instead of `finish`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from anton.core.utils.scratchpad import CELL_NOT_RUN_REASONS, scratchpad_action

from .. import data_files
from .. import sub_tools as protocol
from ..data_files import DATA_NOT_SAVED_WARNING
from ..state import known_data_is_reference, known_data_usable, settled_artifact_type
from . import prompts, sub_tools
from .notes import (
    known_data_sources,
    render_gathering_notes,
    source_key,
    string_list,
    web_verified_sources,
)
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

    False for an `ok=True` result whose cell never ran (the single-scratchpad
    guard's challenge, a change the user didn't approve), hence the `reason`,
    and for a cell the runtime reported as failed: none is working
    data-access code.
    """
    if getattr(outcome, "reason", "") in CELL_NOT_RUN_REASONS:
        return False
    return getattr(outcome, "ok", None) is not False


_NOT_FINISHED = "Not finished — fix the data files and call `finish_gathering` again:\n"
_SAVE_DATA = (
    "Not finished — save the data the page needs into the data files folder named in the "
    "step message and list it in `data_files`, then call `finish_gathering` again. Sources "
    "without a data file: "
)


def _read_by_cells(state: "PrdState", inp: dict) -> list[str]:
    """The declared sources only a cell can have read: not marked as coming
    from usable `known_data` or from the web."""
    elsewhere = known_data_sources(inp, usable=known_data_usable(state))
    if state.web_calls:
        elsewhere |= web_verified_sources(inp)
    return [s for s in state.declared_sources if source_key(s) not in elsewhere]


def _absorb_data_files(state: "PrdState", inp: dict) -> str | None:
    """For an html-app: take the declared data files, or say why gathering
    cannot finish yet. Data read by code is sent back once to be saved; the
    second call is accepted either way, so gathering cannot get stuck on
    data it cannot save.

    A refusal reopens gathering: if the model never calls `finish_gathering`
    again, the data loop has to vouch for the sources instead."""
    if settled_artifact_type(state) != "html-app":
        return None
    declared = data_files.parse_declared(inp.get("data_files"))
    valid, errors = (data_files.check_declared(
        state.artifact_path, declared, state.declared_sources,
        cell_sandbox=data_files.sandbox_of(state.session), protected=state.data_snapshot,
    ) if declared else ([], []))
    if errors:
        state.gathering_complete = False
        # The refused call's own files are not recorded, so they cover nothing.
        covered = data_files.covered_sources(state.data_files)
        state.unverified_sources = [s for s in state.declared_sources if source_key(s) not in covered]
        return _NOT_FINISHED + "\n".join(f"- {e}" for e in errors)
    state.data_files = data_files.merge(state.data_files, valid, state.declared_sources)
    if state.data_save_requested or not state.declared_sources:
        return None
    covered = data_files.covered_sources(state.data_files)
    if known_data_is_reference(state):
        web = web_verified_sources(inp) if state.web_calls else set()
        missing = [s for s in state.declared_sources if source_key(s) not in covered | web]
    elif state.scratchpad_execs and not state.data_files:
        missing = _read_by_cells(state, inp)
    else:
        missing = []
    if not missing:
        return None
    state.data_save_requested = True
    state.gathering_complete = False
    state.unverified_sources = missing
    return _SAVE_DATA + ", ".join(missing) + "."


def _apply_html_app_coverage(state: "PrdState", inp: dict) -> None:
    """Data files verify the sources they cover. A reference to pad data
    verifies nothing on its own, whatever else ran."""
    # A `back_to_gathering` that saves the data clears what an earlier pass marked.
    data_files.clear_data_not_saved(state)
    covered = data_files.covered_sources(state.data_files)
    if known_data_is_reference(state):
        web = web_verified_sources(inp) if state.web_calls else set()
        state.unverified_sources = [
            s for s in state.declared_sources if source_key(s) not in covered | web
        ]
        state.known_data_verified = []
        return
    state.unverified_sources = [s for s in state.unverified_sources if source_key(s) not in covered]
    unsaved = _read_by_cells(state, inp) if state.scratchpad_execs and not state.data_files else []
    if unsaved:
        state.record("gathering", "data_not_saved", ", ".join(unsaved))
        state.data_not_saved = True
        if DATA_NOT_SAVED_WARNING not in state.verify_warnings:
            state.verify_warnings.append(DATA_NOT_SAVED_WARNING)


async def run_gathering_loop(state: "PrdState") -> None:
    """Run phase 1 to completion (or until MAX_ROUNDS is exhausted).

    Mutates `state` in place:
      - `state.messages` gains this call's kickoff/continue message plus
        every round's assistant/tool_result blocks — re-entrant: a second
        call (from the orchestrator's `back_to_gathering` branch) appends
        rather than resetting, so phase 2's exchange survives.
      - `state.qa_log` gains one entry per `ask_user` call actually asked.
      - `state.final_artifact_type` / `state.gathering_notes` are set by
        every `finish_gathering` call, a refused one included. With no call
        at all they stay `""` / the model's last text.
      - `state.gathering_complete` is set by an accepted `finish_gathering`
        and cleared by a refused one. Exhausting `MAX_ROUNDS` or stopping
        without the call leaves it as it was, False on a first pass: the
        best-effort case in which the caller's emergency data loop
        re-checks the data.
    """
    budget = gathering_question_budget(state.session, act_first=state.act_first)
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
    # One save refusal per gathering pass: a re-entry reads new data.
    state.data_save_requested = False
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
        role = "planning" if round_idx == 0 else "coding"
        await sub_tools.signal_thinking(state.session)
        room = await sub_tools.call_with_room(
            state, step, role=role, system=system, messages=state.messages,
            tools=tools, round_idx=round_idx,
        )
        response = room.response
        if room.truncated:
            # Cut off without a retry or after one: half a summary must not
            # become the notes, and a damaged call must not run.
            # `gathering_complete` stays False, so the emergency data loop
            # re-checks the data.
            state.record("gathering", "truncated", f"round {round_idx}")
            state.gathering_complete = False
            return

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
                # emit a type outside the enum. Left unchecked, that string
                # reaches `write_prd`'s `ArtifactStore.update(type=...)`
                # unvalidated until much later, where it raises ValueError
                # and crashes the whole run instead of just falling back to
                # the type that was already known-good. The check is against
                # what this pipeline BUILDS, not every registered type: a
                # `document` or `dataset` is valid metadata but has no
                # generator, and settling on one would build a fullstack
                # app for it and lock the slug out of any repeat call.
                # Where nothing starts an app backend, html-app is the only
                # buildable type (`GenState.buildable_types`).
                state.final_artifact_type = (
                    claimed_type if claimed_type in state.buildable_types
                    else state.artifact_type
                )
                state.gathering_notes = render_gathering_notes(inp)
                state.assumptions = string_list(inp.get("assumptions"))
                state.open_points = string_list(inp.get("open_points"))
                state.declared_sources = string_list(inp.get("data_sources"))
                refusal = _absorb_data_files(state, inp)
                if refusal is not None:
                    state.trace_log.node("gathering", "data_files_refused", detail=refusal[:200])
                    result_blocks.append(protocol.tool_result(tc.id, refusal))
                    continue
                # On the hot path the gathering loop had the data tools and
                # used them, so a source it declares is one it worked with.
                # `web_calls` counts as much as `scratchpad_execs`: a source
                # that IS a web page is verified by having been fetched, and
                # demanding a scratchpad cell for it would send every
                # web-sourced request through the emergency data loop to
                # re-download an article that was already read. An html-app
                # source whose rows the calling agent passed in is verified
                # by the `known_data` mark (`known_data_usable`). The
                # unverified list is what that loop reads, and after a
                # correction it is filled by `redraw_brief`, not here.
                covered = known_data_sources(inp, usable=known_data_usable(state))
                executed = bool(state.scratchpad_execs or state.web_calls)
                state.unverified_sources = (
                    []
                    if executed
                    else [s for s in state.declared_sources if source_key(s) not in covered]
                )
                state.known_data_verified = (
                    []
                    if executed
                    else [s for s in state.declared_sources if source_key(s) in covered]
                )
                if settled_artifact_type(state) == "html-app":
                    _apply_html_app_coverage(state, inp)
                state.gathering_complete = True
                section = data_files.state_files_section(state)
                result_blocks.append(protocol.tool_result(tc.id, "ok" + (f"\n\n{section}" if section else "")))
                finished = True
            elif name == "ask_user":
                # The exhausted-budget case is handled by the gate above, so
                # reaching here means there is a question left to spend.
                outcome = await sub_tools.dispatch_ask_user(state.session, inp)
                questions_asked += 1
                if outcome["status"] == "timeout":
                    state.question_unanswered = True
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
                # that RAN counts: a live run once recorded an exec
                # the single-scratchpad guard had refused, and both generators
                # received its code with the refusal text as its "Output".
                if scratchpad_action(inp) == "exec" and inp.get("code") and _exec_ran(outcome):
                    state.scratchpad_execs.append({
                        "name": inp.get("name"),
                        "code": inp.get("code"),
                        "output": protocol.result_text(content),
                    })
                result_blocks.append(protocol.tool_result(tc.id, content))
            elif name in ("web_search", "web_fetch"):
                handler, field = _web_tools()[name]
                # Both fallbacks answer with a `ToolOutcome`;
                # the model, the trace and `web_notes` all want its text, not
                # the dataclass repr — same unwrap as the scratchpad branch.
                content = protocol.unwrap_outcome(await handler(state.session, inp))
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

    # MAX_ROUNDS exhausted without an accepted finish_gathering — best-effort;
    # the caller's data loop re-checks unless an earlier pass vouched for it.
    state.trace_log.node("gathering", "fail", detail="MAX_ROUNDS exhausted without finish_gathering")
