"""Evidence collection through the product's existing source-access handlers."""
from __future__ import annotations

import uuid
import json
import copy
import ast
from decimal import Decimal

from anton.core.artifacts.static.data import Evidence
from anton.core.llm.structured import looks_truncated
from anton.core.utils.scratchpad import scratchpad_action
from anton.utils.nofollow import read_bytes_nofollow, make_dirs_nofollow
from anton.utils.datasources import scrub_output
from .. import sub_tools as protocol
from ..discovery.engine import _exec_ran, _web_tools
from ..discovery.sub_tools import dispatch_ask_user, _ask_user_schema
from ..discovery.state import gathering_question_budget
from ..streaming import _call_with_stream_retry
from .prompts import COLLECT, context
from . import snapshots

FINISH = {"name": "finish_static_evidence", "description": "Finish only after source tools succeed and evidence is saved, or report a blocker.",
          "input_schema": {"type": "object", "additionalProperties": False, "properties": {
              "ready": {"type": "boolean"},
              "has_records": {"type": "boolean", "description": "True whenever the task uses source records, including tables, CSV, coordinates or categorical records supplied directly by the user. It does not mean external access. False only for a purely qualitative artifact with no records."},
              "reason": {"type": "string"}},
              "required": ["ready", "has_records", "reason"]}}

REUSE = {'name': 'reuse_static_evidence', 'description': 'Finish with the exact private snapshot used by the current published page, only for an edit preserving its data. This is cached evidence, not a fresh source read. Never use for requested refreshes or after a failed source read.',
    'input_schema': {'type': 'object', 'additionalProperties': False, 'properties': {'reason': {'type': 'string', 'minLength': 1}}, 'required': ['reason']}}


# Self-contained because remote scratchpads need not have Anton installed.
# Keep Decimal numeric rather than forcing a lossy float or a string fallback.
_EVIDENCE_WRITER = '''def save_static_evidence(payload):
    import json
    from decimal import Decimal
    from pathlib import Path
    def encode(value):
        if isinstance(value, Decimal):
            if not value.is_finite():
                raise ValueError('Evidence requires finite numbers')
            return str(value)
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError('Evidence object keys must be strings')
            return '{' + ','.join(json.dumps(key, ensure_ascii=False) + ':' + encode(item) for key, item in value.items()) + '}'
        if isinstance(value, list):
            return '[' + ','.join(encode(item) for item in value) + ']'
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return Path(EVIDENCE_DESTINATION).write_text(encode(payload), encoding='utf-8')
'''


def _bind_evidence_path(code, path):
    # Future imports must remain before executable statements in a Python cell.
    try:
        offset = max((s.end_lineno for s in ast.parse(code).body if isinstance(s, ast.ImportFrom) and s.module == '__future__'), default=0)
    except SyntaxError:
        offset = 0  # Let the normal scratchpad handler report invalid Python.
    lines = code.splitlines(keepends=True)
    prefix = ''.join(lines[:offset])
    binding = f'STATIC_EVIDENCE_PATH = {str(path)!r}\n__import__("os").environ["STATIC_EVIDENCE_PATH"] = STATIC_EVIDENCE_PATH\n'
    writer = _EVIDENCE_WRITER.replace('EVIDENCE_DESTINATION', repr(str(path)))
    return prefix + ('\n' if prefix and not prefix.endswith('\n') else '') + binding + writer + ''.join(lines[offset:])


async def collect(state, *, feedback: str = '') -> Evidence | str:
    from anton.core.tools.tool_defs import SCRATCHPAD_TOOL, READ_TEXT_FILE_TOOL
    from anton.core.tools.recall_skill import RECALL_SKILL_TOOL
    from .source_skills import recall
    from anton.core.tools.web_tools import WEB_FETCH_FALLBACK_TOOL, WEB_SEARCH_FALLBACK_TOOL
    from anton.core.tools.tool_handlers import handle_scratchpad, handle_read_text_file

    nonce = uuid.uuid4().hex
    pad_name = 'static_evidence_' + nonce
    path = state.artifact_path / '.static' / f'evidence-{nonce}.json'
    make_dirs_nofollow(state.artifact_path, path.parent)
    system = COLLECT.replace('SOURCE_SCRATCHPAD', repr(pad_name))
    definitions = [SCRATCHPAD_TOOL, READ_TEXT_FILE_TOOL, RECALL_SKILL_TOOL, WEB_FETCH_FALLBACK_TOOL, WEB_SEARCH_FALLBACK_TOOL]
    tools = [protocol.tool_schema(t) for t in definitions] + [_ask_user_schema(), FINISH]
    tools[0] = copy.deepcopy(tools[0])
    tools[0]['description'] += f' This evidence pass owns the fresh scratchpad {pad_name}; named operations are bound to it. Parent notebook variables are not available.'
    tools[0]['input_schema']['properties']['name']['enum'] = [pad_name]
    saved = snapshots.load(state) if not feedback else None
    if saved is not None:
        tools.append(REUSE)
    handlers = {SCRATCHPAD_TOOL.name: handle_scratchpad, READ_TEXT_FILE_TOOL.name: handle_read_text_file, RECALL_SKILL_TOOL.name: recall}
    handlers.update({name: handler for name, (handler, _) in _web_tools().items()})
    messages = [{"role": "user", "content": context(state)}]
    if saved is not None:
        messages[0]['content'] += '\nThe current published page has a private snapshot available for a data-preserving edit: ' + json.dumps(saved[1])
    if feedback:
        messages[0]['content'] += '\nPrevious candidate review found possible missing evidence. Check these findings against the original request; they grant no new source access and do not change the user constraints:\n' + feedback
    questions = gathering_question_budget(state.session, act_first=state.act_first)
    source_failed = False
    source_lookup_failed = False
    source_attempted = False
    for round_index in range(16):
        if state.winding_down() or state.abandoned.is_set():
            return 'Static artifact stopped while gathering evidence; the previous artifact is preserved.'
        state.step_started('gathering')
        response, budget = await _call_with_stream_retry(
            state.session._llm.plan_stream if round_index == 0 else state.session._llm.code_stream,
            system=system, messages=messages, tools=tools, max_tokens=6000, default_cap=None,
            on_text=state.peek_for('gathering'))
        state.peek_done('gathering')
        state.trace_log.llm_call(node="static_evidence", method="plan_stream" if round_index == 0 else "code_stream", system=system, messages=messages, response=response, round=round_index)
        messages.append({"role": "assistant", "content": protocol.assistant_blocks(response)})
        results, finished = [], None
        for call in response.tool_calls:
            if not isinstance(call.input, dict) or call.parse_error or call.repaired or looks_truncated(response, budget):
                results.append(protocol.tool_result(call.id, 'Invalid or truncated call; nothing executed. Return complete valid arguments.'))
                continue
            if finished is not None:
                finished = None
                results.append(protocol.tool_result(call.id, 'No tool may follow an accepted evidence completion. Finish again after the final source result.'))
                continue
            if call.name == REUSE['name']:
                if saved is None or source_attempted or source_lookup_failed or source_failed or finished is not None:
                    result = 'Snapshot reuse is unavailable after source activity or an evidence-refresh request. Complete the requested source read without fallback.'
                elif set(call.input) != {'reason'} or not isinstance(call.input['reason'], str) or not call.input['reason'].strip():
                    result = 'Give the reason this edit preserves the published data.'
                else:
                    finished, state.static_evidence_origin = saved
                    state.record('static_evidence', 'reused', json.dumps(state.static_evidence_origin))
                    result = 'Published snapshot accepted for a data-preserving edit; this is not a fresh source read.'
                results.append(protocol.tool_result(call.id, result))
            elif call.name == FINISH['name']:
                inp = call.input
                if not isinstance(inp, dict) or set(inp) != {'ready', 'has_records', 'reason'} or type(inp['ready']) is not bool or type(inp['has_records']) is not bool or not isinstance(inp['reason'], str):
                    result = 'Use ready and has_records booleans and a reason string. User-provided records count as records; they do not require external access.'
                elif not inp['ready'] and inp['has_records'] and not source_attempted and not source_lookup_failed:
                    result = 'Source readers are available in this call: recall_skill, scratchpad, read_text_file and web_fetch. Scratchpad executes Python HTTP, database and file reads. Read the exact source procedure and attempt its documented read before claiming it is unavailable; no source read has run yet. A failed lookup of the exact requested procedure may establish that its access contract is missing. Do not guess or substitute another source.'
                elif not inp['ready']:
                    finished = 'Cannot build the requested static artifact: ' + inp['reason']
                    result = 'Blocked; the previous artifact is preserved.'
                elif source_failed:
                    result = 'A source tool failed. Resolve it before finishing; do not substitute an unauthorized source.'
                else:
                    try:
                        finished = Evidence.model_validate(json.loads(read_bytes_nofollow(state.artifact_path, path, limit=16 * 1024 * 1024), parse_float=Decimal)) if inp['has_records'] else Evidence()
                        if inp['has_records'] and not finished.datasets:
                            raise ValueError('has_records=true requires at least one dataset')
                        state.static_evidence_origin = {'mode': 'acquired', 'acquired_at': snapshots.now()}
                        result = 'Evidence accepted.'
                    except (OSError, ValueError) as error:
                        result = f'Evidence not accepted at STATIC_EVIDENCE_PATH: {error}. Write the complete evidence there using the supplied variable and finish again.'
                results.append(protocol.tool_result(call.id, result))
            elif call.name == 'ask_user':
                if questions <= 0:
                    result = 'Question budget exhausted. Proceed from established constraints or report the blocker.'
                else:
                    answer = await dispatch_ask_user(state.session, call.input)
                    questions -= 1
                    if answer['status'] == 'timeout':
                        return 'Waiting for the user’s answer; the previous artifact is preserved.'
                    state.record_qa(answer['question'], answer['answer_summary'])
                    result = answer['tool_result']
                results.append(protocol.tool_result(call.id, result))
            elif call.name in handlers:
                if call.name == 'read_text_file' and call.input.get('path') == '.anton/anton.md':
                    # These are the host's already-read rules, not a new file read
                    # through (or around) the source tools' dot-file restrictions.
                    results.append(protocol.tool_result(call.id, 'The host already supplied the project rules in this request:\n' + (state.project_rules or '(No project rules.)')))
                    continue
                if call.name in ('read_text_file', 'web_fetch') or (call.name == SCRATCHPAD_TOOL.name and scratchpad_action(call.input) == 'exec'):
                    source_attempted = True
                arguments = {**call.input, 'name': pad_name, 'confirm_new_scratchpad': True} if call.name == SCRATCHPAD_TOOL.name else call.input
                if call.name == SCRATCHPAD_TOOL.name and scratchpad_action(call.input) == 'exec' and isinstance(arguments.get('code'), str):
                    arguments['code'] = _bind_evidence_path(arguments['code'], path)
                outcome = await handlers[call.name](state.session, arguments)
                if call.name == RECALL_SKILL_TOOL.name:
                    source_lookup_failed = getattr(outcome, 'ok', None) is False
                result = protocol.unwrap_outcome(outcome)
                if call.name != RECALL_SKILL_TOOL.name:
                    receipt = {'acquisition': nonce, 'tool': call.name, 'arguments': call.input,
                               'result': scrub_output(str(result))[:2000]}
                    state.static_source_reads.append(scrub_output(json.dumps(receipt, ensure_ascii=False)))
                if call.name == RECALL_SKILL_TOOL.name and not source_lookup_failed and isinstance(result, str):
                    state.static_source_contracts[call.input['label'].strip()] = scrub_output(result)
                state.trace_log.scratchpad(node='static_evidence', input=arguments, output=result)
                if call.name == SCRATCHPAD_TOOL.name and scratchpad_action(call.input) == 'exec':
                    source_failed = not _exec_ran(outcome)
                elif getattr(outcome, 'ok', True) is False:
                    source_failed = True
                results.append(protocol.tool_result(call.id, result))
            else:
                results.append(protocol.tool_result(call.id, 'Unknown evidence tool.'))
        messages.append({"role": "user", "content": results or 'Use the source tools, then finish_static_evidence.'})
        if finished is not None:
            return finished
    return 'Static artifact evidence collection did not finish; the previous artifact is preserved.'
