"""No API calls: preserve message/tool contracts and safe routing boundaries."""
import copy
import os
from types import SimpleNamespace
import pytest

from anton.core.llm.openai import OpenAIProvider

NOTE = True
TOOL = {'name':'read_file', 'description':'Read a file',
        'input_schema':{'type':'object','properties':{}}}


def kwargs(messages, *, choice=None, tools=True):
    provider = object.__new__(OpenAIProvider)
    provider._supports_vision = True
    provider._flavor = 'openai'
    provider._reasoning_effort = 'low'
    return provider._build_responses_kwargs(model='offline', system='trusted policy',
        messages=messages, tools=[TOOL] if tools else None, tool_choice=choice,
        max_tokens=100, native_web_tools=None)


@pytest.mark.parametrize('followup', [False, True])
def test_turn_note_is_adjacent_and_does_not_mutate_history(followup):
    messages = [{'role':'user','content':'Make a report'},
                {'role':'assistant','content':'Done'}]
    if followup:
        messages += [{'role':'user','content':[{'type':'text','text':'Update it'}]},
            {'role':'assistant','content':[{'type':'tool_use','id':'call_x','name':'read_file','input':{}}]},
            {'role':'user','content':[{'type':'tool_result','tool_use_id':'call_x','content':'actual source'}]}]
    original = copy.deepcopy(messages)
    result = kwargs(messages)
    assert messages == original
    notes = [item for item in result['input'] if item.get('role') == 'developer']
    assert len(notes) == int(NOTE)
    if NOTE:
        index = max(i for i, item in enumerate(result['input']) if item.get('role') == 'user')
        assert result['input'][index + 1] == notes[0]
        assert 'same response' in notes[0]['content']
    if followup:
        assert [item['type'] for item in result['input']][-2:] == ['function_call','function_call_output']
        assert result['input'][-1]['output'] == 'actual source'
    assert result['instructions'] == 'trusted policy'
    assert result['reasoning']['effort'] == 'low'


@pytest.mark.parametrize('messages,choice,tools', [
    ([{'role':'user','content':'Hello'}], None, False),
    ([{'role':'user','content':'Check'}], {'type':'tool','name':'read_file'}, True),
    ([{'role':'assistant','content':'Internal context'}], None, True),
])
def test_no_opening_note_for_non_tool_or_forced_internal_calls(messages, choice, tools):
    assert not any(item.get('role') == 'developer' for item in kwargs(messages,choice=choice,tools=tools)['input'])


