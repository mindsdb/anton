"""Verify cache boundaries preserve policy/context and API compatibility."""
import copy
import pytest
from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext, SESSION_CONTEXT_MARKER
from anton.core.llm.openai import OpenAIProvider


def request(system, model='gpt-6.1-sol', flavor='openai'):
    p=object.__new__(OpenAIProvider)
    p._supports_vision=True;p._flavor=flavor;p._reasoning_effort='low'
    messages=[{'role':'user','content':'Inspect current files'},
        {'role':'assistant','content':[{'type':'tool_use','id':'x','name':'read','input':{}}]},
        {'role':'user','content':[{'type':'tool_result','tool_use_id':'x','content':'real result'}]}]
    original=copy.deepcopy(messages)
    result=p._build_responses_kwargs(model=model, system=system, messages=messages,
        tools=[{'name':'read','description':'Read file','input_schema':{'type':'object','properties':{}}}],
        tool_choice=None,max_tokens=100,native_web_tools=None)
    assert messages==original
    return result


def test_cross_project_static_boundary_and_all_context_retained():
    class Skills:
        def list_summaries(self):
            return [{'label':'Planning','description':'Read actual source and reconcile outputs'}]
    def build(label):
        return ChatSystemPromptBuilder().build(conversation_started='same',
            system_prompt_context=SystemPromptContext(prefix='trusted prefix',suffix='suffix '+label),
            proactive_dashboards=True,output_dir='unused',tool_defs=None,
            project_context='project '+label,self_awareness_context='identity '+label,
            datasource_context='connections '+label,skill_store=Skills(),
            memory_context='memory '+label,workspace_context='workspace '+label)
    a,b=build('alpha'),build('beta')
    prefix,tail=a.split(SESSION_CONTEXT_MARKER,1)
    assert b.split(SESSION_CONTEXT_MARKER,1)[0]==prefix
    assert 'Procedural memory' in prefix and 'Planning' in prefix
    for field in ['project','identity','connections','suffix','memory','workspace']:
        assert field+' alpha' in tail and field+' beta' not in a
    q=request(a)
    assert 'instructions' not in q
    first=q['input'][0];second=q['input'][1]
    assert first['content'][0]['text']==prefix
    assert first['content'][0]['prompt_cache_breakpoint']=={'mode':'explicit'}
    assert first['role']==second['role']=='developer'
    assert second['content']==SESSION_CONTEXT_MARKER+tail
    assert first['content'][0]['text']+second['content']==a
    assert q['input'][-1]['type']=='function_call_output' and q['input'][-1]['output']=='real result'
    assert q['tools'][0]['name']=='read'


@pytest.mark.parametrize('model,flavor,system',[
    ('older-model','openai','policy'+SESSION_CONTEXT_MARKER+'project'),
    ('gpt-6.1-sol','openai-compatible-generic','policy'+SESSION_CONTEXT_MARKER+'project'),
    ('gpt-6.1-sol','openai','plain policy'),
    ('gpt-6.1-sol','openai','policy'+SESSION_CONTEXT_MARKER+'possible quoted marker'+SESSION_CONTEXT_MARKER+'project'),
])
def test_unsupported_or_ambiguous_boundary_preserves_original(model,flavor,system):
    q=request(system,model,flavor)
    assert q['instructions']==system
    assert not any(isinstance(i.get('content'),list) and any('prompt_cache_breakpoint' in b for b in i['content']) for i in q['input'])


def test_builder_without_dynamic_context_retains_unsplit_policy():
    p=ChatSystemPromptBuilder().build(conversation_started='same',system_prompt_context=SystemPromptContext(),
        proactive_dashboards=True,output_dir='unused')
    assert SESSION_CONTEXT_MARKER not in p
    assert request(p)['instructions']==p
