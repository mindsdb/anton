"""Offline protocol controls: incomplete streams cannot silently finish work."""
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
import pytest
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import StreamComplete, StreamToolUseDelta, StreamToolUseEnd
from anton.core.llm.provider import TransientProviderError

class Events:
    def __init__(self, events): self.events=events
    def __aiter__(self): return self.iterate()
    async def iterate(self):
        for event in self.events: yield event
    async def close(self): pass

async def consume(events):
    client=NS(responses=NS(create=AsyncMock(return_value=Events(events))))
    with patch('anton.core.llm.openai.openai.AsyncOpenAI',return_value=client):
        provider=OpenAIProvider(api_key='offline-control',flavor=OpenAIProvider.FLAVOR_OPENAI)
        return [event async for event in provider.stream(model='gpt-6.1-sol',system='s',messages=[{'role':'user','content':'Perform the requested tool work.'}])]

def response(status='completed',output=None,reason=None):
    return NS(status=status,output=output or [],usage=NS(input_tokens=12,output_tokens=8,input_tokens_details=None),model='gpt-6.1-sol',incomplete_details=NS(reason=reason))

def tool():
    return NS(type='function_call',call_id='call_control',name='scratchpad',arguments='{"action":"exec","code":"print(7)"}')

def text(value):
    return NS(type='message',content=[NS(type='output_text',text=value)])

@pytest.mark.asyncio
async def test_explicit_output_limit_is_not_success():
    events=[NS(type='response.output_text.delta',delta='I will verify it.'),
            NS(type='response.incomplete',response=response('incomplete',[text('I will verify it.')],'max_output_tokens'))]
    result=await consume(events)
    complete=next(e.response for e in result if isinstance(e,StreamComplete))
    assert complete.stop_reason=='max_tokens'
    assert complete.usage.input_tokens==12

@pytest.mark.asyncio
async def test_eof_after_tool_start_does_not_complete_acknowledgement():
    with pytest.raises(TransientProviderError):
        await consume([NS(type='response.output_text.delta',delta='I will verify it.'),
                       NS(type='response.output_item.added',output_index=0,item=tool())])

@pytest.mark.asyncio
async def test_final_response_recovers_complete_tool_missing_done_delta():
    result=await consume([NS(type='response.output_item.added',output_index=0,item=tool()),
                         NS(type='response.completed',response=response(output=[tool()]))])
    complete=next(e.response for e in result if isinstance(e,StreamComplete))
    assert len(complete.tool_calls)==1
    assert complete.tool_calls[0].input=={'action':'exec','code':'print(7)'}
    assert [e.json_delta for e in result if isinstance(e,StreamToolUseDelta)]==[tool().arguments]
    assert len([e for e in result if isinstance(e,StreamToolUseEnd)])==1

@pytest.mark.asyncio
async def test_final_response_completes_partial_arguments_without_duplicating_delta():
    prefix='{"action":'
    result=await consume([NS(type='response.output_item.added',output_index=0,item=tool()),
                         NS(type='response.function_call_arguments.delta',output_index=0,delta=prefix),
                         NS(type='response.completed',response=response(output=[tool()]))])
    deltas=[e.json_delta for e in result if isinstance(e,StreamToolUseDelta)]
    assert ''.join(deltas)==tool().arguments
    assert len(deltas)==2

@pytest.mark.asyncio
async def test_terminal_confirmation_does_not_wait_for_transport_eof():
    class ConfirmedButNotClosed:
        def __aiter__(self): return self.iterate()
        async def iterate(self):
            yield NS(type='response.completed',response=response(output=[text('Done.')]))
            raise AssertionError('Read past the authoritative terminal response')
        async def close(self): pass
    client=NS(responses=NS(create=AsyncMock(return_value=ConfirmedButNotClosed())))
    with patch('anton.core.llm.openai.openai.AsyncOpenAI',return_value=client):
        provider=OpenAIProvider(api_key='offline-control',flavor=OpenAIProvider.FLAVOR_OPENAI)
        result=[event async for event in provider.stream(model='gpt-6.1-sol',system='s',messages=[{'role':'user','content':'Complete work.'}])]
    assert next(e.response for e in result if isinstance(e,StreamComplete)).content=='Done.'

@pytest.mark.asyncio
async def test_final_text_reconciles_missing_last_delta():
    result=await consume([NS(type='response.output_text.delta',delta='The result'),
                         NS(type='response.completed',response=response(output=[text('The result is 7.')]))])
    complete=next(e.response for e in result if isinstance(e,StreamComplete))
    assert complete.content=='The result is 7.'

@pytest.mark.asyncio
async def test_completed_tool_is_not_executed_twice():
    result=await consume([NS(type='response.output_item.added',output_index=0,item=tool()),
                         NS(type='response.function_call_arguments.done',output_index=0,arguments=tool().arguments),
                         NS(type='response.completed',response=response(output=[tool()]))])
    complete=next(e.response for e in result if isinstance(e,StreamComplete))
    assert len(complete.tool_calls)==1

@pytest.mark.asyncio
async def test_failed_terminal_does_not_return_success():
    with pytest.raises(TransientProviderError):
        await consume([NS(type='response.output_text.delta',delta='Working.'),
                       NS(type='response.failed',response=response('failed',[text('Working.')]))])

@pytest.mark.asyncio
async def test_content_filter_fails_without_transient_retry():
    from anton.core.llm.provider import ContentValidationError
    with pytest.raises(ContentValidationError) as error:
        await consume([NS(type='response.incomplete',response=response('incomplete',[],'content_filter'))])
    assert error.value.code=='content_filter'

@pytest.mark.asyncio
async def test_complete_response_usage_is_retained_with_cached_tokens():
    final=response(output=[text('Done.')]);final.usage=NS(input_tokens=120,output_tokens=8,input_tokens_details=NS(cached_tokens=100))
    result=await consume([NS(type='response.completed',response=final)])
    complete=next(e.response for e in result if isinstance(e,StreamComplete))
    assert complete.usage.input_tokens==20
    assert complete.usage.cache_read_tokens==100
    assert complete.usage.output_tokens==8
