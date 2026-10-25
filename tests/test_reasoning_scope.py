import asyncio


from types import SimpleNamespace


from unittest.mock import AsyncMock,MagicMock,patch


import pytest


from anton.core.llm.reasoning import current_effort,scoped_effort


from anton.core.llm.openai import OpenAIProvider


from anton.core.llm.anthropic import AnthropicProvider


def completion():
    response=MagicMock();response.choices=[MagicMock(message=MagicMock(content='ok',tool_calls=None),finish_reason='stop')]
    response.usage.prompt_tokens=1;response.usage.completion_tokens=1
    return response


async def test_concurrent_calls_keep_their_own_effort_and_restore_the_provider_default():
    with patch('anton.core.llm.openai.openai') as sdk:
        client=AsyncMock();sdk.AsyncOpenAI.return_value=client
        client.chat.completions.create=AsyncMock(return_value=completion())
        provider=OpenAIProvider(api_key='test-key',reasoning_effort='none')
        async def call(effort):
            with scoped_effort(effort):
                await asyncio.sleep(0)
                await provider.complete(model='test-model',system='test',messages=[])
        await asyncio.gather(call('low'),call('medium'))
        assert {call.kwargs['reasoning_effort'] for call in client.chat.completions.create.call_args_list}=={'low','medium'}
        await provider.complete(model='test-model',system='test',messages=[])
        assert client.chat.completions.create.call_args.kwargs['reasoning_effort']=='none'
        assert provider._reasoning_effort=='none'


async def test_stream_uses_the_scoped_effort_in_the_actual_request():
    async def chunks():
        chunk=MagicMock();chunk.usage=None
        delta=MagicMock(content='ok',tool_calls=None);delta.reasoning_content=None
        chunk.choices=[MagicMock(delta=delta,finish_reason='stop')];yield chunk
    with patch('anton.core.llm.openai.openai') as sdk:
        client=AsyncMock();sdk.AsyncOpenAI.return_value=client
        client.chat.completions.create=AsyncMock(return_value=chunks())
        provider=OpenAIProvider(api_key='test-key',reasoning_effort='none')
        with scoped_effort('medium'):
            _=[event async for event in provider.stream(model='test-model',system='test',messages=[])]
        assert client.chat.completions.create.call_args.kwargs['reasoning_effort']=='medium'


def test_responses_payload_uses_nested_scopes_and_none_omits_effort():
    with patch('anton.core.llm.openai.openai'):
        provider=OpenAIProvider(api_key='test-key',flavor=OpenAIProvider.FLAVOR_OPENAI,reasoning_effort='high')
        def kwargs():return provider._build_responses_kwargs(model='test-model',system='',messages=[],tools=None,tool_choice=None,max_tokens=100,native_web_tools=None)
        with scoped_effort('low'):
            assert kwargs()['reasoning']['effort']=='low'
            with scoped_effort(None):assert 'reasoning' not in kwargs()
            assert kwargs()['reasoning']['effort']=='low'
        assert kwargs()['reasoning']['effort']=='high'


async def test_anthropic_payload_uses_the_same_scoped_allocation():
    with patch('anton.core.llm.anthropic.anthropic.AsyncAnthropic') as factory:
        client=AsyncMock();factory.return_value=client
        client.messages.create=AsyncMock(return_value=SimpleNamespace(content=[SimpleNamespace(type='text',text='ok')],usage=SimpleNamespace(input_tokens=1,output_tokens=1),stop_reason='end_turn',model='test-model'))
        provider=AnthropicProvider(api_key='test-key',reasoning_effort='low')
        with scoped_effort('medium'):
            await provider.complete(model='test-model',system='test',messages=[])
        assert client.messages.create.call_args.kwargs['extra_body']=={'output_config':{'effort':'medium'}}
        assert provider._reasoning_effort=='low'


def test_exception_restores_effort_and_invalid_values_do_not_enter_a_scope():
    with pytest.raises(RuntimeError):
        with scoped_effort('medium'):
            raise RuntimeError('source collection failed')
    assert current_effort('default')=='default'
    with pytest.raises(ValueError):
        with scoped_effort(''):pass
    assert current_effort('default')=='default'
