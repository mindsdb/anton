import importlib
import json

import pytest

import minds.requests.stream as mod


@pytest.fixture()
def streaming_mod(monkeypatch):
    # Make observe a no-op BEFORE importing/reloading the module
    import langfuse as dec

    def mock_observe(f=None, **decorator_kwargs):
        """Mock observe decorator that consumes langfuse parameters"""
        if f is None:
            # Called as @observe(...) - return decorator
            return lambda func: mock_observe(func, **decorator_kwargs)
        else:
            # Called as @observe or @observe() - return wrapped function
            def wrapper(*args, **kwargs):
                # Filter out langfuse parameters that should be consumed by decorator
                filtered_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("langfuse_")}
                return f(*args, **filtered_kwargs)

            return wrapper

    monkeypatch.setattr(dec, "observe", mock_observe)

    importlib.reload(mod)
    return mod


# ---------- basic message creation ----------


def test_create_stream_message_with_custom_id(streaming_mod):
    sm = streaming_mod.create_stream_message(role=streaming_mod.Role.assistant, content="hi", request_id="chatcmpl-123")
    assert isinstance(sm, streaming_mod.StreamMessage)
    assert sm.id == "chatcmpl-123"
    assert sm.role == streaming_mod.Role.assistant
    assert sm.content == "hi"


def test_stream_message_default_id_prefix(streaming_mod):
    sm = streaming_mod.StreamMessage(role=streaming_mod.Role.user, content="x")
    assert sm.id.startswith("chatcmpl-")
    assert sm.role == streaming_mod.Role.user
    assert sm.content == "x"


# ---------- format_messages_for_streaming (SSE) ----------


@pytest.mark.asyncio
async def test_format_messages_for_streaming_emits_sse(streaming_mod):
    model = "model_test_format_messages_for_streaming_emits_sse"

    async def gen():
        yield streaming_mod.StreamMessage(id="chatcmpl-A", role=streaming_mod.Role.user, content="hello")
        yield streaming_mod.StreamMessage(id="chatcmpl-B", role=streaming_mod.Role.assistant, content="test")

    out = []
    async for chunk in streaming_mod.format_messages_for_streaming_chat_completions_api(gen(), model):
        out.append(chunk)

    # Should produce two content SSE events followed by a final "stop" event
    assert all(ch.startswith("event: completion\ndata: ") and ch.endswith("\n\n") for ch in out)
    # Expect three chunks: two content chunks and a final stop chunk
    assert len(out) == 3
    payload0 = json.loads(out[0][len("event: completion\ndata: ") :].strip())
    payload1 = json.loads(out[1][len("event: completion\ndata: ") :].strip())
    payload2 = json.loads(out[2][len("event: completion\ndata: ") :].strip())

    # Basic schema checks for content chunks
    assert payload0["id"] == "chatcmpl-A"
    assert payload1["id"] == "chatcmpl-B"
    assert payload0["model"] == model
    assert payload0["choices"][0]["index"] == 0
    assert payload1["choices"][0]["index"] == 1
    assert payload0["choices"][0]["delta"]["role"] == streaming_mod.Role.user
    assert payload0["choices"][0]["delta"]["content"] == "hello"
    assert payload1["choices"][0]["delta"]["role"] == streaming_mod.Role.assistant
    assert payload1["choices"][0]["delta"]["content"] == "test"
    # The final chunk should be the stop marker
    assert payload0["choices"][0].get("finish_reason") is None
    assert payload1["choices"][0].get("finish_reason") is None
    assert payload2["choices"][0]["finish_reason"] == "stop"
    assert payload2["choices"][0]["delta"]["content"] == ""


# ---------- Streamer (queue-based) ----------


@pytest.mark.asyncio
async def test_streamer_push_iterate_and_close(streaming_mod):
    s = streaming_mod.Streamer(request_id="chatcmpl-xyz")
    await s.push(streaming_mod.Role.user, "hi")
    await s.push(streaming_mod.Role.assistant, "there")
    await s.close()

    seen = []
    async for m in s:
        seen.append(m)

    assert [m.role for m in seen] == [
        streaming_mod.Role.user,
        streaming_mod.Role.assistant,
    ]
    assert [m.content for m in seen] == ["hi", "there"]
    assert all(m.id == "chatcmpl-xyz" for m in seen)  # propagated request_id


# ---------- StreamerCollector ----------


@pytest.mark.asyncio
async def test_streamer_collector_accumulates_and_close_noop(streaming_mod):
    c = streaming_mod.StreamerCollector(request_id="chatcmpl-abc")
    await c.push(streaming_mod.Role.user, "one")
    await c.push(streaming_mod.Role.assistant, "two")
    assert len(c.messages) == 2
    assert c.messages[0].id == "chatcmpl-abc"
    await c.close()  # should be a no-op


# ---------- process_streaming_producer / process_non_streaming_producer ----------


@pytest.mark.asyncio
async def test_process_streaming_producer_emits_sse(streaming_mod):
    model = "model_test_process_streaming_producer_emits_sse"

    # type: ignore - avoid module attribute in type annotation in test-local function
    async def producer(streamer):
        await streamer.push(streaming_mod.Role.user, "hello")
        await streamer.push(streaming_mod.Role.assistant, "world")
        # close() is guaranteed by the wrapper even if we forget it

    resp = await streaming_mod.process_streaming_producer(
        producer,
        request_id="chatcmpl-777",
        format_func=streaming_mod.format_messages_for_streaming_chat_completions_api,
        model=model,
    )
    # headers & type
    assert resp.media_type == "text/event-stream"
    assert resp.headers["Cache-Control"] == "no-cache"
    assert resp.headers["Connection"] == "keep-alive"

    # Collect chunks until we see the final 'stop' chunk
    chunks = []
    found_stop = False
    async for b in resp.body_iterator:
        s = b.decode() if isinstance(b, bytes | bytearray) else b
        chunks.append(s)
        payload = json.loads(s[len("event: completion\ndata: ") :].strip())
        if payload["choices"][0].get("finish_reason") == "stop":
            found_stop = True
            break

    assert found_stop, "Did not find final stop chunk in stream"
    assert all(ch.startswith("event: completion\ndata: ") and ch.endswith("\n\n") for ch in chunks)
    payloads = [json.loads(ch[len("event: completion\ndata: ") :].strip()) for ch in chunks]
    assert [p["choices"][0]["delta"]["content"] for p in payloads if p["choices"][0].get("finish_reason") is None] == [
        "hello",
        "world",
    ]
    assert all(p["id"].startswith("chatcmpl-777") or p["id"] == "chatcmpl-777" for p in payloads)
    # The final streamed choice should include finish_reason == 'stop'
    assert payloads[0]["choices"][0].get("finish_reason") is None
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_process_non_streaming_producer_returns_json(streaming_mod):
    model = "model_test_process_non_streaming_producer_returns_json"

    # type: ignore - avoid module attribute in type annotation in test-local function
    async def producer(collector):
        await collector.push(streaming_mod.Role.user, "u")
        await collector.push(streaming_mod.Role.assistant, "a")

    resp = await streaming_mod.process_non_streaming_producer(
        producer,
        request_id="chatcmpl-42",
        format_func=streaming_mod.format_messages_for_non_streaming_chat_completions_api,
        model=model,
    )
    assert resp.status_code == 200
    data = json.loads(resp.body.decode())
    assert data["model"] == model
    assert len(data["choices"]) == 2
    assert data["choices"][0]["message"]["content"] == "u"
    assert data["choices"][1]["message"]["content"] == "a"
    # Non-streaming responses should include finish_reason == 'stop' for each choice
    assert all(c.get("finish_reason") == "stop" for c in data["choices"])


@pytest.mark.asyncio
async def test_process_non_streaming_filters_out_system_messages(streaming_mod):
    """Non-streaming responses should exclude messages with role==Role.system."""
    model = "model_test_non_streaming_filters_system"

    async def producer(collector):
        await collector.push(streaming_mod.Role.user, "u")
        await collector.push(streaming_mod.Role.system, "thought")
        await collector.push(streaming_mod.Role.assistant, "a")

    resp = await streaming_mod.process_non_streaming_producer(
        producer,
        request_id="chatcmpl-sys",
        format_func=streaming_mod.format_messages_for_non_streaming_chat_completions_api,
        model=model,
    )
    assert resp.status_code == 200
    data = json.loads(resp.body.decode())
    assert data["model"] == model
    # The system/thought message should be filtered out, leaving only user + assistant
    assert len(data["choices"]) == 2
    assert [c["message"]["content"] for c in data["choices"]] == ["u", "a"]
    # Ensure the system content is not present anywhere in the choices
    assert all("thought" not in c["message"]["content"] for c in data["choices"])


@pytest.mark.asyncio
async def test_process_streaming_producer_includes_system_messages(streaming_mod):
    """Streaming responses should include system-role messages (they are not filtered when streaming)."""
    model = "model_test_streaming_includes_system"

    async def producer(streamer):
        await streamer.push(streaming_mod.Role.user, "hello")
        await streamer.push(streaming_mod.Role.system, "internal_thought")
        await streamer.push(streaming_mod.Role.assistant, "world")

    resp = await streaming_mod.process_streaming_producer(
        producer,
        request_id="chatcmpl-stream-sys",
        format_func=streaming_mod.format_messages_for_streaming_chat_completions_api,
        model=model,
    )

    found_system = False
    found_stop = False
    async for b in resp.body_iterator:
        s = b.decode() if isinstance(b, bytes | bytearray) else b
        payload = json.loads(s[len("event: completion\ndata: ") :].strip())
        delta = payload["choices"][0]["delta"]
        # Check for the system message content emitted in the stream
        if delta.get("content") == "internal_thought":
            found_system = True
        if payload["choices"][0].get("finish_reason") == "stop":
            found_stop = True
            break

    assert found_system, "Expected system message to be present in streaming output"
    assert found_stop, "Did not find final stop chunk in stream"
