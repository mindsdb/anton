import importlib
import json

import pytest

import minds.requests.stream as mod


# --- module fixture: neutralize langfuse.observe and reload SUT ---
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
    async for chunk in streaming_mod.format_messages_for_streaming(gen(), model):
        out.append(chunk)

    # Should produce two SSE "data: <json>\n\n" lines
    assert all(ch.startswith("data: ") and ch.endswith("\n\n") for ch in out)
    payload0 = json.loads(out[0][len("data: ") :].strip())
    payload1 = json.loads(out[1][len("data: ") :].strip())

    # Basic schema checks
    assert payload0["id"] == "chatcmpl-A"
    assert payload1["id"] == "chatcmpl-B"
    assert payload0["model"] == model
    assert payload0["choices"][0]["index"] == 0
    assert payload1["choices"][0]["index"] == 1
    assert payload0["choices"][0]["delta"]["role"] == streaming_mod.Role.user
    assert payload0["choices"][0]["delta"]["content"] == "hello"
    assert payload1["choices"][0]["delta"]["role"] == streaming_mod.Role.assistant
    assert payload1["choices"][0]["delta"]["content"] == "test"


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
    trace_name = "trace_name_test_process_streaming_producer_emits_sse"

    async def producer(streamer: streaming_mod.MessageStreamer):
        await streamer.push(streaming_mod.Role.user, "hello")
        await streamer.push(streaming_mod.Role.assistant, "world")
        # close() is guaranteed by the wrapper even if we forget it

    resp = await streaming_mod.process_streaming_producer(
        producer,
        request_id="chatcmpl-777",
        model=model,
        trace_name=trace_name,
    )
    # headers & type
    assert resp.media_type == "text/event-stream"
    assert resp.headers["Cache-Control"] == "no-cache"
    assert resp.headers["Connection"] == "keep-alive"

    # Collect a few chunks from the async body iterator
    chunks = []
    async for b in resp.body_iterator:
        s = b.decode() if isinstance(b, bytes | bytearray) else b
        chunks.append(s)
        if len(chunks) >= 2:
            break

    # Two SSE lines for two pushes
    assert all(ch.startswith("data: ") and ch.endswith("\n\n") for ch in chunks)
    payloads = [json.loads(ch[len("data: ") :].strip()) for ch in chunks]
    assert [p["choices"][0]["delta"]["content"] for p in payloads] == [
        "hello",
        "world",
    ]
    assert all(p["id"].startswith("chatcmpl-777") or p["id"] == "chatcmpl-777" for p in payloads)


@pytest.mark.asyncio
async def test_process_non_streaming_producer_returns_json(streaming_mod):
    model = "model_test_process_non_streaming_producer_returns_json"

    async def producer(collector: streaming_mod.MessageStreamer):
        await collector.push(streaming_mod.Role.user, "u")
        await collector.push(streaming_mod.Role.assistant, "a")

    resp = await streaming_mod.process_non_streaming_producer(producer, request_id="chatcmpl-42", model=model)
    assert resp.status_code == 200
    data = json.loads(resp.body.decode())
    assert data["model"] == model
    assert len(data["choices"]) == 2
    assert data["choices"][0]["message"]["content"] == "u"
    assert data["choices"][1]["message"]["content"] == "a"
