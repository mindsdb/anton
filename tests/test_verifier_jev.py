"""Jev as the completion verifier on MindsHub, with the LLM verifier as fallback.

Jev settles only a confident COMPLETE or WAITING. Everything else, every Jev
failure, and every non-MindsHub (BYOK) route uses the LLM verifier as before.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2 as httpx
import pytest

from tests.conftest import make_mock_llm

from anton.core.llm import jev
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import LLMResponse, ProviderConnectionInfo, StreamComplete, ToolCall, Usage
from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict, _jev_questions
from anton.core.settings import CoreSettings


@pytest.fixture()
def workspace():
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _answer(choice="COMPLETE", p_complete=0.97, status_code=200, body=None):
    payload = body if body is not None else {
        "model": "jev-1.13.0",
        "answers": {
            "status": {"type": "choice", "choice": choice, "confidence": 0.9,
                       "probabilities": {"COMPLETE": p_complete, "WAITING": 0.01,
                                         "INCOMPLETE": 0.01, "STUCK": 0.01}},
            "close_to_done": {"type": "noul", "noul": 0.1},
        },
        "usage": {"input_tokens": 1700, "output_tokens": 0},
    }
    return httpx.Response(status_code, json=payload)


async def _classify(handler, timeout_s=2.0):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await jev.classify(base_url="https://api.example/v1", api_key="k", model="jev-1.13.0",
                                  state={"request": "r", "transcript": "t"}, questions=_jev_questions(),
                                  timeout_s=timeout_s, client=client)


def test_questions_carry_the_production_rubric():
    q = _jev_questions()
    assert list(q["status"]["criteria"]) == ["COMPLETE", "WAITING", "INCOMPLETE", "STUCK"]
    # The criteria are the verifier's own bullets, not a second copy.
    assert q["status"]["criteria"]["STUCK"] in _VerifierVerdict.model_fields["status"].description


def test_questions_refuse_a_rubric_without_all_four_bullets():
    with pytest.raises(ValueError):
        jev.build_questions("Classify:\n- COMPLETE: done\n- WAITING: asks\n", "rubric", "small")


async def test_classify_reads_the_verdict_and_sends_the_request():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        return _answer("INCOMPLETE", p_complete=0.12)

    result = await _classify(handler)
    assert (result.status, result.p_status, result.p_complete, result.error) == ("INCOMPLETE", 0.01, 0.12, "")
    assert seen == {"url": "https://api.example/v1/decisions", "auth": "Bearer k"}


@pytest.mark.parametrize(
    "response,error",
    [
        (_answer(status_code=529), "http_529"),
        (_answer(choice="DONE"), "malformed"),
        (_answer(p_complete=1.5), "malformed"),
        (_answer(p_complete=True), "malformed"),
        (_answer(body={"answers": {}}), "malformed"),
    ],
)
async def test_classify_turns_bad_responses_into_an_error_class(response, error):
    result = await _classify(lambda request: response)
    assert (result.status, result.error) == ("", error)


async def test_classify_reports_transport_failures():
    def handler(request):
        raise httpx.ConnectError("refused")

    assert (await _classify(handler)).error == "transport"


async def test_classify_stops_at_its_wall_clock_bound():
    async def handler(request):
        await asyncio.sleep(5)
        return _answer()

    started = time.monotonic()
    result = await _classify(handler, timeout_s=0.2)
    assert result.error == "timeout"
    assert time.monotonic() - started < 2


async def test_current_api_key_prefers_the_live_supplier():
    refreshing = OpenAIProvider(api_key="stale", base_url="https://api.mindshub.ai/v1",
                                api_key_provider=AsyncMock(return_value="fresh"))
    static = OpenAIProvider(api_key="k", base_url="https://api.mindshub.ai/v1")
    try:
        assert await refreshing.current_api_key() == "fresh"
        assert await static.current_api_key() == "k"
    finally:
        await refreshing.aclose()
        await static.aclose()


class _Stream:
    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _session(workspace, *, jev_setting="on", base_url="https://api.mindshub.ai/v1", llm_status="COMPLETE", ssl_verify=None):
    from anton.core.tools.registry import ToolOutcome

    llm = make_mock_llm()
    llm.generate_object_code = AsyncMock(return_value=_VerifierVerdict(status=llm_status, reason="llm reason"))
    llm.coding_provider.export_connection_info = MagicMock(
        return_value=ProviderConnectionInfo(provider="openai", api_key="k", base_url=base_url, ssl_verify=ssl_verify)
    )
    llm.coding_provider.current_api_key = AsyncMock(return_value="k")
    calls = iter([
        LLMResponse(content="Running.", tool_calls=[ToolCall(id="t1", name="scratchpad", input={"action": "exec", "name": "m", "code": "print(1)"})],
                    usage=Usage(input_tokens=10, output_tokens=5), stop_reason="tool_use"),
        LLMResponse(content="Done: 1.", tool_calls=[], usage=Usage(input_tokens=10, output_tokens=5), stop_reason="end_turn"),
    ])
    llm.plan_stream = lambda **kwargs: _Stream([StreamComplete(response=next(calls))])
    session = ChatSession(ChatSessionConfig(
        llm_client=llm, workspace=workspace, session_id="conv-jev",
        settings=CoreSettings(verifier_jev=jev_setting),
    ))
    session.tool_registry.dispatch_tool = AsyncMock(return_value=ToolOutcome(content="1", ok=True))
    return session, llm


async def _run(session, jev_result=None):
    classify = AsyncMock(return_value=jev_result)
    with patch("anton.core.llm.jev.classify", new=classify), patch("anton.analytics.send_event") as send:
        async for _ in session.turn_stream("run it"):
            pass
    await session.close()
    return send.call_args.kwargs, classify


def _jev(status, p, p_complete=None):
    return jev.JevVerdict(status=status, p_status=p, p_complete=p if p_complete is None else p_complete, ms=700)


def test_on_by_default():
    assert CoreSettings().verifier_jev == "on"


@pytest.mark.parametrize("status", ["COMPLETE", "WAITING"])
async def test_a_confident_jev_verdict_skips_the_llm(workspace, status):
    session, llm = _session(workspace)
    event, _ = await _run(session, _jev(status, 0.97))
    llm.generate_object_code.assert_not_called()
    assert event["ended_by"] == "completed"
    assert (event["jev_checks"], event["jev_decided"], event["jev_last_status"]) == ("1", "1", status)


@pytest.mark.parametrize(
    "result",
    [
        _jev("COMPLETE", 0.6),  # below the threshold
        _jev("INCOMPLETE", 0.95, p_complete=0.02),  # needs the LLM's reason
        _jev("STUCK", 0.95, p_complete=0.01),
        jev.JevVerdict(error="timeout", ms=2000),
    ],
)
async def test_everything_else_falls_back_to_the_llm(workspace, result):
    session, llm = _session(workspace)
    event, _ = await _run(session, result)
    llm.generate_object_code.assert_called_once()  # the LLM verdict (COMPLETE) decided
    assert event["ended_by"] == "completed"
    assert (event["jev_checks"], event["jev_decided"]) == ("1", "0")


async def test_a_fallback_records_when_jev_and_the_llm_disagree(workspace):
    session, llm = _session(workspace)
    event, _ = await _run(session, _jev("INCOMPLETE", 0.9, p_complete=0.05))
    assert (event["jev_disagreements"], event["jev_errors"]) == ("1", "0")


@pytest.mark.parametrize("base_url", ["https://api.openai.com/v1", "https://mindshub.ai.attacker.example/v1", None])
async def test_byok_and_other_routes_never_ask_jev(workspace, base_url):
    session, llm = _session(workspace, base_url=base_url)
    event, classify = await _run(session, _jev("COMPLETE", 0.99))
    classify.assert_not_called()
    llm.generate_object_code.assert_called_once()
    assert "jev_checks" not in event


async def test_the_off_switch_restores_the_llm_verifier(workspace):
    session, llm = _session(workspace, jev_setting="off")
    event, classify = await _run(session, _jev("COMPLETE", 0.99))
    classify.assert_not_called()
    llm.generate_object_code.assert_called_once()
    assert "jev_checks" not in event


async def test_a_credentials_failure_falls_back(workspace):
    session, llm = _session(workspace)
    llm.coding_provider.current_api_key = AsyncMock(side_effect=RuntimeError("vault down"))
    event, _ = await _run(session, _jev("COMPLETE", 0.99))
    llm.generate_object_code.assert_called_once()
    assert (event["jev_errors"], event["jev_last_status"]) == ("1", "credentials")


async def test_a_jev_verdict_leaves_the_llm_latch_counters_alone(workspace):
    session, llm = _session(workspace)
    session._verifier_latch.no_verdict_failures = 1
    await _run(session, _jev("COMPLETE", 0.99))
    assert session._verifier_latch.no_verdict_failures == 1


async def test_the_llm_re_probe_is_never_skipped_by_jev(workspace):
    # A latched session re-probing the LLM must get an LLM verdict, or the latch never clears.
    session, llm = _session(workspace)
    session._verifier_latch.latched = True
    session._verifier_latch.reason = session._verifier_latch.last_no_verdict = "hard"
    session._verifier_latch.skips = 9  # the next check is the re-probe
    event, classify = await _run(session, _jev("COMPLETE", 0.99))
    classify.assert_not_called()
    llm.generate_object_code.assert_called_once()
    assert session._verifier_latch.latched is False


async def test_the_client_follows_ssl_verify_and_is_reused(workspace):
    session, llm = _session(workspace, ssl_verify=False)
    client = MagicMock(aclose=AsyncMock())
    with patch("anton.core.session.httpx.AsyncClient", return_value=client) as make_client:
        event, classify = await _run(session, _jev("COMPLETE", 0.99))
    make_client.assert_called_once()
    assert make_client.call_args.kwargs["verify"] is False
    assert classify.call_args.kwargs["client"] is client
    client.aclose.assert_awaited_once()
