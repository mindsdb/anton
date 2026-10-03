"""Opt-in TypeSafe connection controls, with no network or real credentials."""

import json
from unittest.mock import AsyncMock, patch

import httpx2 as httpx
import pytest

from anton.core.llm import jev
from anton.core.session import _jev_questions
from anton.core.settings import CoreSettings
from tests.test_verifier_jev import _answer, _jev, _run, _session, workspace


def test_direct_key_is_opt_in_and_redacted():
    assert CoreSettings().verifier_jev_direct_api_key is None
    settings = CoreSettings(verifier_jev_direct_api_key="offline-typesafe-control")
    assert "offline-typesafe-control" not in repr(settings)
    assert "offline-typesafe-control" not in settings.model_dump_json()


async def test_direct_typesafe_uses_documented_endpoint_and_retains_usage():
    captured = {}

    def handler(request):
        captured.update(url=str(request.url), auth=request.headers["authorization"], body=json.loads(request.content))
        return _answer()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await jev.classify(
            base_url="https://api.typesafe.ai/v1", api_key="offline-typesafe-control",
            api_path="systemone", model="jev-1.13.0", state={"request": "verify"},
            questions=_jev_questions(), timeout_s=2, client=client,
        )
    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert captured["auth"] == "Bearer offline-typesafe-control"
    assert captured["body"]["questions"] == _jev_questions()
    assert result.status == "COMPLETE"
    assert (result.input_tokens, result.output_tokens) == (1700, 0)


async def test_explicit_direct_key_works_with_byok_without_reusing_openai_key(workspace):
    session, llm = _session(workspace, base_url="https://api.openai.com/v1", direct_api_key="offline-typesafe-control")
    event, classify = await _run(session, _jev("COMPLETE", 0.97))
    call = classify.call_args.kwargs
    assert (call["base_url"], call["api_path"], call["api_key"]) == (
        "https://api.typesafe.ai/v1", "systemone", "offline-typesafe-control",
    )
    llm.coding_provider.current_api_key.assert_not_called()
    llm.generate_object_code.assert_not_called()
    assert event["jev_decided"] == "1"


@pytest.mark.parametrize("result", [
    _jev("COMPLETE", 0.6), _jev("INCOMPLETE", 0.97), _jev("STUCK", 0.97),
    jev.JevVerdict(error="timeout"), jev.JevVerdict(error="http_401"),
    jev.JevVerdict(error="http_429"), jev.JevVerdict(error="http_529"),
    jev.JevVerdict(error="malformed"),
])
async def test_direct_errors_and_unresolved_work_retain_llm_fallback(workspace, result):
    session, llm = _session(workspace, base_url=None, direct_api_key="offline-typesafe-control")
    event, _ = await _run(session, result)
    llm.generate_object_code.assert_called_once()
    assert event["jev_decided"] == "0"


async def test_off_switch_also_disables_direct_connection(workspace):
    session, llm = _session(workspace, jev_setting="off", direct_api_key="offline-typesafe-control")
    _, classify = await _run(session, _jev("COMPLETE", 0.97))
    classify.assert_not_called()
    llm.generate_object_code.assert_called_once()


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": True, "output_tokens": 0}, {"input_tokens": -1, "output_tokens": 0}])
async def test_missing_or_invalid_usage_is_unknown_not_free(usage):
    body = _answer().json()
    body["usage"] = usage
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))) as client:
        result = await jev.classify(
            base_url="https://api.typesafe.ai/v1", api_key="offline", api_path="systemone",
            model="jev-1.13.0", state={}, questions=_jev_questions(), timeout_s=2, client=client,
        )
    assert result.status == "COMPLETE"
    assert result.input_tokens is None


async def test_direct_client_keeps_tls_verification_independent_of_coding_provider(workspace):
    session, _ = _session(workspace, ssl_verify=False, direct_api_key="offline-typesafe-control")
    fake = AsyncMock()
    with patch("anton.core.session.httpx.AsyncClient", return_value=fake) as constructor, patch("anton.core.llm.jev.classify", new=AsyncMock(return_value=_jev("COMPLETE", 0.97))):
        await session._jev_verdict("verify")
        constructor.assert_called_once_with(timeout=2.0, verify=True)
    await session.close()


@pytest.mark.parametrize("distribution", [
    {"COMPLETE": 0.97},
    {"COMPLETE": 0.97, "WAITING": True, "INCOMPLETE": 0.01, "STUCK": 0.01},
    {"COMPLETE": 0.97, "WAITING": -0.01, "INCOMPLETE": 0.02, "STUCK": 0.02},
    {"COMPLETE": 0.97, "WAITING": 0.97, "INCOMPLETE": 0.97, "STUCK": 0.97},
    {"COMPLETE": 0.1, "WAITING": 0.8, "INCOMPLETE": 0.05, "STUCK": 0.05},
])
async def test_bad_distribution_never_supplies_a_completion_verdict(distribution):
    body = _answer().json()
    body["answers"]["status"]["probabilities"] = distribution
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))) as client:
        result = await jev.classify(
            base_url="https://api.typesafe.ai/v1", api_key="offline", api_path="systemone",
            model="jev-1.13.0", state={}, questions=_jev_questions(), timeout_s=2, client=client,
        )
    assert result.status == "" and result.error == "malformed"
    assert result.input_tokens == 1700


@pytest.mark.parametrize("known", [True, False])
async def test_turn_record_keeps_jev_usage_separate_from_coding_tokens(workspace, known):
    session, _ = _session(workspace, base_url=None, direct_api_key="offline-typesafe-control")
    result = jev.JevVerdict(status="COMPLETE", p_status=0.97, p_complete=0.97,
        input_tokens=1700 if known else None, output_tokens=20 if known else None, request_attempted=True)
    event, _ = await _run(session, result)
    assert event["jev_input_tokens"] == ("1700" if known else "0")
    assert event["jev_output_tokens"] == ("20" if known else "0")
    assert event["jev_unknown_usage_calls"] == ("0" if known else "1")
    assert event["jev_request_attempts"] == "1"


async def test_credentials_failure_is_not_a_submitted_billable_call(workspace):
    session, llm = _session(workspace)
    llm.coding_provider.current_api_key = AsyncMock(side_effect=RuntimeError("vault unavailable"))
    event, classify = await _run(session, _jev("COMPLETE", 0.97))
    classify.assert_not_called()
    assert event["jev_request_attempts"] == "0"
    assert event["jev_unknown_usage_calls"] == "0"
