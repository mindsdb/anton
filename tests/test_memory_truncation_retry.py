"""Memory / extraction subsystems survive narrating-model truncation (ENG-1084).

ENG-1081 fixed the completion verifier's token ceiling; the same defect lived
at seven more `generate_object*` call sites — two of them with budgets *tighter*
than the verifier's — and every failure was swallowed. Confirmed live in prod:
identity extraction dying at its 512 cap and consolidation at its 2048 cap on
`mindshub_air`, narrating right up to the budget and never reaching the forced
tool call, with nothing logged.

What must hold now (ticket Done-when):

1. No forced-schema call site asks with a budget inside the measured narration
   range (245–1,654+) without a truncation retry — all sites route through the
   shared `generate_with_truncation_retry` ladder.
2. Prose-at-the-cap through the REAL `LLMClient` is retried with a bigger
   budget and the subsystem produces its result instead of silently degrading.
3. A genuine (non-truncated) failure is not retried — a bigger budget can't
   fix it (fable's `{}` verdicts, ENG-1095) — but it IS logged, content-safe.
4. The no-preamble instruction reaches the provider's system prompt.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.backends.base import Cell
from anton.core.llm.client import LLMClient
from anton.core.llm.provider import LLMResponse, StructuredOutputError, ToolCall, Usage
from anton.core.llm.structured import (
    DEFAULT_STRUCTURED_BUDGETS,
    generate_with_truncation_retry,
    no_preamble_instruction,
)
from anton.core.memory.cerebellum import Cerebellum
from anton.core.memory.consolidator import Consolidator
from anton.core.memory.cortex import Cortex


SECRET = "the user's pasted p@ssw0rd hunter2"


def _prose_at_cap(budget: int) -> LLMResponse:
    """The prod signature: narration fills the whole budget, no tool call,
    and the gateway labels it a normal stop (ENG-1082)."""
    return LLMResponse(
        content="The user is speaking in Turkish. Let me understand… " + SECRET,
        tool_calls=[],
        usage=Usage(input_tokens=100, output_tokens=budget),
        stop_reason="stop",
    )


def _tool_response(name: str, payload: dict) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc_1", name=name, input=payload)],
        usage=Usage(input_tokens=100, output_tokens=60),
        stop_reason="tool_use",
    )


def _client_truncating_once(tool_name: str, payload: dict) -> tuple[LLMClient, AsyncMock]:
    """Real LLMClient over a provider that narrates to the cap on the first
    call, then answers properly on the (bigger-budget) retry."""
    provider = MagicMock()

    async def complete(**kwargs):
        budget = kwargs.get("max_tokens")
        if provider.complete.await_count == 1:
            return _prose_at_cap(budget)
        return _tool_response(tool_name, payload)

    provider.complete = AsyncMock(side_effect=complete)
    client = LLMClient(
        planning_provider=provider,
        planning_model="planner",
        coding_provider=provider,
        coding_model="coder",
    )
    return client, provider.complete


# --------------------------------------------------------------------------
# The shared ladder itself
# --------------------------------------------------------------------------


async def test_ladder_retries_truncation_with_a_bigger_budget():
    client, complete = _client_truncating_once("_Probe", {"value": "ok"})
    from pydantic import BaseModel

    class _Probe(BaseModel):
        value: str

    result = await generate_with_truncation_retry(
        client.generate_object_code, _Probe, system="s",
        messages=[{"role": "user", "content": "m"}],
    )

    assert result.value == "ok"
    budgets = [c.kwargs["max_tokens"] for c in complete.await_args_list]
    assert budgets == list(DEFAULT_STRUCTURED_BUDGETS), (
        "the retry must raise the budget, not re-issue the same call"
    )


async def test_ladder_does_not_retry_a_genuine_failure():
    """fable's `{}` at 8 tokens (ENG-1095): well under budget, so truncation
    is not the diagnosis and a bigger budget buys nothing."""
    provider = MagicMock()
    provider.complete = AsyncMock(
        return_value=LLMResponse(
            content="", tool_calls=[], usage=Usage(input_tokens=10, output_tokens=8),
            stop_reason="stop",
        )
    )
    client = LLMClient(
        planning_provider=provider, planning_model="p",
        coding_provider=provider, coding_model="c",
    )
    from pydantic import BaseModel

    class _Probe(BaseModel):
        value: str

    with pytest.raises(StructuredOutputError) as exc_info:
        await generate_with_truncation_retry(
            client.generate_object_code, _Probe, system="s",
            messages=[{"role": "user", "content": "m"}],
        )

    assert exc_info.value.truncated is False
    assert provider.complete.await_count == 1, "no retry for a non-truncated failure"


async def test_ladder_raises_after_exhausting_every_budget():
    provider = MagicMock()

    async def always_prose(**kwargs):
        return _prose_at_cap(kwargs.get("max_tokens"))

    provider.complete = AsyncMock(side_effect=always_prose)
    client = LLMClient(
        planning_provider=provider, planning_model="p",
        coding_provider=provider, coding_model="c",
    )
    from pydantic import BaseModel

    class _Probe(BaseModel):
        value: str

    with pytest.raises(StructuredOutputError) as exc_info:
        await generate_with_truncation_retry(
            client.generate_object_code, _Probe, system="s",
            messages=[{"role": "user", "content": "m"}],
        )

    assert exc_info.value.truncated is True
    assert provider.complete.await_count == len(DEFAULT_STRUCTURED_BUDGETS)


def test_no_preamble_names_the_tool():
    from anton.core.memory.cortex import _IdentityFacts

    text = no_preamble_instruction(_IdentityFacts)
    assert "_IdentityFacts" in text
    assert "immediately" in text


# --------------------------------------------------------------------------
# Per-subsystem: prose-at-the-cap is rescued, not silently swallowed
# --------------------------------------------------------------------------


def _cortex(client: LLMClient) -> Cortex:
    global_hc = MagicMock()
    project_hc = MagicMock()
    return Cortex(global_hc, project_hc, mode="autopilot", llm_client=client)


async def test_identity_extraction_survives_narration(caplog):
    """The exact prod failure (observation 8e65e78151999148): identity
    extraction narrating to the cap. The retry must rescue it and the facts
    must actually be stored."""
    client, complete = _client_truncating_once(
        "_IdentityFacts", {"facts": ["User's name is Mynda"]}
    )
    cortex = _cortex(client)

    await cortex.maybe_update_identity("hi, I'm Mynda and I work in Excel")

    cortex.global_hc.rewrite_identity.assert_called_once_with(["User's name is Mynda"])
    assert complete.await_count == 2
    system_sent = complete.await_args_list[0].kwargs["system"]
    assert "Call the _IdentityFacts tool immediately" in system_sent


async def test_identity_extraction_failure_is_logged_not_silent(caplog):
    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=RuntimeError(SECRET))
    client = LLMClient(
        planning_provider=provider, planning_model="p",
        coding_provider=provider, coding_model="c",
    )
    cortex = _cortex(client)

    with caplog.at_level(logging.WARNING):
        await cortex.maybe_update_identity("hello")

    cortex.global_hc.rewrite_identity.assert_not_called()
    assert any("identity-extraction failed" in r.message for r in caplog.records), (
        "a dead memory subsystem must be visible in logs (ENG-1084)"
    )
    assert SECRET not in caplog.text, "logs must never quote exception messages"


async def test_cerebellum_diff_survives_narration():
    client, complete = _client_truncating_once(
        "_DiffPassResult",
        {"lessons": [{"text": "progress() before long calls", "topic": "scratchpad"}]},
    )
    cortex_mock = MagicMock()
    cortex_mock.encode = AsyncMock(return_value=[])
    cere = Cerebellum(cortex=cortex_mock, llm=client)
    cere._buffered.append(
        Cell(code="print(1)", stdout="1", stderr="", error=None, description="test")
    )

    lessons = await cere.flush()

    assert [lesson.text for lesson in lessons] == ["progress() before long calls"]
    assert complete.await_count == 2


async def test_cerebellum_failure_log_is_content_safe(caplog):
    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=RuntimeError(SECRET))
    client = LLMClient(
        planning_provider=provider, planning_model="p",
        coding_provider=provider, coding_model="c",
    )
    cere = Cerebellum(cortex=MagicMock(), llm=client)
    cere._buffered.append(
        Cell(code="x", stdout="", stderr="", error=None, description="t")
    )

    with caplog.at_level(logging.WARNING):
        lessons = await cere.flush()

    assert lessons == []
    assert any("cerebellum diff pass failed" in r.message for r in caplog.records)
    assert SECRET not in caplog.text


async def test_consolidation_survives_narration():
    """The other live prod failure (observation 38a951f93924473a):
    consolidation filling its whole 2048 budget with analysis prose."""
    client, complete = _client_truncating_once(
        "_ConsolidatedLessons",
        {"items": [{"text": "cache the API token", "kind": "lesson"}]},
    )
    cells = [Cell(code="print(1)", stdout="1", stderr="", error=None, description="t")]

    engrams = await Consolidator().replay_and_extract(cells, client)

    assert [e.text for e in engrams] == ["cache the API token"]
    assert complete.await_count == 2


async def test_credential_extraction_survives_narration():
    from anton.connect_collector import extract_variables
    from anton.core.datasources.datasource_registry import DatasourceField

    client, complete = _client_truncating_once(
        "_ExtractionResult",
        {"variables": {"host": "db.example.com"}, "is_redirect": False},
    )
    session = MagicMock()
    session._llm = client

    result = await extract_variables(
        "my host is db.example.com",
        expected_fields=[DatasourceField(name="host", secret=False)],
        current_engine="postgres",
        current_engine_display="Postgres",
        known_engine_slugs=["postgres", "mysql"],
        session=session,
    )

    assert result.variables.get("host") == "db.example.com"
    assert complete.await_count == 2


# --------------------------------------------------------------------------
# Which truncation was it (ENG-1523)
# --------------------------------------------------------------------------


def _truncated_mid_json(budget: int) -> LLMResponse:
    """The call started but the budget ran out inside its JSON arguments, so
    `safe_parse_tool_input` salvaged a partial dict and set `parse_error`."""
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="tc_1", name="_Probe", input={},
                parse_error="Unterminated string starting at char 4021",
            )
        ],
        usage=Usage(input_tokens=100, output_tokens=budget),
        stop_reason="length",
    )


def _ladder_over(*responses) -> tuple[LLMClient, AsyncMock]:
    provider = MagicMock()
    calls = iter(responses)

    async def complete(**kwargs):
        return next(calls)(kwargs.get("max_tokens"))

    provider.complete = AsyncMock(side_effect=complete)
    client = LLMClient(
        planning_provider=provider, planning_model="p",
        coding_provider=provider, coding_model="c",
    )
    return client, provider.complete


async def _run_ladder(client, caplog):
    from pydantic import BaseModel

    class _Probe(BaseModel):
        value: str

    with caplog.at_level(logging.WARNING):
        with pytest.raises(StructuredOutputError) as exc_info:
            await generate_with_truncation_retry(
                client.generate_object_code, _Probe, system="s",
                messages=[{"role": "user", "content": "m"}],
                budgets=(256,),
            )
    return exc_info.value


async def test_narration_overflow_is_logged_as_before_call(caplog):
    """No tool call at all — the cure is a bigger budget."""
    client, _ = _ladder_over(_prose_at_cap)
    exc = await _run_ladder(client, caplog)

    assert exc.truncated is True
    assert exc.reached_tool_call is False
    assert "TRUNCATED_BEFORE_CALL" in caplog.text
    assert "TRUNCATED_INSIDE_CALL" not in caplog.text


async def test_payload_overflow_is_logged_as_inside_call(caplog):
    """The call was reached and its arguments were cut off — the cure is a
    smaller response, which a bigger budget only postpones."""
    client, _ = _ladder_over(_truncated_mid_json)
    exc = await _run_ladder(client, caplog)

    assert exc.truncated is True
    assert exc.reached_tool_call is True
    assert "TRUNCATED_INSIDE_CALL" in caplog.text
    assert "TRUNCATED_BEFORE_CALL" not in caplog.text


async def test_the_two_truncations_are_distinguishable_in_the_log(caplog):
    """The whole point: one `TRUNCATED` for both could not say which fix to
    apply, and that ambiguity cost a diagnosis round-trip (ENG-1523)."""
    narrated, _ = _ladder_over(_prose_at_cap)
    payload, _ = _ladder_over(_truncated_mid_json)

    await _run_ladder(narrated, caplog)
    first = caplog.text
    caplog.clear()
    await _run_ladder(payload, caplog)

    assert first != caplog.text


async def test_the_verdict_never_leaks_the_model_text(caplog):
    """The message can quote the conversation, so only the verdict is logged."""
    client, _ = _ladder_over(_prose_at_cap)
    await _run_ladder(client, caplog)

    assert SECRET not in caplog.text
