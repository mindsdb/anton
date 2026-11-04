"""`jev.decide`: every named answer of a question set, failing open.

`classify` reads only the verifier's `status`; Chimera's sidekick checks ask
four yes/no questions at once and need all of them.
"""

from __future__ import annotations

import asyncio

import httpx2 as httpx
import pytest

from anton.core.chimera.jev_questions import sidekick_questions
from anton.core.llm import jev


def _body(**answers):
    return {
        "model": "jev-1.13.0",
        "answers": {name: {"type": "noul", "noul": p} for name, p in answers.items()},
        "usage": {"input_tokens": 1200, "output_tokens": 0},
    }


async def _decide(handler, questions=None, timeout_s=2.0):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await jev.decide(base_url="https://api.example/v1", api_key="k", model="jev-1.13.0",
                                state={"request": "r", "transcript": "t"},
                                questions=questions or sidekick_questions(),
                                timeout_s=timeout_s, client=client)


def test_sidekick_questions_are_four_yes_no_questions():
    questions = sidekick_questions()
    assert list(questions) == ["take_back", "stuck", "off_scope", "done"]
    assert {q["type"] for q in questions.values()} == {"noul"}


async def test_decide_reads_every_answer_and_sends_the_questions():
    seen = {}

    def handler(request):
        import json

        seen["url"] = str(request.url)
        seen["questions"] = list(json.loads(request.content)["questions"])
        return httpx.Response(200, json=_body(take_back=0.91, stuck=0.2, off_scope=0.05, done=0.4))

    decision = await _decide(handler)
    assert decision.error == ""
    assert {name: decision.p(name) for name in decision.answers} == {
        "take_back": 0.91, "stuck": 0.2, "off_scope": 0.05, "done": 0.4,
    }
    assert decision.answers["take_back"].type == "noul"
    assert (decision.model, decision.input_tokens, decision.output_tokens) == ("jev-1.13.0", 1200, 0)
    assert seen == {"url": "https://api.example/v1/decisions",
                    "questions": ["take_back", "stuck", "off_scope", "done"]}


async def test_decide_reads_choice_answers_too():
    body = {"answers": {"status": {"type": "choice", "choice": "STUCK",
                                   "probabilities": {"COMPLETE": 0.1, "STUCK": 0.9}}}}
    decision = await _decide(lambda r: httpx.Response(200, json=body),
                             questions={"status": {"type": "choice", "criteria": {}}})
    answer = decision.answers["status"]
    assert (answer.type, answer.choice, answer.p) == ("choice", "STUCK", 0.9)
    assert answer.probabilities == {"COMPLETE": 0.1, "STUCK": 0.9}


@pytest.mark.parametrize(
    "response,error",
    [
        (httpx.Response(503, json={}), "http_503"),
        (httpx.Response(200, content=b"not json"), "malformed"),
        # One question unanswered is no answer at all.
        (httpx.Response(200, json=_body(take_back=0.9, stuck=0.1, off_scope=0.1)), "malformed"),
        (httpx.Response(200, json=_body(take_back=1.5, stuck=0.1, off_scope=0.1, done=0.1)), "malformed"),
        (httpx.Response(200, json=_body(take_back=True, stuck=0.1, off_scope=0.1, done=0.1)), "malformed"),
    ],
)
async def test_failures_come_back_as_an_error_with_no_answers(response, error):
    decision = await _decide(lambda r: response)
    assert decision.error == error
    assert decision.answers == {}
    assert decision.p("take_back") is None


async def test_an_unusable_answer_keeps_its_usage():
    decision = await _decide(lambda r: httpx.Response(200, json={"answers": {}, "usage": {
        "input_tokens": 900, "output_tokens": 1}}))
    assert (decision.error, decision.input_tokens, decision.output_tokens) == ("malformed", 900, 1)


async def test_decide_fails_open_on_its_time_budget():
    async def slow(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json=_body(take_back=1, stuck=1, off_scope=1, done=1))

    decision = await _decide(slow, timeout_s=0.05)
    assert decision.error == "timeout"
    assert decision.answers == {}


async def test_decide_reports_transport_failures():
    def broken(request):
        raise httpx.ConnectError("down")

    decision = await _decide(broken)
    assert decision.error == "transport"
