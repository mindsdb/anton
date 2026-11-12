"""Bulk labelling through Jev: what scratchpad classify() and classify_yes_no() run on."""

from __future__ import annotations

import json

import httpx2 as httpx
import pytest

from anton.core.llm import jev


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _answer(label: str) -> dict:
    probabilities = {name: (1.0 if name == label else 0.0) for name in ("Billing", "Support")}
    return {"type": "choice", "choice": label, "confidence": 1.0, "probabilities": probabilities}


def _label_all(request, seen=None) -> httpx.Response:
    """Answer every question of a batch: Billing when its text mentions an invoice."""
    body = json.loads(request.content)
    if seen is not None:
        seen.append(body)
    answers = {name: _answer("Billing" if "invoice" in q["instructions"] else "Support")
               for name, q in body["questions"].items()}
    return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers,
                                     "usage": {"input_tokens": 10, "output_tokens": 2}})


async def _run(handler, items, *, pacer=None, sleeps=None, **kwargs):
    async def fake_sleep(seconds):
        if sleeps is not None:
            sleeps.append(seconds)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await jev.decide_many(
            items,
            jev.choice_question(["Billing", "Support"], "Which?"),
            base_url="https://api.mindshub.ai/v1",
            api_key="key",
            pacer=pacer or jev.Pacer(6000),
            client=client,
            sleep=fake_sleep,
            **kwargs,
        )


def test_pacer_spaces_starts_and_a_pause_holds_later_ones():
    clock = _Clock()
    pacer = jev.Pacer(270, clock=clock)
    waits = [pacer.reserve() for _ in range(3)]
    assert waits == pytest.approx([0.0, 60 / 270, 2 * 60 / 270])
    pacer.pause(60)
    assert pacer.reserve() == pytest.approx(60.0)


async def test_each_text_rides_in_its_own_question_and_the_state_holds_only_the_task():
    seen = []
    headers = {"x-extra": "1", "Authorization": "spoofed"}

    def handler(request):
        assert request.headers["authorization"] == "Bearer key"
        assert request.headers["x-extra"] == "1"
        return _label_all(request, seen)

    results = await _run(handler, ["an invoice", "a login error", {"title": "late invoice"}], headers=headers)

    assert [jev.choice_result(d)["label"] for d in results] == ["Billing", "Support", "Billing"]
    (body,) = seen
    assert body["model"] == jev.CLASSIFY_MODEL
    assert body["state"] == jev.BATCH_STATE
    assert [q["instructions"] for q in body["questions"].values()] == [
        "Which?\n\nText:\nan invoice", "Which?\n\nText:\na login error", 'Which?\n\nText:\n{"title": "late invoice"}',
    ]


async def test_category_descriptions_go_once_in_the_state():
    seen = []

    async def fake_sleep(seconds):
        pass

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: _label_all(r, seen))) as client:
        results = await jev.decide_many(
            ["an invoice", "a login error"],
            jev.choice_question({"Billing": "invoices and payments", "Support": None}, "Which?"),
            base_url="https://api.mindshub.ai/v1", api_key="key", pacer=jev.Pacer(6000),
            client=client, sleep=fake_sleep,
        )

    assert [jev.choice_result(d)["label"] for d in results] == ["Billing", "Support"]
    (body,) = seen
    assert body["state"].startswith(jev.BATCH_STATE)
    assert "- Billing: invoices and payments" in body["state"]
    assert "Support:" not in body["state"]
    assert all(q["criteria"] == {"Billing": "", "Support": ""} for q in body["questions"].values())


def test_shared_definitions_leaves_other_questions_alone():
    bare = jev.choice_question(["A", "B"], "Q?")
    yes_no = jev.yes_no_question("Q?", yes="y", no="n")
    assert jev.shared_definitions(bare) == (jev.BATCH_STATE, bare)
    assert jev.shared_definitions(yes_no) == (jev.BATCH_STATE, yes_no)


def test_batches_keep_order_and_respect_both_limits():
    questions = [{"instructions": "x" * 300}] * 5
    assert jev.plan_batches(questions, max_tokens=250) == [[0, 1], [2, 3], [4]]
    assert jev.plan_batches(questions, max_items=3) == [[0, 1, 2], [3, 4]]
    # A question over the token budget still goes, alone.
    assert jev.plan_batches([{"instructions": "x" * 3000}], max_tokens=100) == [[0]]


async def test_many_batches_come_back_in_input_order():
    seen = []
    texts = [f"text {i} {'invoice' if i % 3 == 0 else 'login'}" for i in range(1200)]

    results = await _run(lambda r: _label_all(r, seen), texts)

    assert len(seen) == 3  # 1200 texts, at most 500 a request
    assert [jev.choice_result(d)["label"] for d in results] == [
        "Billing" if i % 3 == 0 else "Support" for i in range(1200)
    ]


async def test_a_batch_over_the_context_limit_is_split_until_it_fits():
    sizes = []

    def handler(request):
        body = json.loads(request.content)
        sizes.append(len(body["questions"]))
        if len(body["questions"]) > 2 or any("huge" in q["instructions"] for q in body["questions"].values()):
            return httpx.Response(400, json={"detail": {"error_type": "max_tokens_exceeded"}})
        return _label_all(request)

    results = await _run(handler, ["invoice 1", "login", "invoice 2", "huge", "invoice 3"], concurrency=1)

    assert [jev.choice_result(d).get("label") for d in results] == ["Billing", "Support", "Billing", None, "Billing"]
    assert jev.choice_result(results[3])["error"] == jev.TEXT_TOO_LONG
    assert sizes[0] == 5


async def test_transient_failures_are_retried_and_a_429_pauses_every_start():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": {}})
        if calls["n"] == 2:
            return httpx.Response(429, json={"detail": "You have reached your request limit."})
        return _label_all(request)

    pacer = jev.Pacer(270, clock=_Clock())
    sleeps = []
    results = await _run(handler, ["x"], pacer=pacer, sleeps=sleeps)

    assert jev.choice_result(results[0])["label"] == "Support"
    assert calls["n"] == 3
    # 503: a backoff sleep. 429: the next start waits out the edge's minute.
    assert 1 in sleeps
    assert any(s == pytest.approx(60.0, abs=1) for s in sleeps)


async def test_empty_texts_and_missing_answers_fail_alone():
    def handler(request):
        response = _label_all(request)
        body = json.loads(response.content)
        body["answers"] = {k: v for k, v in body["answers"].items() if k != "t2"}
        return httpx.Response(200, json=body)

    results = await _run(handler, ["fine", "", "no answer", "fine"])

    assert [jev.choice_result(d).get("error") for d in results] == [None, jev.EMPTY_TEXT, "malformed", None]


@pytest.mark.parametrize("status", [401, 402, 403])
async def test_a_refusal_for_the_whole_batch_stops_it(status):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(status, json={"error": {"code": "x"}})

    with pytest.raises(jev.JevUnavailable, match=f"http_{status}"):
        await _run(handler, ["x"] * 500, concurrency=1)
    assert calls["n"] == 1


async def test_a_batch_whose_texts_all_fail_is_abandoned():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(422, json={"error": {"message": "bad criteria"}})

    with pytest.raises(jev.JevUnavailable, match="http_422"):
        await _run(handler, ["x"] * 500, concurrency=1)
    assert calls["n"] == 1


async def test_progress_is_reported_per_finished_batch():
    reported = []

    await _run(_label_all, ["a"] * 600, on_progress=lambda done, total: reported.append((done, total)),
               concurrency=1)

    assert reported == [(500, 600), (600, 600)]


def test_choice_question_takes_a_list_or_described_categories():
    assert jev.choice_question(["A", "B"], "Q?") == {
        "type": "choice", "instructions": "Q?", "criteria": {"A": "", "B": ""},
    }
    assert jev.choice_question({"A": "about a", "B": None}, "Q?")["criteria"] == {"A": "about a", "B": ""}
    for bad in (["A"], ["A", "A"], ["A", " "], "AB"):
        with pytest.raises((ValueError, TypeError)):
            jev.choice_question(bad, "Q?")


def test_yes_no_question_and_its_result():
    assert jev.yes_no_question("Billing?") == {"type": "noul", "instructions": "Billing?"}
    assert jev.yes_no_question("Billing?", yes="about payments")["criteria"] == {"true": "about payments", "false": None}
    with pytest.raises(ValueError):
        jev.yes_no_question(" ")

    ok = jev.JevDecision(answers={"answer": {"type": "noul", "noul": 0.8}})
    assert jev.yes_no_result(ok) == {"yes": True, "p_yes": 0.8}
    assert jev.yes_no_result(jev.JevDecision(answers={}, error="timeout"))["error"] == "timeout"
    assert jev.yes_no_result(jev.JevDecision(answers={"answer": {"noul": True}}))["error"] == "malformed"


def test_detach_conversation_moves_the_session_into_metadata():
    headers = {"Langfuse-Session-Id": "conv", "Langfuse-Metadata": json.dumps({"turn": "1"}), "X": "y"}
    out = jev.detach_conversation(headers)
    assert "Langfuse-Session-Id" not in out
    assert json.loads(out["Langfuse-Metadata"]) == {"turn": "1", "conversation_id": "conv"}
    assert headers["Langfuse-Session-Id"] == "conv"  # the caller's dict is untouched
