"""Chimera's plain parts: the sidekick's saved state, its report, and the policy."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from anton.core.chimera.config import ChimeraConfig
from anton.core.chimera.policy import ChimeraPolicy, Verdict
from anton.core.chimera.report import SidekickReport, str_list
from anton.core.chimera.state import REUSE_WINDOW, SidekickState
from anton.core.llm.jev import JevAnswer, JevDecision

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


def _state(**over) -> SidekickState:
    fields = dict(
        sidekick_id="sk1", model="gpt-sol",
        messages=[{"role": "user", "content": "do it"}, {"role": "assistant", "content": "done"}],
        summary="Pulled the parts table.", status="open",
        last_call_at=(NOW - timedelta(minutes=2)).isoformat(), signals=1,
    )
    fields.update(over)
    return SidekickState(**fields)


# -- state ----------------------------------------------------------------------


def test_state_round_trips_through_json():
    state = _state()
    data = json.loads(json.dumps(state.to_dict()))
    assert data == {
        "sidekick_id": "sk1", "model": "gpt-sol", "messages": state.messages,
        "summary": "Pulled the parts table.", "status": "open",
        "last_call_at": state.last_call_at, "signals": 1,
    }
    assert SidekickState.from_dict(data) == state


@pytest.mark.parametrize("data", [None, "x", [], {}, {"sidekick_id": ""}, {"sidekick_id": 3}])
def test_anything_but_a_saved_state_reads_as_none(data):
    assert SidekickState.from_dict(data) is None


def test_a_damaged_row_reads_with_defaults():
    state = SidekickState.from_dict({
        "sidekick_id": "sk1", "messages": "nope", "status": "weird", "signals": True, "last_call_at": 5,
    })
    assert (state.messages, state.status, state.signals, state.last_call_at, state.model) == ([], "idle", 0, None, "")


def test_messages_are_reused_under_five_minutes():
    state = _state(last_call_at=(NOW - REUSE_WINDOW + timedelta(seconds=1)).isoformat())
    assert state.reusable(NOW)
    assert state.resume(NOW) == state.messages


@pytest.mark.parametrize("age", [REUSE_WINDOW, timedelta(hours=3)])
def test_older_messages_restart_from_the_summary(age):
    state = _state(last_call_at=(NOW - age).isoformat())
    assert not state.reusable(NOW)
    assert state.resume(NOW) == []
    assert state.messages == []
    assert state.summary == "Pulled the parts table."


def test_no_last_call_or_no_messages_is_never_reused():
    assert not _state(last_call_at=None).reusable(NOW)
    assert not _state(messages=[]).reusable(NOW)
    assert not _state(last_call_at="yesterday").reusable(NOW)


def test_a_naive_timestamp_reads_as_utc():
    assert _state(last_call_at=(NOW - timedelta(minutes=1)).replace(tzinfo=None).isoformat()).reusable(NOW)


@pytest.mark.parametrize(
    "report_status,state_status",
    [("done", "idle"), ("failed", "idle"), ("needs_main", "open"), ("budget_exhausted", "open")],
)
def test_record_opens_the_state_only_for_work_left_to_the_main_agent(report_status, state_status):
    state = SidekickState.new("gpt-sol")
    state.record(messages=[{"role": "user", "content": "x"}], summary="s", report_status=report_status,
                 model="gpt-sol", now=NOW)
    assert (state.status, state.summary, state.last_call_at) == (state_status, "s", NOW.isoformat())


# -- report ---------------------------------------------------------------------


def test_report_content_is_compact_json_with_what_to_do_next():
    content = SidekickReport(status="done", summary="Counted 12 rows.", outputs=["rows.csv"],
                             evidence=["len(df) == 12"], rounds=3, tokens=900).to_tool_content()
    data = json.loads(content)
    assert data["status"] == "done"
    assert data["outputs"] == ["rows.csv"]
    assert "open_questions" not in data and "cost_usd" not in data
    assert "Review" in data["next"]
    assert content == json.dumps(data, ensure_ascii=False, separators=(",", ":"))  # compact


def test_str_list_cleans_model_supplied_lists():
    assert str_list("one") == ["one"]
    assert str_list(["a", " ", None, 3, {"x": 1}]) == ["a", "3"]
    assert str_list(None) == []
    assert len(str_list(["x"] * 50)) == 20
    assert str_list(["y" * 5000]) == ["y" * 2000]


# -- policy ---------------------------------------------------------------------


def _decision(**ps) -> JevDecision:
    return JevDecision(answers={name: JevAnswer(type="noul", p=p) for name, p in ps.items()}, model="jev")


def _policy(decision=None, *, raises=False, **config) -> tuple[ChimeraPolicy, list]:
    calls: list = []

    async def decide(state, questions, metadata):
        calls.append((state, questions, metadata))
        if raises:
            raise RuntimeError("down")
        return decision

    return ChimeraPolicy(ChimeraConfig(enabled=True, **config), decide=decide), calls


@pytest.mark.parametrize(
    "round_idx,errors,finishing,expected",
    [
        (0, 0, False, None),
        (3, 0, False, "every"),
        (7, 0, False, "every"),
        (1, 2, False, "errors"),
        (1, 0, True, "finish"),
        (3, 2, True, "finish"),
    ],
)
def test_when_the_sidekick_is_checked(round_idx, errors, finishing, expected):
    policy, _ = _policy()
    assert policy.trigger(round_idx=round_idx, error_streak=errors, finishing=finishing) == expected


def test_jev_every_zero_checks_only_on_errors_and_finish():
    policy, _ = _policy(jev_every=0)
    assert policy.trigger(round_idx=3, error_streak=0, finishing=False) is None


async def test_answers_count_at_or_above_the_threshold():
    policy, calls = _policy(_decision(take_back=0.8, stuck=0.79, off_scope=0.95, done=0.1), jev_threshold=0.8)
    verdict = await policy.check(trigger="every", brief="b", done_when="d", transcript="t",
                                 metadata={"delegation_id": "d1"})
    assert verdict == Verdict(take_back=True, stuck=False, off_scope=True, not_done=False)
    state, questions, metadata = calls[0]
    assert state == {"request": "b\n\nDone when: d", "transcript": "t"}
    assert list(questions) == ["take_back", "stuck", "off_scope", "done"]
    assert metadata == {"delegation_id": "d1", "jev_trigger": "every"}
    assert policy.checks == [{"trigger": "every", "error": "", "ms": 0, "take_back": 0.8, "stuck": 0.79,
                              "off_scope": 0.95, "done": 0.1}]


async def test_done_counts_only_on_finish():
    policy, _ = _policy(_decision(take_back=0, stuck=0, off_scope=0, done=0.1))
    assert (await policy.check(trigger="finish", brief="b", done_when="d", transcript="t")).not_done
    assert not (await policy.check(trigger="every", brief="b", done_when="d", transcript="t")).not_done


@pytest.mark.parametrize("decision,raises", [
    (None, False),
    (JevDecision(error="timeout"), False),
    (None, True),
])
async def test_every_failure_fails_open(decision, raises):
    policy, _ = _policy(decision, raises=raises)
    assert await policy.check(trigger="finish", brief="b", done_when="d", transcript="t") == Verdict()


async def test_no_jev_means_no_checks():
    policy = ChimeraPolicy(ChimeraConfig(enabled=True))
    assert await policy.check(trigger="every", brief="b", done_when="d", transcript="t") == Verdict()
    assert policy.checks == []


def test_the_upgrade_ladder_moves_one_step_once_earned():
    policy, _ = _policy(upgrade_ladder=("gpt-sol", "kimi-k3", "opus"))
    assert policy.next_model("gpt-sol", 1) is None
    assert policy.next_model("gpt-sol", 2) == "kimi-k3"
    assert policy.next_model("kimi-k3", 5) == "opus"
    assert policy.next_model("opus", 5) is None
    assert policy.next_model("haiku", 5) is None


def test_main_compaction_is_only_logged(caplog):
    with caplog.at_level(logging.INFO, logger="anton.core.chimera.policy"):
        ChimeraPolicy.on_main_compaction(type("S", (), {"_sidekick_state": _state()})())
    assert "main agent compacted" in caplog.text
    assert "sidekick=sk1" in caplog.text
