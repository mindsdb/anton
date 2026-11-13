"""run_quick_edit: the quick-edit bar's direct edit (ENG-4227), with a scripted model and checker."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from anton.core.artifact_editor import prompts
from anton.core.artifact_editor.checkers import CheckResult, Finding
from anton.core.artifact_editor.editor import ArtifactEditor
from anton.core.artifact_editor.quick import QuickEditResult, quick_edit_llm_from_settings, run_quick_edit
from anton.core.artifact_editor.records import LOG_ENV_VAR
from anton.core.artifact_editor.workspace import ArtifactWorkspace
from tests.test_artifact_editor_loop import (
    COLOR_PATCH,
    PAGE,
    TITLE_PATCH,
    ScriptedChecker,
    ScriptedLLM,
    call,
    diff,
    reply,
    texts,
)


@pytest.fixture(autouse=True)
def _record(tmp_path, monkeypatch):
    monkeypatch.setenv(LOG_ENV_VAR, str(tmp_path / "record.jsonl"))
    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)


@pytest.fixture
def app(tmp_path) -> Path:
    folder = tmp_path / "artifacts" / "sales"
    folder.mkdir(parents=True)
    (folder / "index.html").write_text(PAGE, encoding="utf-8")
    (folder / "metadata.json").write_text('{"name": "Sales", "type": "html-app"}', encoding="utf-8")
    return folder


def digest(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def hand_over(reason: str = "This needs last quarter's numbers, which the artifact does not have."):
    return call(prompts.HAND_TO_AGENT, reason=reason)


async def quick(folder: Path, llm, checker=None, message: str = "make the heading navy", **kw) -> QuickEditResult:
    stages: list[dict] = kw.pop("stages", [])
    return await run_quick_edit(folder, message, kind="html-app", llm=llm,
                                checker=checker if checker is not None else ScriptedChecker("passed"),
                                on_progress=stages.append, **kw)


# ── applied ──────────────────────────────────────────────────────────


async def test_a_quick_edit_applies_in_one_call_and_reports_its_stages(app):
    stages: list[dict] = []
    llm = ScriptedLLM([reply(diff(COLOR_PATCH, note="The heading is navy now."))])
    result = await quick(app, llm, stages=stages)

    assert result.status == "applied"
    assert result.changed == ["index.html"]
    assert result.summary == "The heading is navy now."
    assert result.escalate_reason is None
    assert result.check is not None and result.check.status == "passed"
    assert (result.rounds, result.llm_calls) == (0, 1)
    assert result.ms >= 0
    assert "color: navy" in (app / "index.html").read_text()
    assert [s["stage"] for s in stages] == ["reading", "writing", "checking"]


async def test_the_summary_is_one_short_sentence(app):
    long_note = "The heading is navy now. " + "It also has more detail about the change. " * 20
    llm = ScriptedLLM([reply(diff(COLOR_PATCH, note=long_note))])
    result = await quick(app, llm)
    assert len(result.summary) <= 240 and result.summary.startswith("The heading is navy now.")


async def test_unverified_is_applied_and_says_so(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH))])
    result = await quick(app, llm, ScriptedChecker("unverified"))
    assert result.status == "applied" and result.check.status == "unverified"


# ── fix rounds ───────────────────────────────────────────────────────


async def test_a_failed_check_starts_a_fix_round_with_the_findings(app):
    stages: list[dict] = []
    llm = ScriptedLLM([reply(diff(COLOR_PATCH)), reply(diff(TITLE_PATCH))])
    result = await quick(app, llm, ScriptedChecker("failed", "passed"), stages=stages)

    assert result.status == "applied" and result.rounds == 1 and result.llm_calls == 2
    repair = " ".join(texts(llm.calls[1]["messages"][-1:]))
    assert repair.startswith(prompts.CHECK_FAILED) and "TypeError: x is undefined" in repair
    assert {"stage": "fixing", "round": 1, "findings": 1} in stages
    assert [s["stage"] for s in stages] == ["reading", "writing", "checking", "fixing", "checking"]


async def test_two_fix_rounds_are_allowed(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH)), reply(diff(TITLE_PATCH)),
                       reply(diff("@index.html\n-<h1>Sales by region</h1>\n+<h1>Sales by area</h1>\n"))])
    result = await quick(app, llm, ScriptedChecker("failed", "failed", "passed"))
    assert result.status == "applied" and result.rounds == 2 and result.llm_calls == 3


async def test_a_third_failure_gives_up_and_restores_the_folder(app):
    before = digest(app)
    llm = ScriptedLLM([reply(diff(COLOR_PATCH)), reply(diff(TITLE_PATCH)),
                       reply(diff("@index.html\n-<h1>Sales by region</h1>\n+<h1>Sales by area</h1>\n"))])
    result = await quick(app, llm, ScriptedChecker("failed", "failed", "failed"))

    assert result.status == "failed"
    assert result.rounds == 2 and result.llm_calls == 3
    assert result.changed == []
    assert result.check is not None and result.check.status == "failed"
    assert "unchanged" in result.summary and "TypeError" in result.summary
    assert "after 2 fix rounds" in result.error
    assert digest(app) == before


async def test_max_fix_rounds_zero_gives_up_on_the_first_failure(app):
    before = digest(app)
    llm = ScriptedLLM([reply(diff(COLOR_PATCH))])
    result = await quick(app, llm, ScriptedChecker("failed"), max_fix_rounds=0)
    assert result.status == "failed" and result.rounds == 0 and result.llm_calls == 1
    assert digest(app) == before


async def test_static_failures_count_as_fix_rounds_too(app):
    def static_failure() -> CheckResult:
        return CheckResult("failed", "static", [Finding("syntax_error", "index.html: Unexpected token")])

    class StaticFails:
        calls = 0

        async def check(self, ws, changed):
            self.calls += 1
            return static_failure()

    before = digest(app)
    llm = ScriptedLLM([reply(diff(COLOR_PATCH)), reply(diff(TITLE_PATCH)),
                       reply(diff("@index.html\n-<h1>Sales by region</h1>\n+<h1>Sales by area</h1>\n"))])
    checker = StaticFails()
    result = await quick(app, llm, checker)
    assert result.status == "failed" and result.rounds == 2 and checker.calls == 3
    assert digest(app) == before


# ── escalation ───────────────────────────────────────────────────────


async def test_the_editor_can_hand_a_request_to_the_agent(app):
    before = digest(app)
    llm = ScriptedLLM([reply(hand_over())])
    result = await quick(app, llm, message="add last quarter's numbers")

    assert result.status == "escalate"
    assert result.escalate_reason == "This needs last quarter's numbers, which the artifact does not have."
    assert result.summary == result.escalate_reason
    assert result.changed == [] and result.llm_calls == 1
    assert digest(app) == before
    # The tool and the quick-edit rules are offered only here.
    assert prompts.HAND_TO_AGENT in [t["name"] for t in llm.calls[0]["tools"]]
    assert "QUICK EDIT" in llm.calls[0]["system"]


async def test_escalating_after_an_edit_restores_the_folder(app):
    before = digest(app)
    llm = ScriptedLLM([reply(diff(COLOR_PATCH, complete=False)), reply(hand_over("Needs data from the CRM."))])
    result = await quick(app, llm)

    assert result.status == "escalate" and result.escalate_reason == "Needs data from the CRM."
    assert result.changed == []
    assert digest(app) == before


async def test_escalating_in_the_same_batch_as_an_edit_restores_the_folder(app):
    before = digest(app)
    llm = ScriptedLLM([reply(diff(COLOR_PATCH), hand_over())])
    result = await quick(app, llm)
    assert result.status == "escalate"
    assert digest(app) == before


async def test_escalating_during_a_fix_round_restores_the_folder(app):
    before = digest(app)
    llm = ScriptedLLM([reply(diff(COLOR_PATCH)), reply(hand_over("The data has no forecast."))])
    result = await quick(app, llm, ScriptedChecker("failed"))
    assert result.status == "escalate" and result.rounds == 1
    assert digest(app) == before


async def test_a_blank_reason_still_escalates(app):
    llm = ScriptedLLM([reply(hand_over("  "))])
    result = await quick(app, llm)
    assert result.status == "escalate" and result.escalate_reason


async def test_the_default_editor_has_no_hand_over_tool(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH))])
    editor = ArtifactEditor(ArtifactWorkspace(app, kind="html-app"), llm, ScriptedChecker("passed"))
    await editor.edit("navy")
    assert prompts.HAND_TO_AGENT not in [t["name"] for t in llm.calls[0]["tools"]]
    assert "QUICK EDIT" not in llm.calls[0]["system"]


async def test_the_default_editor_treats_a_hand_over_call_as_an_unknown_tool(app):
    llm = ScriptedLLM([reply(hand_over()), reply(diff(COLOR_PATCH))])
    editor = ArtifactEditor(ArtifactWorkspace(app, kind="html-app"), llm, ScriptedChecker("passed"))
    outcome = await editor.edit("navy")
    assert outcome.status == "applied"


# ── noop, history, failures ──────────────────────────────────────────


async def test_a_plain_reply_without_edits_is_a_noop(app):
    before = digest(app)
    llm = ScriptedLLM([reply(text="The heading is already navy.", stop="end_turn")])
    result = await quick(app, llm)
    assert result.status == "noop" and result.summary == "The heading is already navy."
    assert result.changed == [] and digest(app) == before


async def test_an_empty_message_is_a_noop_without_a_model_call(app):
    llm = ScriptedLLM([])
    result = await quick(app, llm, message="  ")
    assert result.status == "noop" and llm.calls == []


async def test_history_gives_a_follow_up_its_context(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH))])
    history = [{"role": "user", "text": "make the heading navy"},
               {"role": "assistant", "text": "The heading is navy now."}]
    await quick(app, llm, message="bigger", history=history)
    request = texts(llm.calls[0]["messages"])[-1]
    assert "- user: make the heading navy" in request
    assert "- you (the editor): The heading is navy now." in request
    assert request.rstrip().endswith("The request now:\nbigger")


async def test_without_history_the_request_is_the_message(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH))])
    await quick(app, llm, message="make the heading navy")
    assert texts(llm.calls[0]["messages"])[-1].endswith("REQUEST:\nmake the heading navy")


async def test_a_model_crash_fails_and_restores(app):
    before = digest(app)
    llm = ScriptedLLM([reply(diff(COLOR_PATCH, complete=False)), RuntimeError("provider exploded")])
    result = await quick(app, llm)
    assert result.status == "failed" and "unchanged" in result.summary
    assert digest(app) == before


async def test_an_unknown_kind_fails_without_a_model_call(app):
    llm = ScriptedLLM([])
    result = await run_quick_edit(app, "navy", kind="spreadsheet", llm=llm, checker=None)
    assert result.status == "failed" and llm.calls == []


async def test_a_progress_callback_that_raises_does_not_fail_the_edit(app):
    def explode(event):
        raise ValueError("ui went away")

    llm = ScriptedLLM([reply(diff(COLOR_PATCH))])
    result = await run_quick_edit(app, "navy", kind="html-app", llm=llm, checker=ScriptedChecker("passed"),
                                  on_progress=explode)
    assert result.status == "applied"


async def test_no_checker_uses_the_environments_checks(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH))])
    result = await run_quick_edit(app, "navy", kind="html-app", llm=llm, checker=None)
    # Static checks only (no browser configured here): never "passed".
    assert result.status == "applied" and result.check.status == "unverified"


def test_quick_edit_llm_from_settings_builds_the_client_like_chat_session(monkeypatch):
    from anton.core.llm import client as client_module

    built = []

    class FakeClient:
        model_call_idle_timeout_s = 600.0

        @classmethod
        def from_settings(cls, settings):
            built.append(settings)
            return cls()

    monkeypatch.setattr(client_module, "LLMClient", FakeClient)

    class Settings:
        model_call_idle_timeout_s = 42.0

    settings = Settings()
    llm = quick_edit_llm_from_settings(settings)
    assert built == [settings] and llm.model_call_idle_timeout_s == 42.0
