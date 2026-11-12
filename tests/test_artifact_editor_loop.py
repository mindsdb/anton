"""ArtifactEditor: the one-conversation edit loop, driven by a scripted model."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from anton.core.artifact_editor import prompts
from anton.core.artifact_editor.checkers import (
    ChainChecker,
    CheckResult,
    Finding,
    PreviewChecker,
    StaticChecker,
)
from anton.core.artifact_editor.editor import QUOTA, ArtifactEditor, decode_json_object
from anton.core.artifact_editor.records import LOG_ENV_VAR
from anton.core.artifact_editor.workspace import ArtifactWorkspace
from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage

PAGE = """<!doctype html>
<html>
<head>
<title>Sales</title>
<style>
  h1 { color: black; }
</style>
</head>
<body>
<h1>Sales</h1>
<div id="total"></div>
<script>
  function render(rows) {
    const total = rows.length;
    document.getElementById('total').textContent = total;
  }
  render([1, 2, 3]);
</script>
</body>
</html>
"""

COLOR_PATCH = "@index.html\n-  h1 { color: black; }\n+  h1 { color: navy; }\n"
TITLE_PATCH = "@index.html\n-<h1>Sales</h1>\n+<h1>Sales by region</h1>\n"
BAD_PATCH = "@index.html\n-<h1>Revenue</h1>\n+<h1>Revenue by region</h1>\n"


# ── a scripted model ─────────────────────────────────────────────────

Step = LLMResponse | Callable[[list[dict]], LLMResponse] | BaseException


class ScriptedLLM:
    """Answers each streamed call with the next scripted step and keeps what it was sent."""

    planning_model = "scripted"
    coding_model = "scripted"
    max_tokens = 8192

    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)
        self.calls: list[dict] = []

    def stream_budget(self, role: str = "planning", **_kw) -> int:
        return 8192

    async def plan_stream(self, *, system, messages, tools=None, max_tokens=None, **_kw):
        self.calls.append({"system": system, "messages": json.loads(json.dumps(messages)), "tools": tools,
                           "max_tokens": max_tokens})
        if not self.steps:
            raise AssertionError("the model was called more often than scripted")
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            step = await step(messages) if asyncio.iscoroutinefunction(step) else step(messages)
        yield StreamComplete(response=step)

    code_stream = plan_stream


_ids = iter(range(10_000))


def call(name: str, **args) -> ToolCall:
    return ToolCall(id=f"call_{next(_ids)}", name=name, input=args)


def diff(patch: str, *, complete: bool = True, note: str = "") -> ToolCall:
    return call(prompts.APPLY_DIFF, patch=patch, task_complete=complete, response_comment_for_user=note)


def reply(*calls: ToolCall, text: str = "", stop: str = "tool_calls", out: int = 50) -> LLMResponse:
    return LLMResponse(content=text, tool_calls=list(calls), usage=Usage(input_tokens=100, output_tokens=out),
                       stop_reason=stop)


def cut(text: str = "") -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[], usage=Usage(input_tokens=100, output_tokens=8192),
                       stop_reason="length")


class ScriptedChecker:
    def __init__(self, *statuses: str) -> None:
        self.statuses = list(statuses)
        self.calls: list[list[str]] = []

    async def check(self, ws, changed):
        self.calls.append(list(changed))
        status = self.statuses.pop(0) if self.statuses else "passed"
        findings = [Finding("uncaught_exception", "TypeError: x is undefined", viewport="desktop")] if status == "failed" else []
        return CheckResult(status, "render", findings, screenshot_b64="aGVsbG8=" if status == "failed" else None)


class Trail:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def status(self, text: str) -> None:
        self.lines.append("status: " + text)

    def log(self, text: str) -> None:
        self.lines.append(text)


@pytest.fixture(autouse=True)
def _record(tmp_path, monkeypatch):
    monkeypatch.setenv(LOG_ENV_VAR, str(tmp_path / "record.jsonl"))


@pytest.fixture
def app(tmp_path) -> Path:
    folder = tmp_path / "artifacts" / "sales"
    folder.mkdir(parents=True)
    (folder / "index.html").write_text(PAGE, encoding="utf-8")
    (folder / "metadata.json").write_text('{"name": "Sales"}', encoding="utf-8")
    return folder


def editor(folder: Path, llm, checker=None, *, kind="html-app", trail=None) -> ArtifactEditor:
    ws = ArtifactWorkspace(folder, kind=kind)
    return ArtifactEditor(ws, llm, checker or ScriptedChecker(), trail or Trail(), slug=folder.name, name="Sales")


def records(tmp_path) -> list[dict]:
    path = tmp_path / "record.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def tool_results(messages: list[dict]) -> list[str]:
    out = []
    for message in messages:
        if isinstance(message.get("content"), list):
            out += [b["content"] for b in message["content"] if b.get("type") == "tool_result"]
    return out


def texts(messages: list[dict]) -> list[str]:
    out = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out += [b["text"] for b in content if b.get("type") == "text"]
    return out


# ── the happy path ───────────────────────────────────────────────────


async def test_happy_path_is_one_model_call(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH, note="The heading is navy now."))])
    checker = ScriptedChecker("passed")
    outcome = await editor(app, llm, checker).edit("make the heading navy")

    assert outcome.status == "applied"
    assert outcome.model_calls == 1
    assert outcome.reply == "The heading is navy now."
    assert outcome.files == ["index.html"]
    assert outcome.check is not None and outcome.check.status == "passed"
    assert "color: navy" in (app / "index.html").read_text()
    assert checker.calls == [["index.html"]]


async def test_small_app_is_preloaded_as_a_read_files_pair_before_the_request(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH))])
    await editor(app, llm).edit("make the heading navy")

    messages = llm.calls[0]["messages"]
    assert messages[1]["role"] == "assistant"
    use = messages[1]["content"][0]
    assert use["type"] == "tool_use" and use["name"] == prompts.READ_FILES and use["id"] == "preload_1"
    assert use["input"] == {"paths": ["index.html"], "matching": ""}
    result = messages[2]["content"][0]
    assert result["type"] == "tool_result" and "<h1>Sales</h1>" in result["content"]
    # The request comes last, after the files: the stable prefix leads.
    assert messages[2]["content"][-1]["text"].endswith("make the heading navy")
    # Housekeeping is never shown.
    assert "metadata.json" not in json.dumps(messages)


async def test_patch_schema_lists_patch_first():
    schema = next(t for t in prompts.TOOLS if t["name"] == prompts.APPLY_DIFF)["input_schema"]
    assert list(schema["properties"])[0] == "patch"
    assert schema["required"][0] == "patch"


# ── partial apply, then repair ───────────────────────────────────────


async def test_partial_apply_writes_what_placed_and_asks_for_the_rest(tmp_path):
    folder = tmp_path / "artifacts" / "orders"
    (folder / "static").mkdir(parents=True)
    (folder / "static" / "index.html").write_text(PAGE)
    (folder / "backend.py").write_text("def total():\n    return 1\n")
    (folder / "requirements.txt").write_text("fastapi\n")
    both = COLOR_PATCH.replace("@index.html", "@static/index.html") + "\n@backend.py\n-    return 2\n+    return 3\n"
    fixed = "@backend.py\n-    return 1\n+    return 3\n"
    llm = ScriptedLLM([reply(diff(both)), reply(diff(fixed))])
    outcome = await editor(folder, llm, kind="fullstack-stateless-app").edit("navy heading, total 3")

    assert outcome.status == "applied" and outcome.model_calls == 2
    first = tool_results(llm.calls[1]["messages"])[-1]
    assert first.startswith("Part of the patch applied. Written now, with every edit for them in place: static/index.html")
    assert "Not applied: backend.py" in first
    assert "Remaining edit calls this turn: 3" in first
    assert "color: navy" in (folder / "static" / "index.html").read_text()
    assert "return 3" in (folder / "backend.py").read_text()
    assert outcome.files == ["static/index.html", "backend.py"]


async def test_failed_patch_shows_the_file_as_it_now_stands_when_it_changed(app):
    llm = ScriptedLLM([
        reply(diff(TITLE_PATCH, complete=False)),
        reply(diff(TITLE_PATCH.replace("Sales by region", "x"))),  # stale: the line is gone now
        reply(diff("@index.html\n-<h1>Sales by region</h1>\n+<h1>Sales by area</h1>\n")),
    ])
    outcome = await editor(app, llm).edit("rename")
    assert outcome.status == "applied"
    rejected = tool_results(llm.calls[2]["messages"])[-1]
    assert "THESE FILES AS THEY NOW STAND" in rejected
    assert "<h1>Sales by region</h1>" in rejected


async def test_every_rejected_patch_is_recorded(app, tmp_path):
    llm = ScriptedLLM([reply(diff(BAD_PATCH)), reply(diff(TITLE_PATCH))])
    await editor(app, llm).edit("rename")
    rows = records(tmp_path)
    rejected = [r for r in rows if r["kind"] == "patch_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["patch"] == BAD_PATCH and rejected[0]["paths"] == ["index.html"]
    assert rejected[0]["reasons"][0]["path"] == "index.html"
    run = [r for r in rows if r["kind"] == "edit_run"]
    assert run[-1]["status"] == "applied" and run[-1]["model_calls"] == 2


# ── the check ────────────────────────────────────────────────────────


async def test_check_failure_goes_back_for_repair_then_passes(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH)), reply(diff(TITLE_PATCH))])
    checker = ScriptedChecker("failed", "passed")
    outcome = await editor(app, llm, checker).edit("navy heading")

    assert outcome.status == "applied" and outcome.model_calls == 2
    assert outcome.check.status == "passed"
    sent = llm.calls[1]["messages"][-1]["content"]
    assert any(b.get("type") == "text" and b["text"].startswith(prompts.CHECK_FAILED) for b in sent)
    assert any("TypeError: x is undefined" in b.get("text", "") for b in sent)
    # The screenshot goes with the findings.
    assert any(b.get("type") == "image" for b in sent)


async def test_check_runs_only_when_the_model_says_done(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH, complete=False)), reply(diff(TITLE_PATCH))])
    checker = ScriptedChecker("passed")
    await editor(app, llm, checker).edit("two things")
    assert len(checker.calls) == 1


async def test_second_check_failure_restores_the_snapshot(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH)), reply(diff(TITLE_PATCH))])
    checker = ScriptedChecker("failed", "failed")
    outcome = await editor(app, llm, checker).edit("navy heading")

    assert outcome.status == "restored"
    assert "check failed 2 times" in outcome.error
    assert outcome.check is not None and outcome.check.status == "failed"
    assert outcome.files == []
    assert (app / "index.html").read_text() == PAGE


async def test_unverified_is_reported_never_as_passed(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH))])
    outcome = await editor(app, llm, ScriptedChecker("unverified")).edit("navy")
    assert outcome.status == "applied" and outcome.check.status == "unverified"


async def test_image_refusal_drops_the_screenshot_and_retries(app):
    from anton.core.llm.provider import ContentValidationError

    llm = ScriptedLLM([reply(diff(COLOR_PATCH)), ContentValidationError("no images"), reply(diff(TITLE_PATCH))])
    ed = editor(app, llm, ScriptedChecker("failed", "passed"))
    outcome = await ed.edit("navy")
    assert outcome.status == "applied"
    assert not any(b.get("type") == "image" for b in llm.calls[2]["messages"][-1]["content"])
    assert ed.images is False


# ── task_complete ────────────────────────────────────────────────────


async def test_task_complete_is_cleared_by_a_later_call_in_the_batch(app):
    llm = ScriptedLLM([
        reply(diff(COLOR_PATCH, complete=True), call(prompts.READ_FILES, paths=["index.html"], matching="")),
        reply(text="Done: the heading is navy.", stop="stop"),
    ])
    checker = ScriptedChecker("passed")
    outcome = await editor(app, llm, checker).edit("navy")
    assert outcome.model_calls == 2 and outcome.reply == "Done: the heading is navy."
    assert len(checker.calls) == 1


async def test_task_complete_is_cancelled_by_a_failed_call_in_the_batch(app):
    llm = ScriptedLLM([
        reply(diff(BAD_PATCH, complete=False), diff(COLOR_PATCH, complete=True)),
        reply(diff(TITLE_PATCH)),
    ])
    checker = ScriptedChecker("passed")
    outcome = await editor(app, llm, checker).edit("navy")
    assert outcome.model_calls == 2
    assert len(checker.calls) == 1


async def test_task_complete_must_be_a_boolean(app):
    bad = call(prompts.APPLY_DIFF, patch=COLOR_PATCH, task_complete="yes", response_comment_for_user="")
    llm = ScriptedLLM([reply(bad), reply(diff(COLOR_PATCH))])
    outcome = await editor(app, llm).edit("navy")
    assert "task_complete must be a boolean" in tool_results(llm.calls[1]["messages"])[-1]
    assert outcome.status == "applied"


# ── quotas and reads ─────────────────────────────────────────────────


async def test_read_quota_is_enforced(app):
    reads = [reply(call(prompts.READ_FILES, paths=["index.html"], matching="")) for _ in range(QUOTA["read_files"] + 1)]
    llm = ScriptedLLM([*reads, reply(diff(COLOR_PATCH))])
    outcome = await editor(app, llm).edit("navy")
    assert outcome.status == "applied"
    last_read = tool_results(llm.calls[QUOTA["read_files"] + 1]["messages"])[-1]
    assert last_read.startswith("ERROR: read_files has been used 8 times")


async def test_reread_of_an_unchanged_file_is_not_resent(app):
    llm = ScriptedLLM([reply(call(prompts.READ_FILES, paths=["index.html"], matching="")), reply(diff(COLOR_PATCH))])
    await editor(app, llm).edit("navy")
    assert prompts.UNCHANGED in tool_results(llm.calls[1]["messages"])[-1]


async def test_narrow_search_attaches_the_file(tmp_path):
    folder = tmp_path / "artifacts" / "big"
    folder.mkdir(parents=True)
    # Too big to preload whole, small enough to attach.
    filler = "\n".join(f"<p>row {i}</p>" for i in range(5000))
    (folder / "index.html").write_text(PAGE.replace("<div id=\"total\"></div>", filler))
    (folder / "notes.md").write_text("the heading is black\n")
    llm = ScriptedLLM([reply(call(prompts.SEARCH, query="heading is")), reply(diff(COLOR_PATCH))])
    await editor(folder, llm).edit("navy")
    result = tool_results(llm.calls[1]["messages"])[-1]
    assert "notes.md:1:the heading is black" in result
    assert "here it is in full" in result


async def test_every_edit_result_names_the_remaining_budget(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH, complete=False)), reply(diff(TITLE_PATCH))])
    await editor(app, llm).edit("navy")
    assert "Remaining edit calls this turn: 3." in tool_results(llm.calls[1]["messages"])[-1]


async def test_housekeeping_edit_is_refused(app):
    llm = ScriptedLLM([reply(diff('@metadata.json\n-{"name": "Sales"}\n+{"name": "X"}\n')), reply(diff(COLOR_PATCH))])
    outcome = await editor(app, llm).edit("rename")
    assert "refused" in tool_results(llm.calls[1]["messages"])[-1]
    assert (app / "metadata.json").read_text() == '{"name": "Sales"}'
    assert outcome.status == "applied"


# ── cut-offs and empty replies ───────────────────────────────────────


async def test_cut_off_reply_is_never_accepted_and_three_in_a_row_is_tolerated(app):
    llm = ScriptedLLM([cut("I will change"), cut(), cut(), reply(diff(COLOR_PATCH))])
    outcome = await editor(app, llm).edit("navy")
    assert outcome.status == "applied" and outcome.model_calls == 4
    assert prompts.CUT_OFF_NO_CALL in texts(llm.calls[1]["messages"])


async def test_four_cut_offs_in_a_row_give_up(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH, complete=False)), cut(), cut(), cut(), cut()])
    outcome = await editor(app, llm).edit("navy")
    assert outcome.status == "restored"
    assert "ran out of output room 4 times" in outcome.error
    assert (app / "index.html").read_text() == PAGE


async def test_cut_off_lowers_the_effort_one_step(app, monkeypatch):
    llm = ScriptedLLM([cut(), reply(diff(COLOR_PATCH))])
    ed = editor(app, llm)
    ed.effort = None
    monkeypatch.setattr(type(ed), "_provider", lambda self: object())
    llm.__class__._effective_effort = staticmethod(lambda provider: "high")
    try:
        routed = []
        from anton.core.llm import client as client_module

        original = client_module.model_route

        def spy(route):
            routed.append(route)
            return original(route)

        monkeypatch.setattr(client_module, "model_route", spy)
        await ed.edit("navy")
    finally:
        del llm.__class__._effective_effort
    assert routed[-1] is not None and routed[-1].effort == "medium"


async def test_cut_off_after_tool_calls_defers_completion(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH, complete=True), stop="length"), reply(text="Done.", stop="stop")])
    checker = ScriptedChecker("passed")
    outcome = await editor(app, llm, checker).edit("navy")
    assert outcome.model_calls == 2 and outcome.reply == "Done."
    assert prompts.CUT_OFF_AFTER_CALLS in texts(llm.calls[1]["messages"])


async def test_damaged_tool_call_is_not_run(app):
    damaged = ToolCall(id="c1", name=prompts.APPLY_DIFF, input={"patch": "@index.html\n-  h1"}, repaired=True)
    llm = ScriptedLLM([reply(damaged), reply(diff(COLOR_PATCH))])
    outcome = await editor(app, llm).edit("navy")
    assert "cut off" in tool_results(llm.calls[1]["messages"])[-1]
    assert outcome.status == "applied"


async def test_empty_replies_get_a_nudge(app):
    llm = ScriptedLLM([reply(stop="stop"), reply(stop="stop"), reply(diff(COLOR_PATCH))])
    outcome = await editor(app, llm).edit("navy")
    assert outcome.status == "applied"
    assert prompts.EMPTY_REPLY in texts(llm.calls[1]["messages"])


async def test_answer_without_edits_is_not_rewritten(app):
    llm = ScriptedLLM([reply(text="This page has no region data, so I cannot add that filter.", stop="stop")])
    outcome = await editor(app, llm).edit("add a region filter")
    assert outcome.status == "answered" and outcome.model_calls == 1
    assert (app / "index.html").read_text() == PAGE


# ── the last resort ──────────────────────────────────────────────────


async def test_whole_file_rewrite_after_the_edit_budget_runs_out(app):
    rewritten = PAGE.replace("<h1>Sales</h1>", "<h1>Revenue by region</h1>")
    llm = ScriptedLLM([
        *[reply(diff(BAD_PATCH)) for _ in range(QUOTA["apply_diff"])],
        reply(text="I could not place the edit.", stop="stop"),
        reply(text=json.dumps({"summary": "Renamed the heading.", "content": rewritten}), stop="stop"),
    ])
    outcome = await editor(app, llm).edit("rename the heading to Revenue by region")
    assert outcome.status == "applied" and outcome.rewrote
    assert outcome.reply == "Renamed the heading."
    assert (app / "index.html").read_text() == rewritten
    # One file per call, with no tools offered.
    assert llm.calls[-1]["tools"] is None and "FILE TO REWRITE:\nindex.html" in llm.calls[-1]["messages"][0]["content"]


async def test_rewrite_refuses_to_empty_a_file(app):
    llm = ScriptedLLM([
        *[reply(diff(BAD_PATCH)) for _ in range(QUOTA["apply_diff"])],
        reply(text="no", stop="stop"),
        reply(text='{"summary": "x", "content": "   "}', stop="stop"),
    ])
    outcome = await editor(app, llm).edit("rename")
    assert outcome.status == "restored" and "refusing to empty" in outcome.error
    assert (app / "index.html").read_text() == PAGE


def test_lenient_json():
    assert decode_json_object('```json\n{"content": "a"}\n```') == {"content": "a"}
    assert decode_json_object('Here: {"content": "a\nb"} ok') == {"content": "a\nb"}
    assert decode_json_object("nothing") is None


# ── giving up and cancellation ───────────────────────────────────────


async def test_crash_restores_every_file_the_edit_wrote(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH, complete=False)), RuntimeError("provider exploded")])
    outcome = await editor(app, llm).edit("navy")
    assert outcome.status == "restored" and "provider exploded" in outcome.error
    assert (app / "index.html").read_text() == PAGE


async def test_cancellation_restores_and_reraises(app):
    started = asyncio.Event()

    async def hang(_messages):
        started.set()
        await asyncio.Event().wait()

    llm = ScriptedLLM([reply(diff(COLOR_PATCH, complete=False)), hang])
    task = asyncio.create_task(editor(app, llm).edit("navy"))
    await asyncio.wait_for(started.wait(), 5)
    assert "color: navy" in (app / "index.html").read_text()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (app / "index.html").read_text() == PAGE


async def test_created_file_is_removed_on_restore(tmp_path):
    folder = tmp_path / "artifacts" / "orders"
    (folder / "static").mkdir(parents=True)
    (folder / "static" / "index.html").write_text(PAGE)
    (folder / "backend.py").write_text("x = 1\n")
    (folder / "requirements.txt").write_text("fastapi\n")
    llm = ScriptedLLM([reply(diff("@+static/app.js\nconsole.log(1);\n"))])
    outcome = await editor(folder, llm, ScriptedChecker("failed"), kind="fullstack-stateless-app").edit("x")
    # One failed check is repairable; script ends -> the second call errors -> restored.
    assert outcome.status == "restored"
    assert not (folder / "static" / "app.js").exists()


# ── checkers ─────────────────────────────────────────────────────────


async def test_preview_checker_unverified_paths(app):
    ws = ArtifactWorkspace(app, kind="html-app")

    async def answer(value):
        async def ask(slug, timeout):
            assert slug == "sales"
            if isinstance(value, BaseException):
                raise value
            return value
        return ask

    for value in (None, {"status": "unavailable"}, {"status": "weird"}, RuntimeError("bridge down")):
        result = await PreviewChecker(await answer(value), slug="sales").check(ws, ["index.html"])
        assert result.status == "unverified", value
    failed = await PreviewChecker(await answer({"status": "failed", "errors": [
        {"type": "error", "message": "boom", "file": "index.html", "line": 3}]}), slug="sales").check(ws, [])
    assert failed.status == "failed" and failed.findings[0].source == "index.html:3"
    passed = await PreviewChecker(await answer({"status": "passed", "errors": [], "ms": 12}), slug="sales").check(ws, [])
    assert passed.status == "passed" and passed.ms == 12


async def test_preview_checker_times_out_to_unverified(app):
    ws = ArtifactWorkspace(app, kind="html-app")

    async def never(slug, timeout):
        await asyncio.sleep(10)

    checker = PreviewChecker(never, slug="sales", timeout_s=0.01)
    import anton.core.artifact_editor.checkers as module

    original = module.asyncio.wait_for

    async def quick(aw, timeout):
        return await original(aw, 0.05)

    module.asyncio.wait_for = quick
    try:
        result = await checker.check(ws, [])
    finally:
        module.asyncio.wait_for = original
    assert result.status == "unverified"


async def test_static_only_chain_is_unverified_never_passed(app):
    ws = ArtifactWorkspace(app, kind="html-app")
    result = await ChainChecker([StaticChecker(node=None)]).check(ws, ["index.html"])
    assert result.status == "unverified"


async def test_chain_hands_over_from_unverified_to_the_next_channel(app):
    ws = ArtifactWorkspace(app, kind="html-app")

    class Fixed:
        def __init__(self, status, channel):
            self.status, self.channel = status, channel

        async def check(self, ws, changed):
            return CheckResult(self.status, self.channel, note=f"{self.channel} said {self.status}")

    result = await ChainChecker([StaticChecker(node=None), Fixed("unverified", "render"), Fixed("passed", "preview")]).check(ws, [])
    assert result.status == "passed" and result.channel == "preview"
    result = await ChainChecker([StaticChecker(node=None), Fixed("unverified", "render"), Fixed("unverified", "preview")]).check(ws, [])
    assert result.status == "unverified" and "render said unverified" in result.note


async def test_static_checker_finds_json_python_and_script_errors(tmp_path):
    folder = tmp_path / "a"
    (folder / "static").mkdir(parents=True)
    (folder / "static" / "data.json").write_text("{bad")
    (folder / "backend.py").write_text("def f(:\n")
    (folder / "static" / "index.html").write_text("<html><body><script>let x = ;</script></body></html>")
    ws = ArtifactWorkspace(folder, kind="fullstack-stateless-app")
    result = await StaticChecker().check(ws, ["static/data.json", "backend.py", "static/index.html"])
    kinds = [f.kind for f in result.findings]
    assert result.status == "failed"
    assert "invalid_json" in kinds and kinds.count("syntax_error") >= 1
    if shutil.which("node"):
        assert any("inline script" in f.message for f in result.findings)


async def test_static_checker_passes_a_good_page(app):
    ws = ArtifactWorkspace(app, kind="html-app")
    assert (await StaticChecker().check(ws, ["index.html"])).status == "passed"


async def test_static_failures_do_not_spend_the_browser_check_budget(app):
    class StaticThenPass:
        def __init__(self):
            self.n = 0

        async def check(self, ws, changed):
            self.n += 1
            if self.n <= 2:
                return CheckResult("failed", "static", [Finding("syntax_error", "does not parse")])
            return CheckResult("passed", "render")

    llm = ScriptedLLM([reply(diff(COLOR_PATCH)), reply(diff(TITLE_PATCH)),
                       reply(diff("@index.html\n-<h1>Sales by region</h1>\n+<h1>Sales</h1>\n"))])
    outcome = await editor(app, llm, StaticThenPass()).edit("navy")
    assert outcome.status == "applied" and outcome.model_calls == 3


async def test_the_embedded_data_shape_is_shown_although_the_data_is_masked(tmp_path):
    from anton.core.tools.generate_artifact import observed_data

    folder = tmp_path / "artifacts" / "survey"
    folder.mkdir(parents=True)
    value = {"teams": ["A", "B"], "team_summaries": {"A": {"score": 4.1}}}
    (folder / "index.html").write_text(observed_data.bind(PAGE, {observed_data.identity(value): value}))
    llm = ScriptedLLM([reply(diff(COLOR_PATCH))])
    await editor(folder, llm).edit("navy")
    intro = llm.calls[0]["messages"][0]["content"]
    assert "EMBEDDED DATA" in intro and '"team_summaries"' in intro
    assert "anton:keep observed-data" in json.dumps(llm.calls[0]["messages"])


async def test_a_headerless_patch_on_a_one_page_app_goes_to_the_page(app):
    llm = ScriptedLLM([reply(diff(COLOR_PATCH.split("\n", 1)[1]))])
    outcome = await editor(app, llm).edit("navy")
    assert outcome.status == "applied" and outcome.model_calls == 1


async def test_a_patch_opening_with_a_bare_separator_goes_to_the_page(app):
    llm = ScriptedLLM([reply(diff("@@\n" + COLOR_PATCH.split("\n", 1)[1]))])
    outcome = await editor(app, llm).edit("navy")
    assert outcome.status == "applied" and outcome.model_calls == 1


async def test_a_header_glued_to_a_separator_is_read_as_the_header(tmp_path):
    folder = tmp_path / "artifacts" / "orders"
    (folder / "static").mkdir(parents=True)
    (folder / "static" / "index.html").write_text(PAGE)
    (folder / "backend.py").write_text("x = 1\n")
    (folder / "requirements.txt").write_text("fastapi\n")
    patch = "@@static/index.html\n" + COLOR_PATCH.split("\n", 1)[1]
    llm = ScriptedLLM([reply(diff(patch))])
    outcome = await editor(folder, llm, kind="fullstack-stateless-app").edit("navy")
    assert outcome.status == "applied" and outcome.model_calls == 1
