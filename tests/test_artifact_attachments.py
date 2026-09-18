"""S-01: files the user attached to the conversation reach the pipeline.

Data files are read by the gathering step from their absolute path; assets
are copied into the artifact by the orchestrator before generation and the
generators are told the relative name. None of it is a model action.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from anton.core.tools.generate_artifact import attachments as att_mod
from anton.core.tools.generate_artifact import engine, orchestrator, prompts
from anton.core.tools.generate_artifact.attachments import (
    ASSET_MAX_BYTES, Attachment, KIND_ASSET, KIND_DATA, render_for_gathering,
    render_for_generation, resolve_attachments, stage_assets,
)
from anton.core.tools.generate_artifact.discovery import checkpoint as cp
from anton.core.tools.generate_artifact.discovery.prompts import build_call_kickoff
from anton.core.tools.generate_artifact.state import GenState


def _state(tmp_path: Path, **kw) -> GenState:
    base = dict(
        session=AsyncMock(), artifact_type="html-app", artifact_path=tmp_path,
        slug="a", brief="Show sales", is_fullstack=False,
        user_request="show sales", agent_understanding="a sales page",
    )
    base.update(kw)
    return GenState(**base)


def _files(tmp_path: Path) -> tuple[Path, Path]:
    src = tmp_path / "uploads"
    src.mkdir()
    csv = src / "sales.csv"
    csv.write_text("day,amount\n1,2\n")
    png = src / "logo.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)
    return csv, png


# ── resolve ─────────────────────────────────────────────────────────────────

def test_resolve_keeps_files_and_classifies_them(tmp_path: Path):
    csv, png = _files(tmp_path)
    kept, dropped = resolve_attachments([str(csv), str(png)])
    assert dropped == []
    assert [(a.name, a.kind) for a in kept] == [("sales.csv", KIND_DATA), ("logo.png", KIND_ASSET)]
    assert kept[1].size == png.stat().st_size


def test_resolve_drops_missing_paths_and_directories_with_a_reason(tmp_path: Path):
    csv, _ = _files(tmp_path)
    kept, dropped = resolve_attachments([str(tmp_path / "nope.png"), str(tmp_path / "uploads"), "", None])
    assert kept == []
    assert dropped == [f"{tmp_path / 'nope.png'}: not a file", f"{tmp_path / 'uploads'}: not a file"]


def test_resolve_collapses_duplicates(tmp_path: Path):
    csv, _ = _files(tmp_path)
    kept, _ = resolve_attachments([str(csv), str(csv.parent / "." / "sales.csv")])
    assert len(kept) == 1


# ── stage ───────────────────────────────────────────────────────────────────

def test_stage_copies_assets_to_the_root_for_an_html_app_and_leaves_data_alone(tmp_path: Path):
    csv, png = _files(tmp_path)
    art = tmp_path / "art"
    art.mkdir()
    kept, _ = resolve_attachments([str(csv), str(png)])
    written = stage_assets(kept, art, static=False)
    assert written == ["logo.png"]
    assert (art / "logo.png").read_bytes() == png.read_bytes()
    assert not (art / "sales.csv").exists()
    assert kept[1].staged == "logo.png" and kept[0].staged == ""


def test_stage_copies_assets_into_static_for_a_fullstack_app(tmp_path: Path):
    _, png = _files(tmp_path)
    art = tmp_path / "art"
    art.mkdir()
    kept, _ = resolve_attachments([str(png)])
    assert stage_assets(kept, art, static=True) == ["static/logo.png"]
    assert (art / "static" / "logo.png").is_file()


def test_stage_skips_an_oversized_asset_and_says_why(tmp_path: Path):
    _, png = _files(tmp_path)
    art = tmp_path / "art"
    art.mkdir()
    big = Attachment(path=str(png), name="logo.png", size=ASSET_MAX_BYTES + 1, kind=KIND_ASSET)
    assert stage_assets([big], art, static=False) == []
    assert not (art / "logo.png").exists()
    assert big.staged == "" and "not copied" in big.skipped


def test_stage_is_idempotent(tmp_path: Path):
    _, png = _files(tmp_path)
    art = tmp_path / "art"
    art.mkdir()
    kept, _ = resolve_attachments([str(png)])
    stage_assets(kept, art, static=False)
    assert stage_assets(kept, art, static=False) == ["logo.png"]


# ── rendering ───────────────────────────────────────────────────────────────

def test_gathering_section_lists_absolute_paths_and_roles(tmp_path: Path):
    csv, png = _files(tmp_path)
    kept, _ = resolve_attachments([str(csv), str(png)])
    text = render_for_gathering(kept)
    assert text.startswith("## Attached files")
    assert f"`{csv.resolve()}`" in text and "read it in the scratchpad" in text
    assert f"`{png.resolve()}`" in text and "copied into the artifact folder" in text


def test_gathering_section_is_empty_without_attachments():
    assert render_for_gathering([]) == ""


def test_generation_section_names_staged_assets_by_relative_name_only(tmp_path: Path):
    csv, png = _files(tmp_path)
    kept, _ = resolve_attachments([str(csv), str(png)])
    kept[1].staged = "logo.png"
    text = render_for_generation(kept)
    assert '<img src="logo.png">' in text
    assert str(png.resolve()) not in text  # never the absolute source path
    assert "`sales.csv`" in text and "not in the artifact folder" in text


def test_generation_section_forbids_an_asset_that_was_not_copied(tmp_path: Path):
    _, png = _files(tmp_path)
    kept, _ = resolve_attachments([str(png)])
    kept[0].skipped = "larger than 15 MB, not copied"
    text = render_for_generation(kept)
    assert "NOT available" in text and "Do not reference it" in text


# ── wiring: kickoffs, orchestrator, prompts ─────────────────────────────────

def test_call_kickoff_carries_the_attached_files_only_when_there_are_any(tmp_path: Path):
    csv, _ = _files(tmp_path)
    st = _state(tmp_path, session=SimpleNamespace(question_count=0))
    assert "## Attached files" not in build_call_kickoff(st)
    st.attachments, _ = resolve_attachments([str(csv)])
    kickoff = build_call_kickoff(st)
    assert "## Attached files" in kickoff and str(csv.resolve()) in kickoff


def test_spec_context_carries_the_generation_section(tmp_path: Path):
    _, png = _files(tmp_path)
    st = _state(tmp_path)
    st.attachments, _ = resolve_attachments([str(png)])
    st.attachments[0].staged = "logo.png"
    assert '<img src="logo.png">' in orchestrator._spec_context(st)


def test_stage_step_copies_assets_and_records_them_as_artifact_files(tmp_path: Path):
    csv, png = _files(tmp_path)
    art = tmp_path / "art"
    art.mkdir()
    st = _state(tmp_path, artifact_path=art)
    st.attachments, _ = resolve_attachments([str(csv), str(png)])
    orchestrator._stage_attachments(st)
    assert (art / "logo.png").is_file()
    assert st.files_written == ["logo.png"]
    assert [r for r in st.trace if r.node == "stage_attachments"][0].outcome == "done"


def test_stage_step_uses_static_for_fullstack(tmp_path: Path):
    _, png = _files(tmp_path)
    art = tmp_path / "art"
    art.mkdir()
    st = _state(tmp_path, artifact_path=art, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.attachments, _ = resolve_attachments([str(png)])
    orchestrator._stage_attachments(st)
    assert st.files_written == ["static/logo.png"]


def test_stage_step_is_silent_without_attachments(tmp_path: Path):
    st = _state(tmp_path)
    orchestrator._stage_attachments(st)
    assert st.trace == [] and st.files_written == []


def test_generator_prompts_allow_only_the_attached_assets_as_local_files():
    html = prompts.build_subagent_system_prompt(Path("/x"))
    assert "except the assets\nlisted under `## Attached files`" in html
    assert "`## Attached files`" in html  # the inputs list names the section
    front = prompts.build_frontend_system_prompt(Path("/x"))
    assert "assets listed under `## Attached files`, already in `static/`" in front


def test_gathering_instruction_treats_an_attached_data_file_as_a_source():
    from anton.core.tools.generate_artifact.discovery.prompts import _GATHERING_INSTRUCTION

    assert "A data file under `## Attached files` is such a source" in _GATHERING_INSTRUCTION


# ── checkpoint round trip and restore ───────────────────────────────────────

def test_checkpoint_persists_attachment_records(tmp_path: Path):
    _, png = _files(tmp_path)
    art = tmp_path / "art"
    art.mkdir()
    st = _state(tmp_path, artifact_path=art)
    st.attachments, _ = resolve_attachments([str(png)])
    st.attachments[0].staged = "logo.png"
    orchestrator._save_checkpoint(st, cp.STAGE_PRD_WRITTEN)
    stored = cp.load(art)
    assert stored.attachments == [st.attachments[0].to_dict()]


def test_restore_inherits_the_stored_records_only_when_the_call_named_none(tmp_path: Path):
    csv, png = _files(tmp_path)
    stored = cp.DiscoveryCheckpoint(
        pipeline_stage=cp.STAGE_PRD_WRITTEN, artifact_type="html-app",
        attachments=[Attachment(path=str(png), name="logo.png", size=1, kind=KIND_ASSET, staged="logo.png").to_dict()],
    )
    st = _state(tmp_path)
    engine._restore(st, stored)
    assert [a.name for a in st.attachments] == ["logo.png"]
    assert st.attachments[0].staged == "logo.png"

    fresh = _state(tmp_path)
    fresh.attachments, _ = resolve_attachments([str(csv)])
    engine._restore(fresh, stored)
    assert [a.name for a in fresh.attachments] == ["sales.csv"]


def test_restore_drops_malformed_records(tmp_path: Path):
    stored = cp.DiscoveryCheckpoint(attachments=[{"name": "x"}, "junk", None])  # no path / not dicts
    st = _state(tmp_path)
    engine._restore(st, stored)
    assert st.attachments == []


# ── generate(): resolution is recorded in the trace ─────────────────────────

async def test_generate_records_kept_and_dropped_attachments(tmp_path: Path, monkeypatch):
    csv, _ = _files(tmp_path)
    art = tmp_path / "art"
    art.mkdir()
    seen = {}

    async def fake_run(state, *, entry):
        seen["attachments"] = list(state.attachments)
        seen["trace"] = list(state.trace)
        return {"status": "generated", "files_written": [], "internal_files": [], "trace": []}

    monkeypatch.setattr(orchestrator, "run", fake_run)
    monkeypatch.setattr(engine, "_scratchpads_context", lambda session: "")
    monkeypatch.setattr(orchestrator, "_datasource_context", lambda session: "")
    session = AsyncMock()
    out = await engine.generate(
        session=session, slug="a", artifact_path=art, artifact_type="html-app",
        user_request="r", agent_understanding="u",
        attachments=[str(csv), str(tmp_path / "missing.png")],
    )
    assert out["status"] == "generated"
    assert [a.name for a in seen["attachments"]] == ["sales.csv"]
    rec = [r for r in seen["trace"] if r.node == "attachments"][0]
    assert rec.outcome == "done" and "sales.csv (data)" in rec.detail and "missing.png: not a file" in rec.detail


def test_module_constants_are_the_ones_the_docs_quote():
    assert att_mod.ASSET_MAX_BYTES == 15 * 1024 * 1024
    assert ".csv" in att_mod.DATA_SUFFIXES and ".png" not in att_mod.DATA_SUFFIXES
