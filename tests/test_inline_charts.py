"""ENG-4684: charts inside the chat reply.

Two halves: the metadata field `display` ("inline" for an interactive view
worth keeping, "widget" for a page that collects the user's input) that marks
an html-app the Cowork chat shows live in the reply, and the IN-REPLY VISUALS prompt block that
teaches ```chartjs / ```mermaid fences and inline artifacts, only on surfaces
whose chat draws them.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from anton.core.artifacts import ArtifactStore
from anton.core.artifacts.internal_files import INLINE_STATE_FILENAME, NON_CONTENT_NAMES
from anton.core.artifacts.listing import FIELDS, SUMMARY_FIELDS, STATE_INLINE_MAX_BYTES, render_listing
from anton.core.artifacts.models import Artifact
from anton.core.llm.prompt_builder import (
    ChatSystemPromptBuilder,
    SystemPromptContext,
    renders_chart_blocks,
)
from anton.core.llm.prompts import IN_REPLY_VISUALS_PROMPT, in_reply_visuals
from anton.core.tools import tool_defs
from anton.core.tools.tool_handlers import (
    handle_create_artifact,
    handle_update_artifact_metadata,
)

_RECORD = dict(
    id="3f2c1a9b7d4e4c0f9a51b2e6c8d0f413",
    slug="price-chart-3f2c1a9b",
    createdAt="2026-11-24T10:00:00+00:00",
    updatedAt="2026-11-24T10:00:00+00:00",
    name="Price chart",
    description="AAPL with range tabs",
    type="html-app",
    primary="index.html",
)


# ── metadata model ──────────────────────────────────────────────


def test_display_defaults_to_the_usual_card():
    assert Artifact(**_RECORD).display is None


def test_a_record_without_display_still_loads():
    """Every metadata.json written before ENG-4684 has no `display` key."""
    assert Artifact.model_validate(dict(_RECORD)).display is None


def test_display_inline_round_trips_through_json():
    artifact = Artifact(**_RECORD, display="inline")
    loaded = Artifact.model_validate(json.loads(artifact.model_dump_json()))
    assert loaded.display == "inline"


@pytest.mark.parametrize("display", ["inline", "widget"])
def test_every_display_round_trips_through_json(display):
    artifact = Artifact(**_RECORD, display=display)
    assert Artifact.model_validate(json.loads(artifact.model_dump_json())).display == display


@pytest.mark.parametrize("bad", ["card", "INLINE", "Widget", "", 1])
def test_display_rejects_other_values(bad):
    with pytest.raises(ValidationError):
        Artifact(**_RECORD, display=bad)


# ── store ───────────────────────────────────────────────────────


def test_store_writes_display_into_metadata_json(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    artifact = store.create(name="Chart", description="d", type="html-app", display="inline")
    on_disk = json.loads(store.metadata_path(artifact.slug).read_text(encoding="utf-8"))
    assert on_disk["display"] == "inline"
    assert store.open(artifact.slug).display == "inline"


def test_store_update_sets_and_clears_display(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    slug = store.create(name="Chart", description="d", type="html-app").slug
    assert store.update(slug, display="inline").display == "inline"
    assert store.update(slug, primary="index.html").display == "inline"  # untouched when omitted
    assert store.update(slug, display=None).display is None


def test_store_refuses_an_unknown_display(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.create(name="Chart", description="d", type="html-app", display="card")  # type: ignore[arg-type]
    slug = store.create(name="Chart", description="d", type="html-app").slug
    with pytest.raises(ValueError):
        store.update(slug, display="card")  # type: ignore[arg-type]
    assert store.open(slug).display is None


# ── tools ───────────────────────────────────────────────────────


def _session(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        _workspace=SimpleNamespace(artifacts_dir=tmp_path), _data_vault=None, _fullstack_offered=True,
    )


def _metadata(tmp_path: Path) -> dict:
    (path,) = tmp_path.glob("*/metadata.json")
    return json.loads(path.read_text(encoding="utf-8"))


async def test_create_artifact_persists_display(tmp_path):
    outcome = await handle_create_artifact(_session(tmp_path), {
        "name": "Price chart", "description": "d", "type": "html-app",
        "primary": "index.html", "display": "inline",
    })
    assert outcome.ok is True
    assert "Inline section" in outcome.content
    assert _metadata(tmp_path)["display"] == "inline"


async def test_create_artifact_without_display_is_the_usual_card(tmp_path):
    outcome = await handle_create_artifact(_session(tmp_path), {
        "name": "Dashboard", "description": "d", "type": "html-app",
    })
    assert outcome.ok is True
    assert "Inline section" not in outcome.content
    assert _metadata(tmp_path)["display"] is None


@pytest.mark.parametrize("tc_input, fragment", [
    ({"type": "html-app", "display": "card"}, "must be one of"),
    ({"type": "document", "display": "inline"}, "only an html-app"),
])
async def test_create_artifact_refuses_a_bad_display_before_any_folder(tmp_path, tc_input, fragment):
    outcome = await handle_create_artifact(_session(tmp_path), {"name": "X", "description": "d", **tc_input})
    assert outcome.ok is False
    assert fragment in outcome.content
    assert not any(tmp_path.iterdir())


async def test_update_artifact_sets_and_clears_display(tmp_path):
    session = _session(tmp_path)
    await handle_create_artifact(session, {"name": "Chart", "description": "d", "type": "html-app"})
    slug = next(tmp_path.iterdir()).name

    outcome = await handle_update_artifact_metadata(session, {"slug": slug, "display": "inline"})
    assert outcome.ok is True
    assert _metadata(tmp_path)["display"] == "inline"

    outcome = await handle_update_artifact_metadata(session, {"slug": slug, "display": ""})
    assert outcome.ok is True
    assert _metadata(tmp_path)["display"] is None


async def test_update_artifact_refuses_inline_for_a_document(tmp_path):
    session = _session(tmp_path)
    await handle_create_artifact(session, {"name": "Notes", "description": "d", "type": "document"})
    slug = next(tmp_path.iterdir()).name
    outcome = await handle_update_artifact_metadata(session, {"slug": slug, "display": "inline"})
    assert outcome.ok is False
    assert _metadata(tmp_path)["display"] is None


@pytest.mark.parametrize("tool", [
    tool_defs.CREATE_ARTIFACT_TOOL,
    tool_defs.STATIC_CREATE_ARTIFACT_TOOL,
    tool_defs.CREATE_ARTIFACT_TOOL_NO_DIRECT_REPORTS,
    tool_defs.STATIC_CREATE_ARTIFACT_TOOL_NO_DIRECT_REPORTS,
])
def test_every_create_artifact_variant_offers_display(tool):
    assert tool.input_schema["properties"]["display"]["enum"] == ["inline", "widget"]
    assert "widget" in tool.description and "Share" in tool.description
    assert "display" not in tool.input_schema["required"]


@pytest.mark.parametrize("tool", [
    tool_defs.UPDATE_ARTIFACT_METADATA_TOOL,
    tool_defs.STATIC_UPDATE_ARTIFACT_METADATA_TOOL,
])
def test_every_update_artifact_variant_can_set_and_clear_display(tool):
    assert tool.input_schema["properties"]["display"]["enum"] == ["inline", "widget", ""]
    assert "keep or share a widget" in tool.description


# ── listing ─────────────────────────────────────────────────────


def test_listing_shows_display_for_an_inline_artifact():
    assert "display" in SUMMARY_FIELDS and "display" in FIELDS
    text = render_listing([(Path("/root"), [Artifact(**_RECORD, display="inline")])], SUMMARY_FIELDS)
    assert text.splitlines()[-1] == "display: inline"


def test_listing_leaves_display_out_for_the_usual_card():
    text = render_listing([(Path("/root"), [Artifact(**_RECORD)])], FIELDS)
    assert "display" not in text


# ── saved state (collaboration surface) ─────────────────────────


def _with_state(tmp_path: Path, body: str | None, **extra) -> str:
    folder = tmp_path / _RECORD["slug"]
    folder.mkdir()
    if body is not None:
        (folder / INLINE_STATE_FILENAME).write_text(body, encoding="utf-8")
    return render_listing([(tmp_path, [Artifact(**_RECORD, **extra)])], SUMMARY_FIELDS)


def test_state_is_content_named_state_json():
    """cowork-server writes the page's setState here; it is the user's work, not housekeeping."""
    assert INLINE_STATE_FILENAME == "state.json"
    assert INLINE_STATE_FILENAME not in NON_CONTENT_NAMES
    assert "state" in FIELDS and "state" in SUMMARY_FIELDS


def test_listing_inlines_a_small_state_as_compact_json(tmp_path):
    text = _with_state(tmp_path, '{\n  "picked": ["ana", "li"]\n}', display="inline")
    assert text.splitlines()[-1] == 'state: {"picked":["ana","li"]} (state.json, the user\'s latest edits)'


def test_listing_flags_a_large_state_by_size(tmp_path):
    body = json.dumps({"picked": ["x" * 20] * 40})
    assert len(body) > STATE_INLINE_MAX_BYTES
    line = _with_state(tmp_path, body, display="inline").splitlines()[-1]
    assert line.startswith(f"state: state.json, {len(body)} bytes")
    assert "read it for the current selections" in line
    assert "xxxx" not in line


def test_listing_flags_an_unreadable_state_without_failing(tmp_path):
    line = _with_state(tmp_path, "{not json", display="inline").splitlines()[-1]
    assert line.startswith("state: state.json, 9 bytes")


def test_a_state_cannot_fake_a_listing_heading(tmp_path):
    text = _with_state(tmp_path, json.dumps({"note": "x\n## other-slug"}), display="inline")
    assert "\n## other-slug" not in text


@pytest.mark.parametrize("body, extra", [
    (None, {"display": "inline"}),          # nothing saved yet
    ('{"picked": []}', {}),                 # a usual card's own state.json is just a file
])
def test_no_state_line_without_an_inline_saved_state(tmp_path, body, extra):
    assert "state:" not in _with_state(tmp_path, body, **extra)


def test_a_linked_state_is_not_followed(tmp_path):
    secret = tmp_path / "secret.json"
    secret.write_text('{"token": "s3cret"}', encoding="utf-8")
    folder = tmp_path / _RECORD["slug"]
    folder.mkdir()
    (folder / INLINE_STATE_FILENAME).symlink_to(secret)
    text = render_listing([(tmp_path, [Artifact(**_RECORD, display="inline")])], SUMMARY_FIELDS)
    assert "s3cret" not in text and "state:" not in text


def test_the_block_teaches_the_collaboration_surface():
    for needle in ("review, pick", "edit or confirm several items", "instead of a long text list",
                   "`ask_user`", "`inspect_artifact`", "(About the artifact `<slug>`.)", "```json",
                   "update that artifact in", "same slug", "do not call", "create_artifact again",
                   "shows only the newest"):
        assert needle in IN_REPLY_VISUALS_PROMPT.replace("\n", " "), needle


def test_the_collaboration_rule_holds_with_proactive_dashboards_off():
    on_request = in_reply_visuals(proactive=False)
    assert "even when you do not build artifacts unprompted" in on_request


def test_the_list_artifacts_tool_mentions_saved_state():
    assert "state.json" in tool_defs.LIST_ARTIFACTS_TOOL.description
    assert "state" in tool_defs.LIST_ARTIFACTS_TOOL.input_schema["properties"]["fields"]["items"]["enum"]


# ── prompt ──────────────────────────────────────────────────────


@pytest.mark.parametrize("surface, expected", [
    ("desktop", True), ("web", True), ("cli", False), (None, False), ("other", False),
])
def test_chart_blocks_follow_the_surface(surface, expected):
    assert renders_chart_blocks(surface) is expected


def test_the_block_teaches_both_fences_the_caps_and_the_opt_out():
    for needle in ("```chartjs", "```mermaid", "pure JSON", "500 data points", "5 datasets",
                   "50 KB", "100 nodes", 'display="inline"', "no charts", "just the data"):
        assert needle in IN_REPLY_VISUALS_PROMPT, needle


def test_the_chartjs_example_is_pure_json():
    example = IN_REPLY_VISUALS_PROMPT.split("```chartjs\n", 1)[1].split("\n```", 1)[0]
    config = json.loads(example)
    assert config["type"] and config["data"]["datasets"]


def test_the_head_follows_proactive_dashboards():
    assert "When a visual makes the answer clearer" in in_reply_visuals(proactive=True)
    on_request = in_reply_visuals(proactive=False)
    assert "do not build artifacts unprompted" in on_request
    assert "only when the user asks" in on_request


def _build(*, proactive: bool, chart_blocks: bool, direct_generation: bool, direct_reports: bool = True) -> str:
    tools = []
    if direct_generation:
        tools = [tool_defs.GENERATE_ARTIFACT_TOOL]
    return ChatSystemPromptBuilder().build(
        conversation_started="2026-11-24T00:00:00+00:00",
        system_prompt_context=SystemPromptContext(runtime_context="test runtime"),
        proactive_dashboards=proactive, output_dir="out", tool_defs=tools,
        direct_reports=direct_reports, chart_blocks=chart_blocks,
    )


@pytest.mark.parametrize("proactive", [True, False])
@pytest.mark.parametrize("direct_generation", [True, False])
@pytest.mark.parametrize("direct_reports", [True, False])
def test_fences_reach_every_variant_on_a_cowork_surface(proactive, direct_generation, direct_reports):
    prompt = _build(proactive=proactive, chart_blocks=True,
                    direct_generation=direct_generation, direct_reports=direct_reports)
    assert "IN-REPLY VISUALS" in prompt
    assert "```chartjs" in prompt and "```mermaid" in prompt
    assert in_reply_visuals(proactive=proactive) in prompt


@pytest.mark.parametrize("proactive", [True, False])
@pytest.mark.parametrize("direct_generation", [True, False])
def test_no_fences_where_the_chat_cannot_draw_them(proactive, direct_generation):
    prompt = _build(proactive=proactive, chart_blocks=False, direct_generation=direct_generation)
    assert "```chartjs" not in prompt and "```mermaid" not in prompt
    assert "IN-REPLY VISUALS" not in prompt


def test_the_markdown_variant_no_longer_assumes_a_terminal():
    prompt = _build(proactive=False, chart_blocks=False, direct_generation=False)
    assert "CLI terminal" not in prompt
    assert "The chat is the primary display" in prompt


# ── skill ───────────────────────────────────────────────────────


def test_the_dashboard_skill_has_the_inline_contract():
    import anton

    body = (
        Path(anton.__file__).parent / "core/memory/builtin_skills/build-html-dashboard/SKILL.md"
    ).read_text(encoding="utf-8")
    inline = body.split('INLINE AND WIDGET (display="inline" or "widget")', 1)[1]
    for needle in ('data-theme="dark"', ":root[data-theme=\"dark\"]", "600 px", "100vh",
                   "NO NETWORK", "fetch", "echarts@5.5.1", "MutationObserver"):
        assert needle in inline, needle


def test_the_dashboard_skill_documents_the_page_api():
    import anton

    body = (
        Path(anton.__file__).parent / "core/memory/builtin_skills/build-html-dashboard/SKILL.md"
    ).read_text(encoding="utf-8")
    inline = body.split('INLINE AND WIDGET (display="inline" or "widget")', 1)[1]
    for needle in ("anton.setState(obj)", "32 KB", "`state.json`", "anton.getState()", "anton.onState(cb)",
                   "anton.submit(obj, message)", "if (window.anton)", "Render from the data in the page first",
                   "same slug", "a checklist in that state shape", "api.submit(", "api.onState(apply)"):
        assert needle in inline, needle
    assert "innerHTML" not in inline.split("a checklist in that state shape", 1)[1]


# ── widget (ENG-4684) ───────────────────────────────────────────


def _skill_inline_section() -> str:
    import anton

    body = (
        Path(anton.__file__).parent / "core/memory/builtin_skills/build-html-dashboard/SKILL.md"
    ).read_text(encoding="utf-8")
    return body.split('INLINE AND WIDGET (display="inline" or "widget")', 1)[1]


def test_store_writes_widget_and_turns_it_inline_in_place(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    artifact = store.create(name="Pick people", description="d", type="html-app", display="widget")
    assert json.loads(store.metadata_path(artifact.slug).read_text(encoding="utf-8"))["display"] == "widget"
    kept = store.update(artifact.slug, display="inline")
    assert (kept.slug, kept.id, kept.display) == (artifact.slug, artifact.id, "inline")


async def test_create_artifact_persists_a_widget(tmp_path):
    outcome = await handle_create_artifact(_session(tmp_path), {
        "name": "Pick people", "description": "d", "type": "html-app",
        "primary": "index.html", "display": " Widget ",
    })
    assert outcome.ok is True
    assert "as a widget" in outcome.content and "Inline section" in outcome.content
    assert _metadata(tmp_path)["display"] == "widget"


async def test_create_artifact_refuses_a_widget_for_a_document(tmp_path):
    outcome = await handle_create_artifact(_session(tmp_path), {
        "name": "Notes", "description": "d", "type": "document", "display": "widget",
    })
    assert outcome.ok is False and "only an html-app" in outcome.content
    assert not any(tmp_path.iterdir())


async def test_keeping_a_widget_is_a_metadata_change_on_the_same_slug(tmp_path):
    session = _session(tmp_path)
    await handle_create_artifact(session, {"name": "Pick", "description": "d", "type": "html-app", "display": "widget"})
    slug = next(tmp_path.iterdir()).name
    outcome = await handle_update_artifact_metadata(session, {"slug": slug, "display": "inline"})
    assert outcome.ok is True and "display=inline" in outcome.content
    assert [p.name for p in tmp_path.iterdir()] == [slug]
    assert _metadata(tmp_path)["display"] == "inline"
    outcome = await handle_update_artifact_metadata(session, {"slug": slug, "display": "widget"})
    assert outcome.ok is True and _metadata(tmp_path)["display"] == "widget"


def test_listing_shows_a_widget_and_its_state(tmp_path):
    text = _with_state(tmp_path, '{"people": [{"name": "Ana Li", "picked": true}]}', display="widget")
    lines = text.splitlines()
    assert "display: widget" in lines
    assert lines[-1] == 'state: {"people":[{"name":"Ana Li","picked":true}]} (state.json, the user\'s latest edits)'


def test_the_ladder_runs_text_chart_widget_inline_full():
    prompt = IN_REPLY_VISUALS_PROMPT
    rungs = ["1. PLAIN TEXT", "2. CHART BLOCK", "3. WIDGET", "4. INLINE ARTIFACT", "5. FULL ARTIFACT"]
    positions = [prompt.index(r) for r in rungs]
    assert positions == sorted(positions)
    widget = prompt[positions[2]:positions[3]]
    assert 'display="widget"' in widget
    assert "`ask_user`" in widget
    assert "no Share" in widget and "Artifacts list" in widget
    assert 'display="inline")`' in widget and "same slug" in widget.replace("\n", " ")
    assert 'display="inline"' in prompt[positions[3]:positions[4]]
    assert "WORKING TOGETHER IN AN INLINE ARTIFACT" not in prompt  # folded into the widget rung


def test_both_heads_point_at_rung_five():
    for proactive in (True, False):
        head = in_reply_visuals(proactive=proactive).split("\n1. PLAIN TEXT", 1)[0]
        assert "rung 5" in head and "option 3" not in head
    assert "a widget (rung 3) is not a chart and needs no request" in in_reply_visuals(proactive=False).replace("\n", " ")


def test_the_widget_rung_is_voice_friendly():
    flat = IN_REPLY_VISUALS_PROMPT.replace("\n", " ")
    for needle in ("voice call", "title, visible text and current state", "set the state or submit",
                   "self-describing JSON", '{"people":[{"name":"Ana Li","picked":true}]}', "not opaque ids",
                   "render the page fully from any state", "short and human"):
        assert needle in flat, needle


def test_the_skill_teaches_widgets_and_voice():
    inline = _skill_inline_section()
    for needle in ('display="widget"', "not a deliverable", "no Share", "Artifacts list",
                   'update_artifact(slug=..., display="inline")', "VOICE", "voice call",
                   '{"people":[{"name":"Ana Li","picked":true}]}', "Render fully from `onState`",
                   "pushed from outside at any time", "short and human"):
        assert needle in inline, needle


def test_the_skill_example_uses_the_readable_state_shape():
    example = _skill_inline_section().split("```html", 1)[1].split("```", 1)[0]
    assert "people:" in example and "name:" in example and "picked:" in example
    assert "api.onState(apply)" in example and "api.submit(" in example
    assert "Contact " in example  # a short, human submit message
    assert "innerHTML" not in example
