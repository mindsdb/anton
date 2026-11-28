"""`ChatSessionConfig.app_backends_reachable`: False where the user's browser
cannot open a backend the session starts (cloud turns), so full-stack apps
reach the user only through their published preview."""
from __future__ import annotations

from types import SimpleNamespace

from anton.core.artifacts import ArtifactStore
from anton.core.artifacts.backend_launcher import app_backends_reachable
from anton.core.settings import CoreSettings
from anton.core.tools.tool_handlers import handle_update_artifact_metadata
from anton.workspace import Workspace


def test_the_flag_defaults_to_reachable(make_session):
    assert make_session()._app_backends_reachable is True
    assert make_session(app_backends_reachable=False)._app_backends_reachable is False


def test_the_reader_treats_a_session_without_the_attribute_as_reachable():
    assert app_backends_reachable(SimpleNamespace()) is True
    assert app_backends_reachable(SimpleNamespace(_app_backends_reachable=True)) is True
    assert app_backends_reachable(SimpleNamespace(_app_backends_reachable=False)) is False


def _update_tool(session):
    session._build_tools()
    return next(t for t in session.tool_registry.get_tool_defs() if t.name == "update_artifact")


def test_update_artifact_offers_no_port_where_backends_are_unreachable(make_session, tmp_path):
    unreachable = make_session(workspace=Workspace(tmp_path), app_backends_reachable=False)
    reachable = make_session(workspace=Workspace(tmp_path))

    assert "port" not in _update_tool(unreachable).input_schema["properties"]
    assert "port" in _update_tool(reachable).input_schema["properties"]


async def test_update_artifact_refuses_a_port_where_backends_are_unreachable(tmp_path):
    """Providers do not always drop input fields the schema leaves out."""
    store = ArtifactStore(tmp_path / "artifacts")
    slug = store.create(name="Orders", description="d", type="fullstack-stateless-app").slug
    session = SimpleNamespace(
        _workspace=SimpleNamespace(artifacts_dir=tmp_path / "artifacts"), _app_backends_reachable=False,
    )

    outcome = await handle_update_artifact_metadata(session, {"slug": slug, "port": 5000})

    assert outcome.ok is False and outcome.reason == "port_not_allowed"
    assert store.open(slug).port is None


def test_an_allowlisted_editor_is_withheld_when_the_editor_is_off(make_session, tmp_path, monkeypatch):
    """An allowlist name that matches no registered tool is fatal; the editor
    setting decides whether core registers edit_artifact at all."""
    monkeypatch.delenv("ANTON_ARTIFACT_EDITOR", raising=False)
    session = make_session(
        workspace=Workspace(tmp_path), settings=CoreSettings(artifact_editor="off"),
        tool_allowlist=frozenset({"scratchpad", "edit_artifact"}),
    )

    session._build_tools()

    assert "edit_artifact" not in {t.name for t in session.tool_registry.get_tool_defs()}
