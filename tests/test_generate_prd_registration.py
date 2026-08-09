"""GENERATE_PRD_TOOL registration: gated on a bound workspace, same as the
other artifact tools (create_artifact, update_artifact, ...)."""
from __future__ import annotations


def _tool_names(session) -> set[str]:
    session._build_tools()
    return {t["name"] for t in session.tool_registry.dump()}


def test_generate_prd_present_when_workspace_is_bound(make_session, tmp_path):
    from types import SimpleNamespace

    session = make_session()
    session._workspace = SimpleNamespace(artifacts_dir=tmp_path / "artifacts")
    assert "generate_prd" in _tool_names(session)


def test_generate_prd_absent_without_a_workspace(make_session):
    session = make_session()
    assert session._workspace is None
    assert "generate_prd" not in _tool_names(session)
