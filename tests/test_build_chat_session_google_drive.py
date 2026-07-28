"""End-to-end: build_chat_session() -> ChatSession._build_system_prompt() actually
surfaces Google Drive Picker guidance in the real assembled system prompt.

ENG-687 review (PR #241): the picked-files/OAuth guidance moved out of
build_chat_session (anton/core/runtime.py) into build_datasource_context
(anton/utils/datasources.py), which ChatSession already calls fresh every
turn. This exercises the real integration point end-to-end rather than just
the relocated function in isolation, and guards a real crash the move
uncovered: SystemPromptContext.suffix is typed str (default ""), but
build_chat_session could pass suffix=None when no system_prompt_suffix was
given, and ChatSystemPromptBuilder.build() calls suffix.strip() unconditionally.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """LocalDataVault() with no args resolves under $HOME — isolate it so this
    test never touches the real ~/.anton/data_vault or ~/.anton/memory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # build_chat_session constructs a real LLMClient; a syntactically-plausible
    # dummy key is enough since this test never calls the LLM (no turn_stream()).
    monkeypatch.setenv("ANTON_ANTHROPIC_API_KEY", "sk-ant-test-00000000000000000000000000")
    return home


@pytest.fixture()
def workspace_path(tmp_path):
    p = tmp_path / "workspace"
    p.mkdir()
    return p


async def test_picked_files_reach_the_real_system_prompt(workspace_path):
    from anton.core.datasources.data_vault import LocalDataVault
    from anton.core.runtime import build_chat_session

    vault = LocalDataVault()
    vault.save("google_drive", "work", {
        "auth_type": "oauth",
        "account_email": "user@example.com",
        "_picked_files": json.dumps([{"id": "f1", "name": "Roadmap.gdoc"}]),
    })

    session = await build_chat_session(session_id="test-picked-files", workspace_path=str(workspace_path))
    prompt = await session._build_system_prompt()

    assert "Connected Google Drive accounts are available" in prompt
    assert "IMPORTANT — additional Drive files" in prompt
    assert "Roadmap.gdoc" in prompt


async def test_no_connections_and_no_suffix_does_not_crash(workspace_path):
    """Regression: on staging, build_chat_session could pass suffix=None to
    SystemPromptContext (typed str) whenever there was nothing to add,
    crashing ChatSystemPromptBuilder.build()'s suffix.strip() call. Any
    session with zero connections and no explicit system_prompt_suffix hits
    this — verifying it explicitly rather than relying on it being masked by
    an unrelated google_drive connection existing in the vault."""
    from anton.core.runtime import build_chat_session

    session = await build_chat_session(session_id="test-empty", workspace_path=str(workspace_path))
    prompt = await session._build_system_prompt()  # must not raise

    assert "Google Drive" not in prompt


async def test_explicit_system_prompt_suffix_still_appended(workspace_path):
    from anton.core.runtime import build_chat_session

    session = await build_chat_session(
        session_id="test-suffix",
        workspace_path=str(workspace_path),
        system_prompt_suffix="Host-specific note: reply in French.",
    )
    prompt = await session._build_system_prompt()

    assert "Host-specific note: reply in French." in prompt
