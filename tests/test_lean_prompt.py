"""The `lean_prompt` setting: a short system prompt and short tool descriptions."""

import pytest

from anton.config.settings import AntonSettings
from anton.core.tools.tool_defs import lean_scratchpad_description


def _settings(**over) -> AntonSettings:
    return AntonSettings(_env_file=None, **over)


def test_the_setting_is_off_unless_the_env_turns_it_on(monkeypatch):
    monkeypatch.delenv("ANTON_LEAN_PROMPT", raising=False)
    assert AntonSettings(_env_file=None).lean_prompt is False
    monkeypatch.setenv("ANTON_LEAN_PROMPT", "true")
    assert AntonSettings(_env_file=None).lean_prompt is True


@pytest.mark.asyncio
async def test_a_lean_session_keeps_every_tool_and_schema_with_shorter_descriptions(make_session):
    full = make_session(settings=_settings())
    lean = make_session(settings=_settings(lean_prompt=True))
    full_tools = {t["name"]: t for t in full._build_tools()}
    lean_tools = {t["name"]: t for t in lean._build_tools()}

    assert set(lean_tools) == set(full_tools)
    assert all(lean_tools[n]["input_schema"] == full_tools[n]["input_schema"] for n in full_tools)
    assert sum(len(t["description"]) for t in lean_tools.values()) < sum(
        len(t["description"]) for t in full_tools.values()) / 2


@pytest.mark.asyncio
async def test_a_session_without_settings_keeps_the_full_prompt(make_session):
    session = make_session()
    assert not session._lean_prompt_enabled()
    assert "ARTIFACTS (" in await session._build_system_prompt("hi")


@pytest.mark.parametrize("internet_research", [True, False])
@pytest.mark.parametrize("classify", [True, False])
def test_the_lean_scratchpad_offers_only_the_helpers_the_session_has(internet_research, classify):
    text = lean_scratchpad_description(internet_research=internet_research, classify=classify)
    assert ("web_search" in text) is internet_research
    assert ("classify" in text) is classify

