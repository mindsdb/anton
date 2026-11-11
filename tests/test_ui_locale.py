"""The user's UI language reaches the system prompt as one line (ENG-4149)."""
import json

import pytest

from anton.cloud_turn.contract import TurnRequestV1
from anton.core.llm.prompt_builder import (
    SESSION_CONTEXT_MARKER,
    ChatSystemPromptBuilder,
    SystemPromptContext,
)
from anton.core.llm.ui_locale import (
    UI_LOCALE_NAMES,
    ui_locale_line,
    validated_ui_locale,
)
from anton.core.session import ChatSession, ChatSessionConfig
from tests.conftest import make_mock_llm


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("en", "en"),
        ("es", "es"),
        ("pt-BR", "pt-BR"),
        (" PT-br ", "pt-BR"),
        ("ES", "es"),
        ("ja", "ja"),
        ("JA", "ja"),
        ("ja-JP", None),
        (None, None),
        ("", None),
        ("   ", None),
        ("fr", "fr"),
        ("DE", "de"),
        ("ko", "ko"),
        ("it", None),
        ("pt", None),
        ("es-MX", None),
        ("en-US,en;q=0.9", None),
        (42, None),
        (["es"], None),
    ],
)
def test_validated_ui_locale(value, expected):
    assert validated_ui_locale(value) == expected


@pytest.mark.parametrize("tag", sorted(UI_LOCALE_NAMES))
def test_line_names_the_language_and_the_fallback_rule(tag):
    line = ui_locale_line(tag)
    name = UI_LOCALE_NAMES[tag]
    assert f"The user's app is set to {name} ({tag})." in line
    assert "Reply in the language the user writes in" in line
    assert f"reply in {name}." in line


@pytest.mark.parametrize("value", [None, "", "it", "xx-YY"])
def test_no_line_without_a_supported_locale(value):
    assert ui_locale_line(value) == ""


def _build(ui_locale_context: str = "", suffix: str = "") -> str:
    return ChatSystemPromptBuilder().build(
        conversation_started="Tuesday, November 10, 2026",
        system_prompt_context=SystemPromptContext(runtime_context="test runtime", suffix=suffix),
        proactive_dashboards=False,
        output_dir="",
        memory_context="\n\nVOLATILE MEMORY",
        ui_locale_context=ui_locale_context,
    )


def test_builder_without_a_locale_is_unchanged():
    without = ChatSystemPromptBuilder().build(
        conversation_started="Tuesday, November 10, 2026",
        system_prompt_context=SystemPromptContext(runtime_context="test runtime", suffix="HOST SUFFIX"),
        proactive_dashboards=False,
        output_dir="",
        memory_context="\n\nVOLATILE MEMORY",
    )
    assert _build(suffix="HOST SUFFIX") == without
    assert "## Language" not in without


def test_builder_puts_the_line_in_the_session_part_before_the_volatile_tail():
    line = ui_locale_line("es")
    prompt = _build(line, suffix="HOST SUFFIX")
    shared, session = prompt.split(SESSION_CONTEXT_MARKER)
    # Per user, so the prefix every session shares stays one cache entry.
    assert line not in shared
    assert session.index("HOST SUFFIX") < session.index(line) < session.index("VOLATILE MEMORY")
    assert prompt.count("## Language") == 1


async def test_session_prompt_carries_the_configured_locale():
    session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm(), ui_locale="pt-BR"))
    first = await session._build_system_prompt("oi")
    assert ui_locale_line("pt-BR") in first
    # Stable across turns, so it never busts the cached prefix.
    assert await session._build_system_prompt("e agora?") == first


@pytest.mark.parametrize("value", [None, "it", "  "])
async def test_session_prompt_without_a_supported_locale_matches_no_locale(value):
    plain = await ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))._build_system_prompt("hi")
    other = await ChatSession(ChatSessionConfig(llm_client=make_mock_llm(), ui_locale=value))._build_system_prompt("hi")
    assert other == plain
    assert "## Language" not in plain


def _request(**extra) -> TurnRequestV1:
    body = {"protocol_version": 1, "conversation_id": "c", "input": "hi", **extra}
    return TurnRequestV1.from_json(json.dumps(body))


def test_cloud_request_reads_ui_locale():
    assert _request(ui_locale="es").ui_locale == "es"


@pytest.mark.parametrize("value", [None, 1, ["es"], {"tag": "es"}])
def test_cloud_request_drops_a_non_string_ui_locale(value):
    assert _request(ui_locale=value).ui_locale is None


def test_cloud_request_without_ui_locale():
    """A controller that does not forward the field changes nothing."""
    assert _request().ui_locale is None
