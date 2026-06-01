"""Verify backward compat for moved types."""


def test_import_from_new_location():
    """Types can be imported from minds.inference.types."""
    from minds.inference.types import GenericToolType, UsageBox, _messages_to_dicts

    assert UsageBox is not None
    assert _messages_to_dicts is not None
    assert GenericToolType is not None


def test_import_from_old_location():
    """Old imports still work (backward compat)."""
    from minds.agents.passthrough_agent.common import (
        GenericToolType,
        UsageBox,
        _messages_to_dicts,
    )

    assert UsageBox is not None
    assert _messages_to_dicts is not None
    assert GenericToolType is not None


def test_old_and_new_imports_are_same():
    """Both import paths point to identical objects."""
    from minds.agents.passthrough_agent.common import UsageBox as OldUsageBox
    from minds.inference.types import UsageBox as NewUsageBox

    assert NewUsageBox is OldUsageBox


def test_usage_box_functionality():
    """UsageBox works as before."""
    from minds.inference.types import UsageBox

    box = UsageBox()
    assert box.value is None
    assert box.output_payload is None
    assert box.server_artifacts == []

    box.value = (100, 50)
    assert box.value == (100, 50)


def test_generic_tool_type_enum():
    """GenericToolType enum works correctly."""
    from minds.inference.types import GenericToolType

    assert GenericToolType.WEB_SEARCH == "web_search"
    assert GenericToolType.FETCH == "fetch"


def test_messages_to_dicts_conversion():
    """_messages_to_dicts converter works correctly."""
    from minds.inference.types import _messages_to_dicts
    from minds.schemas.chat import Message, Role

    messages = [
        Message(role=Role.user, content="hello"),
        Message(role=Role.assistant, content="hi"),
    ]

    dicts = _messages_to_dicts(messages)

    assert len(dicts) == 2
    assert dicts[0]["role"] == "user"
    assert dicts[0]["content"] == "hello"
    assert dicts[1]["role"] == "assistant"
    assert dicts[1]["content"] == "hi"
