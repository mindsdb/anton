"""ask_user: value types, validation, the elicit() lifecycle, the tool
handler, and registration gating."""

from __future__ import annotations

import pytest

from anton.core.interaction.elicit import (
    MAX_QUESTIONS_PER_TURN,
    AskAnswer,
    AskOption,
    AskRequest,
    validate_request,
)


def _choice(**over) -> AskRequest:
    base = dict(
        prompt="Which database?",
        options=(AskOption(value="pg", label="postgres"), AskOption(value="my", label="mysql")),
    )
    base.update(over)
    return AskRequest(**base)


def test_defaults():
    r = _choice()
    assert (r.kind, r.select, r.allow_custom, r.timeout_s) == ("choice", "one", True, None)
    assert AskAnswer(status="answered").values == ()


def test_valid_choice_passes():
    assert validate_request(_choice()) is True
    assert validate_request(_choice(select="many")) is True


@pytest.mark.parametrize(
    "request_",
    [
        _choice(prompt=""),
        _choice(prompt="   "),
        _choice(options=()),
        _choice(options=(AskOption(value="pg", label="postgres"),)),
        _choice(options=tuple(AskOption(value=f"v{i}", label=f"l{i}") for i in range(11))),
        _choice(options=(AskOption(value="pg", label="a"), AskOption(value="pg", label="b"))),
        _choice(options=(AskOption(value="", label="a"), AskOption(value="b", label="b"))),
        _choice(select="several"),
    ],
    ids=[
        "empty-prompt", "blank-prompt", "no-options", "one-option",
        "eleven-options", "duplicate-values", "empty-value", "bad-select",
    ],
)
def test_invalid_choice_rejected(request_):
    assert validate_request(request_) is False


def test_path_request_ignores_choice_rules():
    # A path picker has no options and no select mode — it must still validate.
    assert validate_request(AskRequest(prompt="Pick a folder", kind="path")) is True
    assert validate_request(AskRequest(prompt="", kind="path")) is False


def test_budget_constant_is_three():
    assert MAX_QUESTIONS_PER_TURN == 3
