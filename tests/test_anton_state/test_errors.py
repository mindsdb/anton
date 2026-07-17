from anton_state.errors import (
    ConditionalCheckFailed,
    StateError,
    StateThrottled,
    StateValidationError,
)


def test_hierarchy():
    for cls in (StateValidationError, ConditionalCheckFailed, StateThrottled):
        assert issubclass(cls, StateError)


def test_can_raise_and_message():
    try:
        raise StateThrottled("rate exceeded")
    except StateError as e:
        assert "rate exceeded" in str(e)
