import pytest

from anton_state import StateUnavailable, pop_last_state_error  # exported from the package
from anton_state import errors
from anton_state.errors import (
    ConditionalCheckFailed,
    StateError,
    StateThrottled,
    StateValidationError,
)


def test_hierarchy():
    for cls in (StateValidationError, ConditionalCheckFailed, StateThrottled, StateUnavailable):
        assert issubclass(cls, StateError)


def test_state_unavailable_raises_as_state_error():
    with pytest.raises(StateError):
        raise StateUnavailable("broker down")


def test_can_raise_and_message():
    try:
        raise StateThrottled("rate exceeded")
    except StateError as e:
        assert "rate exceeded" in str(e)


# --- runner-visible outage slot (mindshub_services PR #164 item 11) ---------

@pytest.fixture(autouse=True)
def _clear_slot():
    """The slot is module-global; don't let one test's recording leak into
    the next (mirrors why artifact_runner resets it before every invoke)."""
    pop_last_state_error()
    yield
    pop_last_state_error()


def test_record_sets_slot_for_throttled():
    errors._record(StateThrottled("rate exceeded"))
    assert pop_last_state_error() == "StateThrottled"


def test_record_sets_slot_for_unavailable():
    errors._record(StateUnavailable("broker down"))
    assert pop_last_state_error() == "StateUnavailable"


def test_record_ignores_validation_error():
    """Artifact's own bad input — not a state-plane outage, must not be
    tagged as one (README's "start there" would send an operator to the
    wrong place)."""
    errors._record(StateValidationError("bad key"))
    assert pop_last_state_error() is None


def test_record_ignores_conditional_check_failed():
    """Lost an optimistic-concurrency race — normal artifact logic, not an
    outage."""
    errors._record(ConditionalCheckFailed("version mismatch"))
    assert pop_last_state_error() is None


def test_pop_clears_the_slot():
    errors._record(StateUnavailable("broker down"))
    assert pop_last_state_error() == "StateUnavailable"
    assert pop_last_state_error() is None  # second pop: already cleared


def test_pop_before_invoke_discards_stale_value():
    """Simulates artifact_runner's reset-before-invoke: a value left over from
    a previous request must not survive into the next one, even if nobody
    read it in between."""
    errors._record(StateUnavailable("stale from a previous request"))
    pop_last_state_error()  # runner's reset, discarding the return value
    assert pop_last_state_error() is None
