from anton.core.backends.base import Cell
from anton.core.backends.wire import TRUNCATION_PREFIX, truncation_note
from anton.core.utils.scratchpad import format_cell_result


def test_long_output_is_cut_once_with_its_total():
    result = format_cell_result(Cell(code="x", stdout="a" * 12_001, stderr="", error=None))

    assert result.count(TRUNCATION_PREFIX) == 1
    assert "12001 chars total" in result


def test_output_already_cut_keeps_its_note_after_scrubbing_shortened_it():
    """Scrubbing can leave the note starting before the limit and ending after it."""
    stdout = "a" * 9_900 + truncation_note(30_000) + "\n(cwd is the project folder)"

    result = format_cell_result(Cell(code="x", stdout=stdout, stderr="", error=None))

    assert result.count(TRUNCATION_PREFIX) == 1
    assert "30000 chars total" in result
    assert "(cwd is the project folder)" in result
