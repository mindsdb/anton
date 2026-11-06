"""A cell that reads a JSON result marked as cut gets a note ahead of its output."""
from __future__ import annotations

import json

import pytest

from anton.core.backends.local import LocalScratchpadRuntime

_DEFAULTS = dict(
    coding_provider="anthropic",
    coding_model="",
    coding_api_key="",
    coding_base_url="",
)


def _sum_cell(body: dict, rows_key: str) -> str:
    return (
        "import json\n"
        f"data = json.loads({json.dumps(json.dumps(body))})\n"
        f"print('total', sum(r['n'] for r in data[{rows_key!r}]))\n"
    )


_ROWS = [{"n": 1}, {"n": 2}]


@pytest.mark.parametrize(
    "body, rows_key, marker",
    [
        ({"data": _ROWS, "has_more": True}, "data", "has_more"),
        ({"value": _ROWS, "@odata.nextLink": "https://example.test/next"}, "value", "@odata.nextLink"),
        ({"items": _ROWS, "meta": {"next_cursor": "abc"}}, "items", "next_cursor"),
        ({"rows": _ROWS, "truncated": True}, "rows", "truncated"),
    ],
)
async def test_cut_result_is_named_ahead_of_the_output(body, rows_key, marker):
    pad = LocalScratchpadRuntime(name="cut-results", **_DEFAULTS)
    await pad.start()
    try:
        cut = await pad.execute(_sum_cell(body, rows_key))
        # The note is per cell: the next cell starts clean.
        whole = await pad.execute(_sum_cell({rows_key: _ROWS}, rows_key))
    finally:
        await pad.close()

    assert cut.error is None
    assert marker in cut.stdout.splitlines()[0]
    assert cut.stdout.endswith("total 3\n")
    assert whole.stdout == "total 3\n"


@pytest.mark.parametrize(
    "body",
    [
        {"data": _ROWS, "has_more": False},
        {"value": _ROWS, "@odata.nextLink": None},
        {"items": _ROWS, "meta": {"next_cursor": ""}},
    ],
)
async def test_last_page_is_left_alone(body):
    rows_key = next(k for k in body if isinstance(body[k], list))
    pad = LocalScratchpadRuntime(name="cut-results", **_DEFAULTS)
    await pad.start()
    try:
        cell = await pad.execute(_sum_cell(body, rows_key))
    finally:
        await pad.close()

    assert cell.stdout == "total 3\n"
