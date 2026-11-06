"""A cell that reads a JSON result marked as cut gets a note ahead of its output."""
from __future__ import annotations

import json

from anton.core.backends.local import LocalScratchpadRuntime

_DEFAULTS = dict(
    coding_provider="anthropic",
    coding_model="",
    coding_api_key="",
    coding_base_url="",
)


def _sum_cell(body: dict) -> str:
    return (
        "import json\n"
        f"data = json.loads({json.dumps(json.dumps(body))})\n"
        "rows = data.get('Rows') or data.get('data')\n"
        "print('total', sum(r[1] for r in rows))\n"
    )


async def test_cut_result_is_named_and_whole_result_left_alone():
    pad = LocalScratchpadRuntime(name="cut-results", **_DEFAULTS)
    await pad.start()
    try:
        cut = await pad.execute(_sum_cell({"Rows": [["a", 1], ["b", 2]], "RowCount": 2, "Truncated": True}))
        whole = await pad.execute(_sum_cell({"Rows": [["a", 1], ["b", 2]], "RowCount": 2, "Truncated": False}))
        nested = await pad.execute(_sum_cell({"data": [["a", 1]], "meta": {"next_cursor": "abc"}}))
        last_page = await pad.execute(_sum_cell({"data": [["a", 1]], "meta": {"next_cursor": None}}))
    finally:
        await pad.close()

    assert cut.error is None
    assert "Truncated" in cut.stdout.splitlines()[0]
    assert cut.stdout.endswith("total 3\n")
    # The note is per cell: the next cell starts clean.
    assert whole.stdout == "total 3\n"
    assert "next_cursor" in nested.stdout.splitlines()[0]
    assert last_page.stdout == "total 1\n"
