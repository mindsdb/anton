"""The boundary between the scratchpad's files and yolo's.

`<name>.data.js` is produced by whatever computes it — the scratchpad —
and `<name>.schema.json` sits beside it saying what the columns are and,
critically, what global the data file defines.

These tests pin the two halves of the contract: the schema is always put
in front of the model, and the data file is never edited by it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anton.core.yolo import PatchError, Workspace, apply_patch_text
from anton.core.yolo.workspace import (
    DATA_SUFFIX,
    SCHEMA_SUFFIX,
    is_generated_data,
    is_schema,
    schema_for,
)

SCHEMA = {
    "global": "ANTON_DATA_prices",
    "file": "prices.data.js",
    "shape": "array<object>",
    "rows": 8432,
    "fields": {"date": {"type": "string"}, "price": {"type": "number", "unit": "USD"}},
    "notes": "Daily close. Gaps on weekends — do not interpolate.",
}


def artifact(tmp_path: Path) -> Workspace:
    workspace = Workspace(tmp_path)
    workspace.write("index.html", "<div id='chart'></div>\n")
    workspace.write("chart.js", "// draws the chart\n")
    # A data file far too large to ever inline.
    workspace.write("prices.data.js", "window.ANTON_DATA_prices=[" + "0," * 50000 + "];\n")
    workspace.write("prices.schema.json", json.dumps(SCHEMA, indent=2))
    return workspace


def test_the_naming_convention_classifies_files():
    assert is_generated_data("prices" + DATA_SUFFIX)
    assert not is_generated_data("chart.js")  # an ordinary .js is yolo's
    assert is_schema("prices" + SCHEMA_SUFFIX)
    assert schema_for("reports/q3.data.js") == "reports/q3.schema.json"


def test_the_map_inlines_the_schema_and_never_the_data(tmp_path: Path):
    """The whole point. A line reading `prices.data.js (2.1 MB)` tells the
    model nothing it can write code against; its sidecar tells it
    everything, for a few hundred bytes."""
    file_map = artifact(tmp_path).map()

    # The schema is there in full, including the part that actually
    # matters — the global the data file defines.
    assert "ANTON_DATA_prices" in file_map
    assert "do not interpolate" in file_map
    # The data file is listed but its contents never appear.
    assert "prices.data.js" in file_map
    assert "0,0,0,0,0" not in file_map
    # And the listing points from one to the other.
    assert "see prices.schema.json" in file_map


def test_a_data_file_with_no_sidecar_says_so(tmp_path: Path):
    """Silence would read as 'this file is unimportant'. It is not — it is
    unusable, and that is worth saying."""
    workspace = Workspace(tmp_path)
    workspace.write("orphan.data.js", "window.X=[1,2,3];\n")
    assert "no schema sidecar" in workspace.map()


def test_yolo_refuses_to_edit_generated_data(tmp_path: Path):
    """Enforced, not merely instructed. A diff against two megabytes of
    rows is not a change anyone reviewed."""
    workspace = artifact(tmp_path)
    before = workspace.read("prices.data.js")

    with pytest.raises(PatchError, match="generated data"):
        apply_patch_text(
            workspace,
            "--- a/prices.data.js\n+++ b/prices.data.js\n@@\n"
            "-window.ANTON_DATA_prices=[" + "0," * 50000 + "];\n"
            "+window.ANTON_DATA_prices=[1];\n",
        )
    assert workspace.read("prices.data.js") == before


def test_the_refusal_says_what_to_do_instead(tmp_path: Path):
    """The message is fed back to the model, so it has to be actionable."""
    workspace = artifact(tmp_path)
    with pytest.raises(PatchError) as caught:
        apply_patch_text(
            workspace,
            "*** Begin Patch\n*** Add File: new.data.js\n+window.X=[];\n*** End Patch\n",
        )
    message = str(caught.value)
    assert "let it be written again" in message
    assert SCHEMA_SUFFIX in message


def test_a_mixed_patch_writes_nothing(tmp_path: Path):
    """One legitimate file and one data file in the same patch is still a
    refusal — all files or none, as everywhere else."""
    workspace = artifact(tmp_path)
    with pytest.raises(PatchError, match="generated data"):
        apply_patch_text(
            workspace,
            "--- a/chart.js\n+++ b/chart.js\n@@\n-// draws the chart\n+// updated\n"
            "*** Begin Patch\n*** Add File: extra.data.js\n+window.Y=[];\n*** End Patch\n",
        )
    assert workspace.read("chart.js") == "// draws the chart\n"


def test_code_beside_data_is_still_editable(tmp_path: Path):
    """The convention must not make ordinary .js files untouchable."""
    workspace = artifact(tmp_path)
    written = apply_patch_text(
        workspace,
        "--- a/chart.js\n+++ b/chart.js\n@@\n-// draws the chart\n"
        "+const rows = window.ANTON_DATA_prices;\n",
    )
    assert written == {"chart.js"}
    assert "ANTON_DATA_prices" in workspace.read("chart.js")
