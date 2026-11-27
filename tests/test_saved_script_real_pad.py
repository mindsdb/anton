"""A script a step writes and runs, through a real local pad.

Asked to "write and run a script", agents save the script from a literal and
run the file in the same step. The write check reads the script's text, and
the pad runs the step through a prelude that runs the file only while it
holds that text (`CellWrites.run_code`), so a writer outside the step that
changes the file stops the run, and its text never runs. Run as written, the
same step runs whatever the file holds.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from anton.core.backends.local import LocalScratchpadRuntime
from anton.core.utils.write_check import find_cell_writes, snapshot_project_files

pytestmark = [pytest.mark.slow]

_TEXT = "import sys\nprint('checked', sys.argv[0].endswith('job.py'))"
# What a swapped run prints when the prelude stops it because the file
# changed after the check read it (`_SCRIPT_PRELUDE`).
_CHANGED = "did not run: it changed after the write check read it"


@pytest.mark.parametrize("folder, writer, swapped", [
    ("", None, True),
    ("", "racing", True),
    ("", "after-the-save", True),
    ("", "after-the-save", False),
    ("tools", None, True),
], ids=["alone", "while-another-writer-changes-the-file", "changed-between-the-save-and-the-run",
        "as-written-changed-between-the-save-and-the-run", "in-a-subfolder"])
async def test_a_real_pad_runs_the_saved_script_text_its_check_read(
    tmp_path: Path, folder: str, writer: str | None, swapped: bool,
):
    """With the swap, the step prints what the text the check read prints,
    for a script in a subfolder too, or the prelude stops the run when
    another writer changed the file. Without the swap, the step runs the
    file as it is, so it prints what the other writer put there. The
    `after-the-save` writer changes the file once, while the step waits
    between its save and its run."""
    project = tmp_path / "project"
    (project / folder).mkdir(parents=True)
    script = project / folder / "job.py"
    wait = "time.sleep(1)\n" if writer == "after-the-save" else ""
    code = (
        "from pathlib import Path\nimport subprocess, sys, time\n"
        f"script = '''{_TEXT}'''\n"
        f"path = Path({str(script)!r})\npath.write_text(script)\n{wait}"
        "r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)\n"
        "print(r.stdout, r.stderr, end='')\n"
    )
    snap = snapshot_project_files(root=project, label="project", taken_at=time.time() + 60)
    writes = find_cell_writes(code=code, earlier=(), snapshots=(snap,), swaps=True)
    assert not writes.needs_approval() and writes.run_code is not None
    pad = LocalScratchpadRuntime(
        "main", coding_provider="anthropic", coding_model="", coding_api_key="", coding_base_url="",
        workspace_path=project, _venvs_base=tmp_path / "venvs",
    )
    assert pad.runs_checked_code
    stop = threading.Event()

    def rewrite() -> None:
        while not stop.is_set():
            if writer == "racing":
                script.write_text("print('raced')")
                continue
            # Once the step saved its text, change the file before its run.
            if script.exists() and script.read_text() == _TEXT:
                script.write_text("print('raced')")
                return
            time.sleep(0.001)

    await pad.start()
    racer = threading.Thread(target=rewrite)
    if writer:
        racer.start()
    try:
        if swapped:
            pad.expect_run(code, writes.run_code)
        cell = await pad.execute(code)
    finally:
        stop.set()
        if writer:
            racer.join()
        await pad.close()
    assert cell.error in (None, ""), cell.error
    out = cell.stdout.strip()
    if writer == "racing":
        assert out == "checked True" or out.endswith(_CHANGED), out
    elif writer == "after-the-save":
        assert out.endswith(_CHANGED) if swapped else out == "raced", out
    else:
        assert out == "checked True", out
    # The cell keeps the code as the step wrote it, for later checks to read.
    assert cell.code == code
