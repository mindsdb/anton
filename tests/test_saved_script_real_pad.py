"""A script a step writes and runs, through a real local pad.

Asked to "write and run a script", agents save the script from a literal and
run the file in the same step. The write check reads the script's text, and
the pad runs the step with that text in place of the file
(`CellWrites.run_code`), so a writer outside the step that changes the file
changes nothing that runs.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from anton.core.backends.local import LocalScratchpadRuntime
from anton.core.utils.write_check import find_cell_writes, snapshot_project_files

pytestmark = [pytest.mark.slow]


async def test_a_real_pad_runs_the_saved_script_text_its_check_read(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    script = project / "job.py"
    code = (
        "from pathlib import Path\nimport subprocess, sys\n"
        "script = '''import sys\nprint('checked', sys.argv[0].endswith('job.py'))'''\n"
        f"path = Path({str(script)!r})\npath.write_text(script)\n"
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
            script.write_text("print('raced')")

    await pad.start()
    racer = threading.Thread(target=rewrite)
    racer.start()
    try:
        pad.expect_run(code, writes.run_code)
        cell = await pad.execute(code)
    finally:
        stop.set()
        racer.join()
        await pad.close()
    assert cell.error in (None, ""), cell.error
    assert cell.stdout.strip() == "checked True"
    # The cell keeps the code as the step wrote it, for later checks to read.
    assert cell.code == code
