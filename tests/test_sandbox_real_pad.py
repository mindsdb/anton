"""A whole pad under the real lock: venv build and check, cell boot, and cells.

Needs Linux with Landlock ABI 3 or later and skips elsewhere with that reason.
Each denial check runs in a cell that also shows an allowed action working.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anton.core.backends import sandbox
from anton.core.backends.local import LocalScratchpadRuntime
from anton.core.backends.sandbox import CellSandbox

_ABI = sandbox.landlock_abi()
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        _ABI < sandbox.MIN_LANDLOCK_ABI,
        reason=f"needs Linux with Landlock ABI >= {sandbox.MIN_LANDLOCK_ABI}; this host reports {_ABI}",
    ),
]

_CELL = """
import errno, getpass, os, multiprocessing, subprocess, sys
with open("made.txt", "w") as f:
    f.write("ok")
print("cwd-write", open("made.txt").read())
for label, path in (("secret", SECRET), ("server-home", "/proc/1/environ")):
    try:
        open(path, "rb").read()
        print(label, "allowed")
    except OSError as exc:
        print(label, errno.errorcode[exc.errno])
print("token", "COWORK_AUTH_TOKEN" in os.environ, "TOKEN-VALUE" in repr(dict(os.environ)))
print("home", os.environ["HOME"] == HOME, "user", getpass.getuser())
print("devnull", subprocess.run([sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL).returncode)
with multiprocessing.get_context("fork").Pool(2) as pool:
    print("pool", sum(pool.map(abs, [-1, -2])))
import pydantic
print("host-packages", bool(pydantic.VERSION))
"""


async def test_a_pad_builds_checks_and_runs_cells_under_the_lock(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("COWORK_AUTH_TOKEN", "TOKEN-VALUE")
    project, home, tmp = tmp_path / "project", tmp_path / "home", tmp_path / "tmp"
    for folder in (project, home, tmp, tmp_path / "outside"):
        folder.mkdir()
    secret = tmp_path / "outside" / "secret.txt"
    secret.write_text("not yours")
    policy = CellSandbox(project_dir=project, home_dir=home, tmp_dir=tmp, user_name="bob")
    pad = LocalScratchpadRuntime(
        "main", coding_provider="anthropic", coding_model="", coding_api_key="", coding_base_url="",
        workspace_path=project, cell_sandbox=policy,
    )
    await pad.start()
    try:
        cell = await pad.execute(f"SECRET = {str(secret)!r}\nHOME = {str(home)!r}\n" + _CELL)
    finally:
        await pad.close()
    assert cell.error in (None, ""), cell.error
    assert cell.stdout.splitlines()[:8] == [
        "cwd-write ok",
        "secret EACCES",
        "server-home EACCES",
        "token False False",
        "home True user bob",
        "devnull 0",
        "pool 3",
        "host-packages True",
    ]
    assert (project / ".anton" / "scratchpad-venvs" / "main" / "bin" / "python").exists()
