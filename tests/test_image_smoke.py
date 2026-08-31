"""The scratchpad image's build-time smoke (`docker/image_smoke.py`).

That script is a build gate, so its own failure mode matters: a smoke that
reports success no matter what is worse than no smoke, because the build stays
green while claiming the image was exercised. These pin the ways it could go
quietly blind.

The script lives outside the package (it runs inside the image, against the
image's venv), so it is loaded by path rather than imported.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SMOKE_PATH = Path(__file__).resolve().parents[1] / "docker" / "image_smoke.py"


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location("image_smoke", SMOKE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _completed(returncode: int, stdout: bytes = b"", stderr: bytes = b""):
    return subprocess.CompletedProcess(args=["python"], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


def test_cloud_turn_passes_on_one_terminal_event(smoke, monkeypatch):
    line = json.dumps({"kind": "turn_failed", "error": "JSONDecodeError: ..."}).encode()
    monkeypatch.setattr(smoke, "run_module", lambda _: _completed(0, line + b"\n"))
    assert smoke.check_cloud_turn() == []


def test_cloud_turn_fails_when_the_entrypoint_dies(smoke, monkeypatch):
    """The image that cannot start at all — the shape of the 2026-08-31 pods."""
    monkeypatch.setattr(
        smoke, "run_module",
        lambda _: _completed(1, b"", b"ModuleNotFoundError: No module named 'dill'"),
    )
    (failure,) = smoke.check_cloud_turn()
    assert "exited 1" in failure
    # The reason has to survive into the build log, or the gate just says "no".
    assert "dill" in failure


def test_cloud_turn_fails_on_a_silent_exit(smoke, monkeypatch):
    """Exit 0 with nothing on stdout is the worst case, not a pass.

    The controller reads the terminal off the event, so a turn that emits none
    hangs until the stall timer fires rather than failing.
    """
    monkeypatch.setattr(smoke, "run_module", lambda _: _completed(0, b""))
    (failure,) = smoke.check_cloud_turn()
    assert "expected 1 protocol line" in failure


def test_cloud_turn_fails_when_something_else_writes_to_stdout(smoke, monkeypatch):
    """stdout is the protocol wire. A stray line corrupts the stream the
    controller parses, which is why the entrypoint isolates FD 1 at all."""
    noise = b'starting up\n{"kind": "turn_failed", "error": "x"}\n'
    monkeypatch.setattr(smoke, "run_module", lambda _: _completed(0, noise))
    (failure,) = smoke.check_cloud_turn()
    assert "expected 1 protocol line" in failure


def test_cloud_turn_fails_on_a_non_terminal_event(smoke, monkeypatch):
    monkeypatch.setattr(
        smoke, "run_module",
        lambda _: _completed(0, json.dumps({"kind": "delta", "text": "hi"}).encode()),
    )
    (failure,) = smoke.check_cloud_turn()
    assert "turn_failed" in failure


def test_a_check_that_raises_is_reported_not_swallowed(smoke):
    """A crashed check must not read as a passed check.

    Without this the script would exit non-zero from the traceback and the build
    would fail for the right reason by accident — until someone wrapped the call.
    """
    def explodes():
        raise RuntimeError("boom")

    (failure,) = smoke.run_checks(checks=(explodes,))
    assert "explodes" in failure
    assert "RuntimeError: boom" in failure


def test_one_broken_check_does_not_hide_the_others(smoke):
    def explodes():
        raise RuntimeError("boom")

    def reports():
        return ["venv: not writable"]

    failures = smoke.run_checks(checks=(explodes, reports))
    assert len(failures) == 2


def test_main_reports_success_only_when_nothing_failed(smoke, monkeypatch):
    monkeypatch.setattr(smoke, "run_checks", lambda: [])
    assert smoke.main() == 0
    monkeypatch.setattr(smoke, "run_checks", lambda: ["cloud_turn: exited 1"])
    assert smoke.main() == 1


def test_the_dockerfile_runs_the_smoke_below_the_runtime_user(smoke):
    """The layer's position is the whole premise: above `USER 1000` every check
    runs as root and passes for an image the pod still cannot use."""
    dockerfile = (SMOKE_PATH.parents[1] / "Dockerfile").read_text().splitlines()
    user_line = next(i for i, ln in enumerate(dockerfile) if ln.strip() == "USER 1000")
    smoke_line = next(i for i, ln in enumerate(dockerfile) if "image_smoke.py" in ln)
    assert smoke_line > user_line
