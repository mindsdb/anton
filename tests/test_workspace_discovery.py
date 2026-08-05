"""ENG-578 slice 2: cold-start discovery — manager accessors + context block."""
from __future__ import annotations

import time
from pathlib import Path

from anton.core.backends.local import (
    LocalScratchpadRuntime,
    local_scratchpad_runtime_factory,
    snapshot_file,
)
from anton.core.backends.manager import ScratchpadManager

_MANAGER_DEFAULTS = dict(
    runtime_factory=local_scratchpad_runtime_factory,
    coding_provider="anthropic",
    coding_model="",
    coding_api_key="",
    coding_base_url="",
)


def make_manager(tmp_path: Path | None = None, session_id: str | None = "conv1") -> ScratchpadManager:
    return ScratchpadManager(
        **_MANAGER_DEFAULTS,
        workspace_path=tmp_path,
        session_id=session_id,
    )


class TestManagerAccessors:
    def test_workspace_path_exposed(self, tmp_path):
        assert make_manager(tmp_path).workspace_path == tmp_path

    def test_snapshot_mtime_none_when_unscoped(self, tmp_path):
        mgr = make_manager(tmp_path, session_id=None)
        assert mgr.pad_snapshot_mtime("anything") is None

    def test_snapshot_mtime_none_when_missing(self, tmp_path):
        assert make_manager(tmp_path).pad_snapshot_mtime("ghost") is None

    def test_snapshot_mtime_reads_existing_snapshot(self, tmp_path):
        # Compose the snapshot path exactly the way the runtime does, write a
        # stub snapshot, and confirm the mtime comes back.
        from anton.core.backends.local import default_venvs_base

        mgr = make_manager(tmp_path)
        path = snapshot_file(default_venvs_base(tmp_path), "conv1", "campaign")
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")
        got = mgr.pad_snapshot_mtime("campaign")
        assert got is not None and abs(got - time.time()) < 60
