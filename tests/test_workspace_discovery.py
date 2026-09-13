"""ENG-578 slice 2: cold-start discovery — manager accessors + context block."""
from __future__ import annotations

from pathlib import Path

from anton.core.backends.local import (
    LocalScratchpadRuntime,
    default_venvs_base,
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


from anton.core.utils.scratchpad import build_workspace_discovery_context


def seed_agent_pads(mgr: ScratchpadManager, names: list[str]) -> None:
    for n in names:
        mgr.record_agent_pad(n)


class TestDiscoveryBlock:
    def test_empty_everything_renders_nothing(self, tmp_path):
        empty_root = tmp_path / "root"
        empty_root.mkdir()
        mgr = make_manager(empty_root)
        assert build_workspace_discovery_context(mgr) == ""

    def test_pads_and_root_listing(self, tmp_path):
        (tmp_path / "campaign_engine.py").write_text("x")
        (tmp_path / "BRIEFING.md").write_text("x")
        (tmp_path / "campaigns").mkdir()
        (tmp_path / ".env").write_text("PLACEHOLDER=1")  # hidden: excluded
        mgr = make_manager(tmp_path)
        seed_agent_pads(mgr, ["catanah"])
        block = build_workspace_discovery_context(mgr)
        assert "Workspace state:" in block
        assert "catanah" in block
        assert "campaign_engine.py" in block
        assert "campaigns/" in block
        assert ".env" not in block

    def test_live_pad_labeled_active_system_pads_hidden(self, tmp_path):
        mgr = make_manager(tmp_path)
        seed_agent_pads(mgr, ["catanah"])
        # Simulate live pads: one agent-recorded, one system-created.
        mgr._pads["catanah"] = object()
        mgr._pads["artifact-slug-pad"] = object()
        block = build_workspace_discovery_context(mgr)
        assert "catanah (active)" in block
        assert "artifact-slug-pad" not in block

    def test_snapshotted_pad_carries_no_age(self, tmp_path):
        # A pad with an on-disk snapshot used to render "(snapshot 4m old)".
        # That label ticked every minute and changed the prompt bytes on
        # every turn, so the block must stay byte-stable while nothing in the
        # workspace changes: bare name, no age.
        mgr = make_manager(tmp_path)
        seed_agent_pads(mgr, ["campaign"])
        path = snapshot_file(default_venvs_base(tmp_path), "conv1", "campaign")
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")
        block = build_workspace_discovery_context(mgr)
        assert "Scratchpads for this conversation: campaign —" in block
        assert "snapshot" not in block
        assert build_workspace_discovery_context(mgr) == block

    def test_caps_and_remainders(self, tmp_path):
        for i in range(35):
            (tmp_path / f"file{i:02d}.txt").write_text("x")
        mgr = make_manager(tmp_path)
        seed_agent_pads(mgr, [f"pad{i:02d}" for i in range(12)])
        block = build_workspace_discovery_context(mgr)
        assert "… and 2 more" in block   # 12 pads, cap 10
        assert "… and 5 more" in block   # 35 files, cap 30

    def test_listing_failure_renders_without_root(self, tmp_path, monkeypatch):
        import anton.core.utils.scratchpad as scratchpad_utils

        mgr = make_manager(tmp_path)
        seed_agent_pads(mgr, ["catanah"])
        monkeypatch.setattr(
            scratchpad_utils.os,
            "scandir",
            lambda path: (_ for _ in ()).throw(PermissionError()),
        )
        block = build_workspace_discovery_context(mgr)
        assert "catanah" in block
        assert "Project root" not in block


class TestVolatileTailPlacement:
    def test_workspace_context_lands_in_volatile_tail(self):
        # Cache-stability contract (ENG-1122): the block is volatile and must
        # sit after the volatile-tail marker so it never busts the cached
        # prefix. The marker is now the timestamp note (ENG-1092 moved the live
        # clock onto each message), the last cache-stable line before the tail.
        from anton.core.llm.prompt_builder import (
            ChatSystemPromptBuilder,
            SystemPromptContext,
        )

        builder = ChatSystemPromptBuilder()
        prompt = builder.build(
            conversation_started="Monday, January 05, 2026",
            system_prompt_context=SystemPromptContext(),
            proactive_dashboards=False,
            output_dir="/tmp/out",
            workspace_context="\n\nWorkspace state:\nScratchpads for this conversation: catanah",
        )
        assert "Workspace state:" in prompt
        assert prompt.index("Workspace state:") > prompt.index(
            "prefixed with the time they were sent"
        )

    def test_empty_workspace_context_adds_nothing(self):
        from anton.core.llm.prompt_builder import (
            ChatSystemPromptBuilder,
            SystemPromptContext,
        )

        builder = ChatSystemPromptBuilder()
        prompt = builder.build(
            conversation_started="Monday, January 05, 2026",
            system_prompt_context=SystemPromptContext(),
            proactive_dashboards=False,
            output_dir="/tmp/out",
        )
        assert "Workspace state:" not in prompt
