from __future__ import annotations

import asyncio
import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.backends.base import Cell
from anton.core.memory.cerebellum import Cerebellum, _DiffPassResult, _LessonDraft
from anton.core.memory.base import MemoryRead
from anton.core.memory.cortex import Cortex, _CompactionResult
from anton.core.memory.hippocampus import Engram, Hippocampus
from anton.core.session import ChatSession
from anton.core.tools.tool_handlers import handle_memorize


@pytest.fixture()
def dirs(tmp_path):
    g = tmp_path / "global"
    p = tmp_path / "project"
    g.mkdir()
    p.mkdir()
    return g, p


@pytest.fixture()
def cortex(dirs):
    g, p = dirs
    return Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="copilot")


@pytest.fixture()
def read_only_dir(tmp_path):
    """Build folders the way an image ships one: 0o444 files in a 0o555 folder.

    Yields a factory taking a folder name and `{file name: content}`. Modes are
    restored at teardown so tmp_path can be removed.
    """
    if sys.platform == "win32" or os.geteuid() == 0:
        pytest.skip("file modes don't stop this user writing, so read-only proves nothing")
    made: list[Path] = []

    def make(name: str, files: dict[str, str]) -> Path:
        folder = tmp_path / name
        folder.mkdir()
        for file_name, content in files.items():
            path = folder / file_name
            path.write_text(content, encoding="utf-8")
            path.chmod(0o444)
        folder.chmod(0o555)
        made.append(folder)
        return folder

    yield make
    for folder in made:
        folder.chmod(0o755)


def _snapshot(folder: Path) -> dict[str, bytes]:
    """Every file in `folder` by name, so a new file (a `.tmp`) shows too."""
    return {path.name: path.read_bytes() for path in sorted(folder.iterdir())}


class TestBuildMemoryContext:
    async def test_empty_returns_empty(self, cortex):
        assert await cortex.build_memory_context() == ""

    async def test_includes_identity(self, cortex, dirs):
        g, _ = dirs
        Hippocampus(g).rewrite_identity(["Name: Jorge", "TZ: PST"])
        result = await cortex.build_memory_context()
        assert "Identity" in result
        assert "Name: Jorge" in result

    async def test_includes_global_rules(self, cortex, dirs):
        g, _ = dirs
        Hippocampus(g).encode_rule("Use httpx", kind="always", confidence="high", source="user")
        result = await cortex.build_memory_context()
        assert "Global Rules" in result
        assert "Use httpx" in result

    async def test_includes_project_rules(self, cortex, dirs):
        _, p = dirs
        Hippocampus(p).encode_rule("Use Django ORM", kind="always", confidence="high", source="user")
        result = await cortex.build_memory_context()
        assert "Project Rules" in result
        assert "Use Django ORM" in result

    async def test_includes_lessons(self, cortex, dirs):
        g, p = dirs
        Hippocampus(g).encode_lesson("Global fact")
        Hippocampus(p).encode_lesson("Project fact")
        result = await cortex.build_memory_context()
        assert "Global Lessons" in result
        assert "Project Lessons" in result
        assert "Global fact" in result
        assert "Project fact" in result

    async def test_excludes_scratchpad_when_rules(self, cortex, dirs):
        """Scratchpad-related "when" rules already surface via the scratchpad
        tool description (get_scratchpad_context) — showing them here too
        would double their token cost, so they're excluded from the system
        prompt. Unrelated "when" rules are unaffected."""
        g, _ = dirs
        hc = Hippocampus(g)
        hc.encode_rule(
            "If a scratchpad API is paginated, use progress()",
            kind="when", confidence="high", source="user",
        )
        hc.encode_rule(
            "If the user writes in Spanish, respond in Spanish",
            kind="when", confidence="high", source="user",
        )
        result = await cortex.build_memory_context()
        assert "paginated" not in result
        assert "Spanish" in result

    async def test_excludes_scratchpad_lessons(self, cortex, dirs):
        """Same exclusion as above, for lessons."""
        g, _ = dirs
        hc = Hippocampus(g)
        hc.encode_lesson("Scratchpad cells timeout at 30s")
        hc.encode_lesson("CoinGecko rate-limits at 50/min")
        result = await cortex.build_memory_context()
        assert "timeout at 30s" not in result
        assert "rate-limits" in result

    async def test_global_and_project_rule_filters_run_concurrently(self, cortex, dirs):
        for d in dirs:
            lines = "\n".join(f"- {e.text}" for e in _when_engrams(12))
            (d / "rules.md").write_text(f"## When\n{lines}\n")

        in_flight = peak = 0

        class _SlowLLM:
            async def code(self, **kwargs):
                from anton.core.llm.provider import LLMResponse, Usage

                nonlocal in_flight, peak
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1
                # Empty response keeps every rule.
                return LLMResponse(content="", usage=Usage())

        cortex._llm = _SlowLLM()
        result = await cortex.build_memory_context("do the august campaign")
        assert peak == 2
        assert result.index("Global Rules") < result.index("Project Rules")


class TestGetScratchpadContext:
    def test_empty_returns_empty(self, cortex):
        assert cortex.get_scratchpad_context() == ""

    def test_combines_scopes(self, cortex, dirs):
        g, p = dirs
        (g / "rules.md").write_text(
            "# Rules\n\n## Always\n\n## Never\n\n## When\n- If scratchpad is slow → batch\n"
        )
        (p / "lessons.md").write_text("# Lessons\n- Scratchpad times out at 30s\n")
        result = cortex.get_scratchpad_context()
        assert "slow" in result
        assert "Scratchpad times out" in result


class TestMemoryRead:
    """What the latest prompt and scratchpad description held from the learned
    tiers."""

    def test_none_before_the_memory_is_built(self, cortex):
        assert cortex.memory_read is None

    async def test_counts_what_the_prompt_and_the_scratchpad_hold(self, cortex, dirs):
        g, p = dirs
        global_hc, project_hc = Hippocampus(g), Hippocampus(p)
        global_hc.rewrite_identity(["Name: Ana", "TZ: EST"])
        global_hc.encode_rule("Use httpx", kind="always", confidence="high", source="user")
        project_hc.encode_rule("Use Django ORM", kind="always", confidence="high", source="user")
        global_hc.encode_lesson("Global fact")
        project_hc.encode_lesson("Project fact one")
        project_hc.encode_lesson("Project fact two")
        global_hc.encode_rule(
            "If a scratchpad API is paginated, use progress()",
            kind="when", confidence="high", source="user",
        )
        project_hc.encode_lesson("Scratchpad cells time out at 30s")

        # ChatSession builds its tools, and so the scratchpad description,
        # before the system prompt.
        cortex.get_scratchpad_context()
        await cortex.build_memory_context()

        assert cortex.memory_read == MemoryRead(
            learned=True, rules=2, lessons=3, profile_lines=2, scratchpad=2
        )
        assert cortex.memory_read.entries == 9

    async def test_counts_only_the_rules_the_prompt_shows(self, cortex, dirs):
        g, _ = dirs
        (g / "rules.md").write_text("## Preferences\n- Prefer short answers\n\n## Always\n- Use httpx\n")

        prompt = await cortex.build_memory_context()

        assert "Use httpx" in prompt and "Prefer short answers" not in prompt
        assert cortex.memory_read.rules == 1

    async def test_counts_the_profile_lines_the_prompt_shows(self, cortex, dirs, monkeypatch):
        g, _ = dirs
        hc = Hippocampus(g)
        hc.rewrite_identity(["Name: Ana"])
        filter_rules = cortex._retrieve_relevant_rules

        async def filter_while_another_turn_learns(engrams, user_message):
            hc.rewrite_identity(["Name: Ana", "TZ: EST"])
            return await filter_rules(engrams, user_message)

        monkeypatch.setattr(cortex, "_retrieve_relevant_rules", filter_while_another_turn_learns)

        prompt = await cortex.build_memory_context()

        assert "Name: Ana" in prompt and "TZ: EST" not in prompt
        assert cortex.memory_read.profile_lines == 1

    async def test_counts_only_the_lessons_the_budget_lets_in(self, cortex, dirs):
        g, _ = dirs
        hc = Hippocampus(g)
        for i in range(60):
            hc.encode_lesson(f"Lesson {i:02d}: {'detail ' * 20}")

        prompt = await cortex.build_memory_context()

        shown = sum(1 for i in range(60) if f"Lesson {i:02d}:" in prompt)
        assert 0 < shown < 60
        assert cortex.memory_read.lessons == shown

    async def test_counts_the_rules_the_filter_keeps(self, cortex, dirs, captured_events):
        g, _ = dirs
        rules = _when_engrams(12)
        lines = "\n".join(f"- {e.text}" for e in rules)
        (g / "rules.md").write_text(f"## When\n{lines}\n")
        cortex._llm = _FakeLLM(content="\n".join(f"- {rules[i].text}" for i in (0, 1)))

        await cortex.build_memory_context("please do the august campaign")

        assert cortex.memory_read.rules == 2

    async def test_the_operators_layer_is_not_counted(self, dirs, tmp_path):
        g, p = dirs
        managed = tmp_path / "managed"
        managed.mkdir()
        Hippocampus(managed).encode_rule("Always cite the source", kind="always", confidence="high", source="user")
        Hippocampus(managed).encode_lesson("Orders ship on Mondays")
        Hippocampus(managed).encode_lesson("Scratchpad cells have pandas installed")
        cortex = Cortex(
            global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="copilot",
            managed_hc=Hippocampus(managed),
        )

        prompt = await cortex.build_memory_context()
        wisdom = cortex.get_scratchpad_context()

        assert "Always cite the source" in prompt and "Orders ship on Mondays" in prompt
        assert "Scratchpad cells have pandas installed" in wisdom
        assert cortex.memory_read == MemoryRead(learned=True)

    @pytest.mark.parametrize("scope", ["off", "unavailable"])
    async def test_a_turn_with_learned_memory_off_reads_none(self, cortex, dirs, scope):
        g, p = dirs
        Hippocampus(g).encode_rule("Use httpx", kind="always", confidence="high", source="user")
        Hippocampus(p).encode_lesson("Scratchpad cells time out at 30s")

        await cortex.build_memory_context(learned_memory_scope=scope)
        cortex.get_scratchpad_context(learned_memory_scope=scope)

        assert cortex.memory_read == MemoryRead(learned=False)


class TestEncode:
    async def test_encode_rule_to_project(self, cortex, dirs):
        _, p = dirs
        engram = Engram(text="Use httpx", kind="always", scope="project", confidence="high")
        actions = await cortex.encode([engram])
        assert any("always rule" in a.lower() for a in actions)
        assert (p / "rules.md").exists()

    async def test_encode_lesson_to_global(self, cortex, dirs):
        g, _ = dirs
        engram = Engram(text="CoinGecko rate limit", kind="lesson", scope="global", topic="api")
        actions = await cortex.encode([engram])
        assert any("lesson" in a.lower() for a in actions)
        assert (g / "lessons.md").exists()

    async def test_encode_profile(self, cortex, dirs):
        g, _ = dirs
        engram = Engram(text="Name: Jorge", kind="profile", scope="global", confidence="high")
        actions = await cortex.encode([engram])
        assert any("identity" in a.lower() for a in actions)
        assert (g / "profile.md").exists()
        assert "Name: Jorge" in (g / "profile.md").read_text()

    async def test_encode_profile_with_project_scope_routes_to_global(self, cortex, dirs):
        g, p = dirs
        engram = Engram(text="Name: Jorge", kind="profile", scope="project")
        await cortex.encode([engram])
        assert (g / "profile.md").exists()
        assert "Name: Jorge" in (g / "profile.md").read_text()
        assert not (p / "profile.md").exists()

    async def test_off_mode_returns_disabled(self, dirs):
        g, p = dirs
        cortex = Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="off")
        engram = Engram(text="test", kind="lesson", scope="global")
        actions = await cortex.encode([engram])
        assert any("disabled" in a.lower() for a in actions)


class TestOrphanedIdentityMigration:
    def test_migrates_project_identity_to_global_on_init(self, dirs):
        g, p = dirs
        # Simulate orphaned state from the old bug: identity entries in project scope.
        Hippocampus(p).rewrite_identity(["Name: Jorge", "TZ: PST"])
        assert (p / "profile.md").exists()

        Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="copilot")

        assert (g / "profile.md").exists()
        merged = (g / "profile.md").read_text()
        assert "Name: Jorge" in merged
        assert "TZ: PST" in merged
        assert not (p / "profile.md").exists()

    def test_migration_does_not_overwrite_fresh_global_entries(self, dirs):
        # Orphaned project data is likely stale (old bug wrote it, user may
        # have since corrected to global). Global must win on key conflicts.
        g, p = dirs
        Hippocampus(g).rewrite_identity(["Name: Alejandro"])
        Hippocampus(p).rewrite_identity(["Name: Alec", "TZ: PST"])

        Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="copilot")

        merged = (g / "profile.md").read_text()
        assert "Name: Alejandro" in merged
        assert "Name: Alec" not in merged
        assert "TZ: PST" in merged  # non-conflicting keys still migrate
        assert not (p / "profile.md").exists()

    def test_migration_noop_when_project_identity_empty(self, dirs):
        g, p = dirs
        Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="copilot")
        assert not (g / "profile.md").exists()
        assert not (p / "profile.md").exists()

    async def test_a_read_only_global_tier_keeps_the_project_lines(
        self, dirs, read_only_dir, caplog
    ):
        """A host builds a Cortex every turn. If the move raised, every turn in
        that project would fail, and the line that caused it would never go."""
        _, p = dirs
        g = read_only_dir("read-only-global", {})
        Hippocampus(p).rewrite_identity(["TZ: PST"])

        logger_name = "anton.core.memory.cortex"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            cortex = Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="autopilot")

        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.name == logger_name and r.levelno == logging.WARNING
        ]
        assert warnings == ["memory: project profile not moved (PermissionError)"]
        assert "TZ: PST" in (p / "profile.md").read_text()
        assert _snapshot(g) == {}
        # The turn still gets its memory.
        assert await cortex.build_memory_context() == ""


class TestEncodingGate:
    def test_autopilot_never_confirms(self, dirs):
        g, p = dirs
        cortex = Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="autopilot")
        engram = Engram(text="test", kind="lesson", scope="global", confidence="low")
        assert cortex.encoding_gate(engram) is False

    def test_off_never_confirms(self, dirs):
        g, p = dirs
        cortex = Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="off")
        engram = Engram(text="test", kind="lesson", scope="global", confidence="high")
        assert cortex.encoding_gate(engram) is False

    def test_copilot_confirms_low_confidence(self, dirs):
        g, p = dirs
        cortex = Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="copilot")
        low = Engram(text="test", kind="lesson", scope="global", confidence="medium")
        high = Engram(text="test", kind="lesson", scope="global", confidence="high")
        assert cortex.encoding_gate(low) is True
        assert cortex.encoding_gate(high) is False


class TestNeedsCompaction:
    def test_below_threshold(self, cortex):
        assert cortex.needs_compaction() is False

    def test_above_threshold(self, cortex, dirs):
        g, _ = dirs
        hc = Hippocampus(g)
        for i in range(55):
            hc.encode_lesson(f"Fact number {i}")
        assert cortex.needs_compaction() is True


class TestMaybeUpdateIdentity:
    async def test_no_llm_does_nothing(self, cortex, dirs):
        # cortex has no LLM by default in fixture
        await cortex.maybe_update_identity("I'm Jorge")
        g, _ = dirs
        assert not (g / "profile.md").exists()

    async def test_off_mode_does_nothing(self, dirs):
        g, p = dirs
        mock_llm = AsyncMock()
        cortex = Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="off", llm_client=mock_llm)
        await cortex.maybe_update_identity("I'm Jorge")
        mock_llm.generate_object_code.assert_not_called()

    async def test_extracts_identity(self, dirs):
        from anton.core.memory.cortex import _IdentityFacts

        g, p = dirs
        mock_llm = AsyncMock()
        mock_llm.generate_object_code = AsyncMock(
            return_value=_IdentityFacts(facts=["Name: Jorge"])
        )
        cortex = Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="copilot", llm_client=mock_llm)
        await cortex.maybe_update_identity("Hi, I'm Jorge")
        assert (g / "profile.md").exists()
        assert "Name: Jorge" in (g / "profile.md").read_text()


# ─────────────────────────────────────────────────────────────────────────────
# ENG-1390 — the hidden LLM call in prompt assembly
# ─────────────────────────────────────────────────────────────────────────────


def _when_engrams(n: int, filler: int = 500) -> list[Engram]:
    """Enough conditional rules to clear every gate.

    Four must all hold before the call happens: a user message + an attached LLM,
    formatted rules over `_RULES_BUDGET_CHARS` (6000), at least one `when` rule,
    and a conditional block of at least 1000 chars. 12 x ~545 chars clears the
    6000 budget; `filler=5` deliberately does not (see the short-circuit test).
    """
    return [
        Engram(
            text=f"When condition {i:03d} applies, do the thing {'x' * filler}",
            kind="when",
            scope="global",
            confidence="high",
        )
        for i in range(n)
    ]


class _FakeLLM:
    """Stands in for LLMClient.code, recording the trace tags it was called under."""

    def __init__(self, content: str = "", raises: bool = False, stop_reason: str | None = None):
        self._content = content
        self._raises = raises
        self._stop_reason = stop_reason
        self.calls = 0
        self.tags_seen: tuple[str, ...] = ()

    async def code(self, **kwargs):
        from anton.core.llm.provider import LLMResponse, Usage
        from anton.core.llm.tracing import get_trace_context

        self.calls += 1
        ctx = get_trace_context()
        self.tags_seen = ctx.tags if ctx else ()
        if self._raises:
            raise RuntimeError("provider exploded")
        return LLMResponse(
            content=self._content,
            usage=Usage(input_tokens=4258, output_tokens=252),
            stop_reason=self._stop_reason,
        )


@pytest.fixture()
def captured_events(monkeypatch):
    """Capture the analytics events the rule-retrieval path emits."""
    events: list[dict] = []
    import anton.core.memory.cortex as cortex_mod

    # raising=False on purpose: on code without the instrumentation the hook does
    # not exist, and we want these tests to FAIL with "no event was emitted"
    # rather than ERROR on a missing attribute. An error says the fixture broke;
    # a failure says the property under test is absent, which is the point.
    monkeypatch.setattr(
        cortex_mod, "_emit_rule_retrieval", lambda **shape: events.append(shape), raising=False
    )
    return events


def _one_event(events: list[dict]) -> dict:
    """The single rule-retrieval event, with a failure message worth reading.

    Indexing `events[0]` directly fails as a bare IndexError, which tells a
    future reader nothing about what broke.
    """
    assert len(events) == 1, (
        f"expected exactly one rule_retrieval event, got {len(events)}: {events}. "
        "Zero means the call is not reporting its outcome at all — which is the "
        "defect ENG-1390 exists to close."
    )
    return events[0]


class TestRuleRetrievalObservability:
    """ENG-1390 — the call must be countable, and its outcomes distinguishable.

    Before this, `filtered_when = when_engrams` was reached three different ways
    (no exact match, empty response, exception) and all three were byte-identical
    in their effect on the prompt — so a count of invocations could never
    separate a working filter from a broken one. Measured in production at 1.77%
    of all LLM calls before any of this existed.
    """

    @pytest.fixture(autouse=True)
    def _stub_the_analytics_emit(self, captured_events):
        """Route the emit through the capture for EVERY test in this class.

        Autouse rather than per-test opt-in, so a test added here without asking
        for `captured_events` still cannot reach the real `_emit_rule_retrieval`.

        This is defence in depth, NOT a live-leak fix: `tests/conftest.py`
        already disables analytics for the whole suite, precisely because the
        default `analytics_url` is the real collector and turn-cost tests began
        reaching that sink. So an unguarded test here does real work (builds
        `AntonSettings`, calls `send_event`) but sends nothing. Keeping the stub
        local means hermeticity does not depend on that global staying in place.
        Caught in review by @pnewsam.
        """
        return captured_events

    async def _run(self, cortex, llm, rules):
        cortex._llm = llm
        return await cortex._retrieve_relevant_rules(rules, "please do the august campaign")

    async def test_filtered_outcome_is_reported(self, cortex, captured_events):
        rules = _when_engrams(12)
        # Echo back exactly two rules, verbatim, as the prompt demands.
        echo = "\n".join(f"- {rules[i].text}" for i in (0, 1))
        llm = _FakeLLM(content=echo)
        out = await self._run(cortex, llm, rules)

        assert llm.calls == 1
        assert len(out) == 2, "behaviour must be unchanged: only the echoed rules survive"
        ev = _one_event(captured_events)
        assert ev["outcome"] == "filtered"
        assert ev["when_rules"] == 12 and ev["kept_rules"] == 2

    async def test_dropped_all_is_its_own_outcome(self, cortex, captured_events):
        """`NONE` discards every conditional rule — measured at 30% and absent
        from the ticket's original three-outcome taxonomy."""
        rules = _when_engrams(12)
        out = await self._run(cortex, _FakeLLM(content="NONE"), rules)
        assert out == [], "no mandatory rules present, so nothing should remain"
        ev = _one_event(captured_events)
        assert ev["outcome"] == "dropped_all"
        assert ev["kept_rules"] == 0

    async def test_no_exact_match_is_distinguishable_from_keeping_everything(
        self, cortex, captured_events
    ):
        """The silent fallback. The model answered, but nothing matched verbatim —
        so every rule is kept, which is indistinguishable in the PROMPT from the
        filter deciding all rules were relevant. Only the event separates them."""
        rules = _when_engrams(12)
        # Plausible model output that fails exact string equality.
        llm = _FakeLLM(content="- When condition 001 applies, do the thing (reworded)")
        out = await self._run(cortex, llm, rules)

        assert len(out) == 12, "behaviour must be unchanged: everything is kept"
        ev = _one_event(captured_events)
        assert ev["outcome"] == "kept_all_no_match"
        assert ev["kept_rules"] == 12

    async def test_empty_response_is_distinguishable(self, cortex, captured_events):
        rules = _when_engrams(12)
        out = await self._run(cortex, _FakeLLM(content="   "), rules)
        assert len(out) == 12
        assert _one_event(captured_events)["outcome"] == "kept_all_empty"

    async def test_exception_is_reported_at_all(self, cortex, captured_events):
        """The one outcome no trace could ever show: a failed call leaves no
        observation to count, so retroactive measurement is blind to it."""
        rules = _when_engrams(12)
        out = await self._run(cortex, _FakeLLM(raises=True), rules)
        assert len(out) == 12, "the permissive fallback must still apply"
        assert _one_event(captured_events)["outcome"] == "error"

    async def test_truncation_is_reported_via_stop_reason(self, cortex, captured_events):
        """A verbatim echo cut at the ceiling is accepted as 'the relevant rules',
        so truncation masquerades as a successful filter — 8 of 9 truncated calls
        in production were classified that way. `stop_reason` is what separates
        them; output_tokens == max_tokens is a heuristic with false positives."""
        rules = _when_engrams(12)
        echo = "\n".join(f"- {rules[i].text}" for i in (0, 1))
        llm = _FakeLLM(content=echo, stop_reason="max_tokens")
        await self._run(cortex, llm, rules)
        ev = _one_event(captured_events)
        assert ev["outcome"] == "filtered"
        assert ev["stop_reason"] == "max_tokens", "a truncated filter must be separable"

    async def test_the_call_is_tagged_so_it_is_isolatable_in_a_trace(self, cortex):
        """Done-when #1: distinguishable from every other `llm.code` call."""
        from anton.core.llm.tracing import (
            TraceContext,
            reset_trace_context,
            set_trace_context,
        )

        rules = _when_engrams(12)
        llm = _FakeLLM(content="NONE")
        token = set_trace_context(TraceContext(session_id="conv-1", turn_id=3, harness="anton"))
        try:
            await self._run(cortex, llm, rules)
        finally:
            reset_trace_context(token)

        assert "rule-retrieval" in llm.tags_seen, llm.tags_seen
        # …and the tag must not leak past the call it annotates.
        from anton.core.llm.tracing import get_trace_context

        assert get_trace_context() is None

    async def test_a_user_abort_is_not_reported_as_a_filter_failure(self, cortex, captured_events):
        """`CancelledError` is a BaseException, so `except Exception` misses it —
        but the `finally` still fires. Without a distinct branch every user
        pressing STOP (or an abandoned SSE stream) was counted as `error`,
        inflating the failure rate in the one metric this instrumentation exists
        to produce. Found by adversarial self-review, reproduced before fixing."""
        import asyncio

        class _Hanging:
            async def code(self, **kwargs):
                await asyncio.sleep(30)

        cortex._llm = _Hanging()
        task = asyncio.create_task(
            cortex._retrieve_relevant_rules(_when_engrams(12), "do the august campaign")
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert _one_event(captured_events)["outcome"] == "cancelled", (
            "a cancelled turn must be distinguishable from a filter that failed"
        )

    async def test_the_tag_reaches_the_actual_langfuse_header(self, cortex):
        """Done-when #1 is about a TRACE, so asserting the ContextVar is not enough.

        The tag has to survive `_build_trace_headers` and its sanitiser (which
        drops commas and control chars) to reach the wire. Without this, an
        allowlist or a rename in that builder would break the ticket's
        requirement with every other test still green.
        """
        from anton.core.llm.openai import OpenAIProvider
        from anton.core.llm.tracing import (
            TraceContext,
            reset_trace_context,
            set_trace_context,
            tagged_trace,
        )

        provider = OpenAIProvider(api_key="x", base_url="https://api.mindshub.ai/v1")
        token = set_trace_context(TraceContext(session_id="c", turn_id=1, harness="anton"))
        try:
            with tagged_trace("rule-retrieval"):
                tagged = (provider._build_trace_headers() or {}).get("Langfuse-Tags", "")
            after = (provider._build_trace_headers() or {}).get("Langfuse-Tags", "")
        finally:
            reset_trace_context(token)

        assert "rule-retrieval" in tagged.split(","), tagged
        assert "rule-retrieval" not in after.split(","), (
            "the tag must not outlive the call it annotates"
        )

    async def test_no_rule_text_or_user_message_reaches_analytics(self, cortex, captured_events):
        """The ticket's security note: shape only — counts, sizes, enums."""
        rules = _when_engrams(12)
        await self._run(cortex, _FakeLLM(content="NONE"), rules)
        ev = _one_event(captured_events)
        blob = repr(ev)
        assert "august campaign" not in blob, "user message must never be sent"
        assert "do the thing" not in blob, "rule text must never be sent"
        assert set(ev) == {
            "outcome",
            "when_rules",
            "kept_rules",
            "rules_chars",
            "stop_reason",
            "input_tokens",
            "output_tokens",
            "duration_ms",
        }

    async def test_gates_still_short_circuit_without_an_llm_call(self, cortex, captured_events):
        """Below the budget the call must not happen — and must not be reported."""
        llm = _FakeLLM(content="NONE")
        small = _when_engrams(1, filler=5)
        out = await self._run(cortex, llm, small)
        assert llm.calls == 0 and captured_events == []
        assert out == small



class _CompactionLLM:
    """Stands in for the coding model, capturing what compaction asked it."""

    def __init__(self, keep: list[int]) -> None:
        self._keep = keep
        self.prompt = ""
        self.answer: _CompactionResult | None = None

    async def generate_object_code(self, schema, *, system, messages, max_tokens):
        self.prompt = messages[0]["content"]
        self.answer = _CompactionResult(keep=self._keep)
        return self.answer


class TestCompactFile:
    """Compaction selects entries by index; the file is rebuilt from disk.

    Driven through `_compact_file` rather than a helper, because the defect
    worth guarding is a mismatch between the numbering the model is shown and
    the numbering applied to the result — an off-by-one there deletes the wrong
    memory, and no test of the selection alone can see it.
    """

    @staticmethod
    def _lessons(dirs, count: int) -> Hippocampus:
        hc = Hippocampus(dirs[0])
        for i in range(count):
            hc.encode_lesson(f"Fact number {i}", topic=f"t{i}")
        return hc

    def _entries(self, hc: Hippocampus) -> list[str]:
        return [
            ln for ln in hc._lessons_path.read_text().splitlines() if ln.startswith("- ")
        ]

    async def _compact(self, dirs, hc, keep: list[int]) -> _CompactionLLM:
        llm = _CompactionLLM(keep)
        cortex = Cortex(
            global_hc=hc, project_hc=Hippocampus(dirs[1]), llm_client=llm
        )
        await cortex._compact_file(hc, hc._lessons_path, "lesson")
        return llm

    async def test_numbering_shown_matches_the_numbering_applied(self, dirs):
        """The index→entry mapping must survive the round trip."""
        hc = self._lessons(dirs, 10)
        before = self._entries(hc)
        llm = await self._compact(dirs, hc, keep=[4])
        assert llm.prompt.splitlines()[3].startswith("4. Fact number 3")
        assert self._entries(hc) == [before[3]]

    async def test_survivors_are_byte_identical_including_metadata(self, dirs):
        hc = self._lessons(dirs, 10)
        before = self._entries(hc)
        await self._compact(dirs, hc, keep=[1, 5, 10])
        assert self._entries(hc) == [before[0], before[4], before[9]]

    async def test_output_follows_file_order_not_the_models_order(self, dirs):
        """File position is the recency signal budget-limited recall reads."""
        hc = self._lessons(dirs, 10)
        before = self._entries(hc)
        await self._compact(dirs, hc, keep=[9, 2, 5])
        assert self._entries(hc) == [before[1], before[4], before[8]]

    async def test_invented_and_duplicate_indices_cost_nothing(self, dirs):
        hc = self._lessons(dirs, 10)
        before = self._entries(hc)
        await self._compact(dirs, hc, keep=[2, 2, 0, -1, 11, 9999])
        assert self._entries(hc) == [before[1]]

    @staticmethod
    def _with_hand_written_note(hc: Hippocampus) -> str:
        """Seed a line the rebuild does not preserve, and return the file.

        The rebuild keeps `- ` entries and nothing else, so on an already
        canonical file "returned early" and "rewrote every entry" produce the
        same bytes — a test on such a file cannot tell a working guard from a
        missing one. A hand-written note makes the two outcomes differ.
        """
        note = "A note the user typed here by hand.\n"
        hc._lessons_path.write_text(hc._lessons_path.read_text() + note)
        return hc._lessons_path.read_text()

    async def test_empty_answer_leaves_the_file_untouched(self, dirs):
        """A model that names nothing gets to skip compaction, not to wipe it."""
        hc = self._lessons(dirs, 10)
        before = self._with_hand_written_note(hc)
        await self._compact(dirs, hc, keep=[])
        assert hc._lessons_path.read_text() == before

    async def test_llm_failure_leaves_the_file_untouched(self, dirs):
        hc = self._lessons(dirs, 10)
        before = self._with_hand_written_note(hc)
        llm = AsyncMock()
        llm.generate_object_code.side_effect = RuntimeError("gateway down")
        cortex = Cortex(
            global_hc=hc, project_hc=Hippocampus(dirs[1]), llm_client=llm
        )
        await cortex._compact_file(hc, hc._lessons_path, "lesson")
        assert hc._lessons_path.read_text() == before

    async def test_short_file_is_not_sent_to_the_model(self, dirs):
        hc = self._lessons(dirs, 7)
        llm = await self._compact(dirs, hc, keep=[1])
        assert llm.prompt == ""
        assert len(self._entries(hc)) == 7

    _LOGGER = "anton.core.memory.cortex"

    async def test_a_rewrite_reports_how_much_it_deleted(self, dirs, caplog):
        """Over-pruning is silent and, by index, cheaper than keeping.

        The exact message is asserted so entry text cannot start riding along:
        these files hold user content.
        """
        hc = self._lessons(dirs, 10)
        with caplog.at_level(logging.INFO, logger=self._LOGGER):
            await self._compact(dirs, hc, keep=[1, 5, 10])
        recs = [r for r in caplog.records if r.name == self._LOGGER]
        assert len(recs) == 1
        assert recs[0].getMessage() == "memory-compaction: lessons.md kept 3 of 10 entries"

    async def test_a_skipped_rewrite_reports_nothing(self, dirs, caplog):
        """The line means "the file changed", so it must sit past the guard."""
        hc = self._lessons(dirs, 10)
        with caplog.at_level(logging.INFO, logger=self._LOGGER):
            await self._compact(dirs, hc, keep=[])
        assert [r for r in caplog.records if r.name == self._LOGGER] == []

    async def test_the_answer_fits_the_budget_at_the_size_that_broke_it(self, dirs):
        """Rebuild the shape of the file that produced the bug report.

        Echoing survivors back cost 5,272 output tokens on that file and blew
        both rungs of the (4096, 8192) ladder. Every other test here runs 10
        short entries, where an echo-back answer still fits — so only a fixture
        at the real size can tell a schema that scales from one that does not.
        """
        hc = Hippocampus(dirs[0])
        for i in range(78):
            hc.encode_lesson(f"Fact number {i}: " + "detail " * 40, topic=f"t{i}")

        llm = await self._compact(dirs, hc, keep=list(range(1, 79)))

        assert len(self._entries(hc)) == 78
        # Guards the fixture itself: a shrunken file would pass the budget
        # assertion below for the wrong reason.
        assert len(llm.prompt) > 20_000
        # ~4 chars/token, the same convention `_filter_by_token_budget` uses.
        assert len(llm.answer.model_dump_json()) / 4 < 4096

    async def test_survivors_reparse_as_engrams(self, dirs):
        """Metadata must survive intact, or recall loses topic and recency."""
        hc = self._lessons(dirs, 10)
        await self._compact(dirs, hc, keep=[3, 7])
        engrams = hc.get_lessons()
        assert [e.text for e in engrams] == ["Fact number 2", "Fact number 6"]
        assert [e.topic for e in engrams] == ["t2", "t6"]
        assert all(e.updated_at is not None for e in engrams)


class TestCompactRulesFile:
    """The rules rebuild must round-trip each entry's section, not guess it.

    `rules.md` differs from `lessons.md` in a way that matters: its rewrite is
    not a no-op, so a bug here can grow or misfile the file the agent reads as
    its own behavioural gates.
    """

    @staticmethod
    def _rules(dirs, count: int) -> Hippocampus:
        hc = Hippocampus(dirs[0])
        for i in range(count):
            # "when ... always ..." matches two of the old keyword buckets at
            # once — the shape that got written twice.
            hc.encode_rule(
                f"When condition {i} holds, always prefer option {i}",
                kind="when", confidence="high", source="user",
            )
        hc.encode_rule("Use httpx", kind="always", confidence="high", source="user")
        hc.encode_rule("Never log secrets", kind="never", confidence="high", source="user")
        return hc

    async def _compact(self, dirs, hc, keep: list[int]) -> None:
        cortex = Cortex(
            global_hc=hc, project_hc=Hippocampus(dirs[1]), llm_client=_CompactionLLM(keep)
        )
        await cortex._compact_file(hc, hc._rules_path, "rules")

    async def test_keeping_everything_is_idempotent(self, dirs):
        """A rule matching two headings' keywords must still be written once."""
        hc = self._rules(dirs, 8)
        before = hc.get_rules()
        await self._compact(dirs, hc, keep=list(range(1, 11)))
        after = hc.get_rules()
        assert [(e.kind, e.text) for e in after] == [(e.kind, e.text) for e in before]
        assert len(after) == 10

    async def test_entries_stay_under_their_original_heading(self, dirs):
        # File order is by section, not by insertion: 1 always, 2 never, 3+ when.
        hc = self._rules(dirs, 8)
        await self._compact(dirs, hc, keep=[1, 2, 3])
        by_kind = {e.kind: e.text for e in hc.get_rules()}
        assert by_kind["always"] == "Use httpx"
        assert by_kind["never"] == "Never log secrets"
        assert by_kind["when"].startswith("When condition 0 holds")

    async def test_empty_answer_does_not_rewrite_the_rules_file(self, dirs):
        """Same guard as on the lessons path, checked on the file that matters."""
        hc = self._rules(dirs, 8)
        hc._rules_path.write_text(
            hc._rules_path.read_text() + "\nA note the user typed here by hand.\n"
        )
        before = hc._rules_path.read_text()
        await self._compact(dirs, hc, keep=[])
        assert hc._rules_path.read_text() == before

    async def test_an_emptied_section_keeps_its_heading(self, dirs):
        """`get_rules` needs the heading to assign a kind to later entries."""
        hc = self._rules(dirs, 8)
        await self._compact(dirs, hc, keep=[3])  # one "when" rule, nothing else
        content = hc._rules_path.read_text()
        assert "## Always" in content and "## Never" in content and "## When" in content
        hc.encode_rule("Added later", kind="never", confidence="high", source="user")
        assert {e.kind for e in hc.get_rules()} == {"never", "when"}

    async def test_entries_above_the_first_heading_are_not_lost(self, dirs):
        """Hand-edited files exist; a stray rule must survive, not vanish."""
        hc = self._rules(dirs, 8)
        hc._rules_path.write_text(
            "# Rules\n- Stray hand-added rule <!-- confidence:high source:user ts:2026-01-01 -->\n"
            + hc._rules_path.read_text().removeprefix("# Rules\n")
        )
        await self._compact(dirs, hc, keep=list(range(1, 12)))
        assert "Stray hand-added rule" in hc._rules_path.read_text()
        assert len(hc.get_rules()) == 11

    async def test_entries_under_an_unknown_heading_are_not_lost(self, dirs):
        """`save_rules` only writes three headings, so a fourth must be folded
        into one of them rather than silently dropped on rebuild."""
        hc = self._rules(dirs, 8)
        hc._rules_path.write_text(
            hc._rules_path.read_text()
            + "\n## Notes\n- Hand-added note <!-- confidence:high source:user ts:2026-01-01 -->\n"
        )
        await self._compact(dirs, hc, keep=list(range(1, 12)))
        assert "Hand-added note" in hc._rules_path.read_text()
        assert len(hc.get_rules()) == 11


# ─────────────────────────────────────────────────────────────────────────────
# The operator's read-only layer
# ─────────────────────────────────────────────────────────────────────────────

# A person's tiers with every section build_memory_context renders. Timestamps
# are fixed so the rendered lessons are stable.
_PERSON_GLOBAL = {
    "profile.md": "# Profile\n- Name: Ada\n- Timezone: PST\n",
    "rules.md": (
        "# Rules\n\n## Always\n- Use httpx <!-- confidence:high source:user ts:2026-10-01 -->\n"
        "\n## Never\n- Never commit secrets <!-- confidence:high source:user ts:2026-10-01 -->\n"
        "\n## When\n- If the user writes in Spanish, respond in Spanish"
        " <!-- confidence:high source:user ts:2026-10-01 -->\n"
        "- If a scratchpad API is paginated, use progress()"
        " <!-- confidence:high source:user ts:2026-10-02 -->\n"
    ),
    "lessons.md": (
        "# Lessons\n- CoinGecko rate-limits at 50/min <!-- topic:api ts:2026-10-01 -->\n"
        "- Scratchpad cells time out at 30s <!-- topic:scratchpad ts:2026-10-02 -->\n"
    ),
}
_PERSON_PROJECT = {
    "rules.md": (
        "# Rules\n\n## Always\n- Use the Django ORM"
        " <!-- confidence:high source:user ts:2026-10-01 -->\n"
    ),
    "lessons.md": (
        "# Lessons\n- Orders are partitioned by month <!-- ts:2026-10-01 -->\n"
        "- Warehouse schema notes <!-- topic:minds-datasource ts:2026-10-01 -->\n"
        "- Use one scratchpad per report <!-- topic:scratchpad ts:2026-10-03 -->\n"
    ),
}

# What Cortex rendered for the tiers above before the operator's layer existed,
# taken from that code. A Cortex with no managed layer must still produce it.
_TODAY_PROMPT = (
    "\n\n## Your Memory — Identity\n# Profile\n- Name: Ada\n- Timezone: PST"
    "\n\n## Your Memory — Global Rules\n## Always\n- Use httpx\n## Never\n- Never commit secrets"
    "\n## When\n- If the user writes in Spanish, respond in Spanish"
    "\n\n## Your Memory — Project Rules\n## Always\n- Use the Django ORM"
    "\n\n## Your Memory — Global Lessons\n# Lessons"
    "\n- CoinGecko rate-limits at 50/min <!-- topic:api ts:2026-10-01 -->\n"
    "\n\n## Your Memory — Project Lessons\n# Lessons"
    "\n- Warehouse schema notes <!-- topic:minds-datasource ts:2026-10-01 -->"
    "\n- Orders are partitioned by month <!-- ts:2026-10-01 -->\n"
    "\n\n## Minds — Datasource Context\n# minds-datasource"
    "\n- Warehouse schema notes <!-- topic:minds-datasource ts:2026-10-01 -->\n"
)
_TODAY_SCRATCHPAD = (
    "- If a scratchpad API is paginated, use progress()\n"
    "- Scratchpad cells time out at 30s\n"
    "- Use one scratchpad per report"
)

# The operator's folder: a note line operators read (not a bullet, so it never
# reaches the agent), one rule of each kind, a scratchpad rule and lesson that
# belong in the tool description, and a profile that must never be read.
_MANAGED_FILES = {
    "rules.md": (
        "# Rules\n"
        'Only "- " bullets under "## Always", "## Never" and "## When" reach the agent.\n'
        "\n## Always\n- Write requirements.txt with package names only\n"
        "\n## Never\n- Never print a connection's password\n"
        "\n## When\n- If the question names a quarter, use calendar quarters\n"
        "- If a scratchpad cell loads a large table, page through it\n"
    ),
    "lessons.md": (
        "# Lessons\n- Sales figures are in thousands of dollars <!-- ts:2026-10-01 -->\n"
        "- Scratchpad cells stop after 120 seconds <!-- topic:scratchpad ts:2026-10-01 -->\n"
    ),
    "profile.md": "# Profile\n- Name: Operator\n",
}
# The line the first operator heading carries, so the model knows the
# operator's entries win over the person's own.
_OPERATOR_WINS = (
    "The operator's rules and lessons apply to everyone and win over Your Memory "
    "when the two conflict.\n"
)
_OPERATOR_SECTIONS = (
    "\n\n## Operator Memory — Rules\n" + _OPERATOR_WINS
    + "## Always\n- Write requirements.txt with package names only"
    "\n## Never\n- Never print a connection's password"
    "\n## When\n- If the question names a quarter, use calendar quarters"
    "\n\n## Operator Memory — Lessons\n# Lessons"
    "\n- Sales figures are in thousands of dollars <!-- ts:2026-10-01 -->\n"
)


def _write(folder: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        (folder / name).write_text(content, encoding="utf-8")


@pytest.fixture()
def managed(read_only_dir) -> Path:
    return read_only_dir("managed", _MANAGED_FILES)


@pytest.fixture()
def person(dirs) -> tuple[Path, Path]:
    g, p = dirs
    _write(g, _PERSON_GLOBAL)
    _write(p, _PERSON_PROJECT)
    return g, p


def _managed_cortex(dirs, managed: Path, **kwargs) -> Cortex:
    g, p = dirs
    return Cortex(
        global_hc=Hippocampus(g),
        project_hc=Hippocampus(p),
        mode="autopilot",
        managed_hc=Hippocampus(managed),
        **kwargs,
    )


class TestManagedLayer:
    """A read-only store an operator manages, read ahead of the person's own.

    The folder is read-only on disk, as a host ships it, so a write that
    reached it would raise. Memorize's background encode, the cerebellum and
    the profile move all swallow that error, so every write test also checks
    the entry landed in the person's own tier.
    """

    async def test_no_managed_layer_is_byte_identical_to_today(self, person):
        g, p = person
        for cortex in (
            Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="autopilot"),
            Cortex(
                global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="autopilot",
                managed_hc=None,
            ),
        ):
            assert await cortex.build_memory_context("hola") == _TODAY_PROMPT
            assert cortex.get_scratchpad_context() == _TODAY_SCRATCHPAD

    def test_managed_hc_is_keyword_only(self, dirs, managed):
        g, p = dirs
        with pytest.raises(TypeError):
            Cortex(Hippocampus(g), Hippocampus(p), "autopilot", None, None, Hippocampus(managed))

    async def test_operator_sections_lead_and_leave_the_rest_unchanged(self, person, managed):
        prompt = await _managed_cortex(person, managed).build_memory_context("hola")
        assert prompt == _OPERATOR_SECTIONS + _TODAY_PROMPT

    async def test_operator_rules_come_before_identity_and_the_persons_rules(
        self, person, managed
    ):
        prompt = await _managed_cortex(person, managed).build_memory_context("hola")
        headings = [
            "## Operator Memory — Rules",
            "## Your Memory — Identity",
            "## Your Memory — Global Rules",
            "## Your Memory — Project Rules",
        ]
        positions = [prompt.index(h) for h in headings]
        assert positions == sorted(positions)

    async def test_operator_lessons_come_before_the_persons_lessons(self, person, managed):
        prompt = await _managed_cortex(person, managed).build_memory_context("hola")
        headings = [
            "## Operator Memory — Lessons",
            "## Your Memory — Identity",
            "## Your Memory — Global Lessons",
            "## Your Memory — Project Lessons",
        ]
        positions = [prompt.index(h) for h in headings]
        assert positions == sorted(positions)
        assert prompt.index("## Operator Memory — Rules") < positions[0]

    async def test_the_operator_heading_says_operator_entries_win(self, person, managed):
        """Rendering first doesn't say which tier wins, so the first operator
        heading says it, once. Without the layer the line never renders."""
        prompt = await _managed_cortex(person, managed).build_memory_context("hola")
        assert prompt.count(_OPERATOR_WINS) == 1
        assert "## Operator Memory — Rules\n" + _OPERATOR_WINS in prompt

        g, p = person
        bare = Cortex(
            global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="autopilot",
            managed_hc=None,
        )
        assert _OPERATOR_WINS.strip() not in await bare.build_memory_context("hola")

    async def test_a_managed_rule_with_a_kind_comment_renders_under_its_heading(
        self, person, read_only_dir
    ):
        """A trailing `<!-- kind:... -->` on a managed rule must not reach
        Engram next to the heading's kind, which raises TypeError on every
        turn. The "## " heading decides the kind; the comment's kind is
        dropped."""
        managed = read_only_dir("managed", {
            "rules.md": "# Rules\n\n## When\n- Use pallets <!-- kind:always -->\n",
        })
        prompt = await _managed_cortex(person, managed).build_memory_context("hola")
        assert (
            "## Operator Memory — Rules\n" + _OPERATOR_WINS + "## When\n- Use pallets"
        ) in prompt

    async def test_operator_wisdom_leads_the_scratchpad_context(self, person, managed):
        context = _managed_cortex(person, managed).get_scratchpad_context()
        # Within the operator's tier, Hippocampus's own order: the dated
        # lesson ranks ahead of the undated rule. The operator's block is
        # labeled and the person's own follows under its own label.
        assert context == (
            "Operator (applies to everyone, wins on conflict):\n"
            "- Scratchpad cells stop after 120 seconds\n"
            "- If a scratchpad cell loads a large table, page through it\n"
            "Your own:\n" + _TODAY_SCRATCHPAD
        )

    async def test_an_operator_layer_without_scratchpad_wisdom_leaves_the_context_unlabeled(
        self, person, read_only_dir
    ):
        managed = read_only_dir("managed", {"rules.md": "# Rules\n\n## Always\n- Use pallets\n"})
        assert _managed_cortex(person, managed).get_scratchpad_context() == _TODAY_SCRATCHPAD

    async def test_operator_wisdom_alone_carries_no_your_own_label(self, dirs, managed):
        # `dirs` gives the person empty global and project tiers.
        assert _managed_cortex(dirs, managed).get_scratchpad_context() == (
            "Operator (applies to everyone, wins on conflict):\n"
            "- Scratchpad cells stop after 120 seconds\n"
            "- If a scratchpad cell loads a large table, page through it"
        )

    async def test_the_operator_profile_is_never_read(self, person, managed):
        cortex = _managed_cortex(person, managed)
        assert "Operator" not in (await cortex.build_memory_context("hola")).replace(
            "Operator Memory", ""
        )
        # The scratchpad context labels the operator's block, so the word
        # "Operator" is expected there; what must never leak is the profile.
        assert "Name: Operator" not in cortex.get_scratchpad_context()

    async def test_an_operator_datasource_lesson_renders_as_an_operator_lesson(
        self, person, read_only_dir
    ):
        """The Minds section reads only the person's project tier, so a layer
        lesson tagged `minds-datasource` shows once, with the operator's
        lessons. Filtering it out of those, or adding it to the Minds section,
        breaks the equality."""
        managed = read_only_dir("managed", {
            "lessons.md": (
                "# Lessons\n- Revenue lives in fct_revenue"
                " <!-- topic:minds-datasource ts:2026-10-01 -->\n"
            ),
        })

        prompt = await _managed_cortex(person, managed).build_memory_context("hola")

        assert prompt == (
            "\n\n## Operator Memory — Lessons\n" + _OPERATOR_WINS + "# Lessons"
            "\n- Revenue lives in fct_revenue <!-- topic:minds-datasource ts:2026-10-01 -->\n"
            + _TODAY_PROMPT
        )

    async def test_operator_lessons_have_their_own_budget(self, dirs, read_only_dir):
        """Each tier keeps about 4,000 characters of lessons. Sharing one budget
        with the person's tiers would cut half of these."""
        g, _ = dirs

        def lessons(label: str) -> str:
            lines = [f"- {label} lesson {i} {'x' * 480} <!-- ts:2026-10-01 -->" for i in range(7)]
            return "# Lessons\n" + "\n".join(lines) + "\n"

        managed = read_only_dir("managed", {"lessons.md": lessons("operator")})
        _write(g, {"lessons.md": lessons("person")})
        prompt = await _managed_cortex(dirs, managed).build_memory_context("hola")
        for label in ("operator", "person"):
            for i in range(7):
                assert f"{label} lesson {i} " in prompt

    async def test_operator_when_rules_go_through_the_same_filter(self, dirs, read_only_dir):
        """All three tiers' filters run in one gather, not one after another."""
        def when_rules(label: str) -> str:
            lines = "\n".join(f"- {label}: {e.text}" for e in _when_engrams(12))
            return f"## When\n{lines}\n"

        managed = read_only_dir("managed", {"rules.md": when_rules("operator")})
        for d, label in zip(dirs, ("global", "project")):
            _write(d, {"rules.md": when_rules(label)})

        in_flight = peak = 0

        class _SlowLLM:
            async def code(self, **kwargs):
                from anton.core.llm.provider import LLMResponse, Usage

                nonlocal in_flight, peak
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1
                return LLMResponse(content="", usage=Usage())

        cortex = _managed_cortex(dirs, managed, llm_client=_SlowLLM())
        prompt = await cortex.build_memory_context("do the august campaign")
        assert peak == 3
        # An empty answer keeps every rule.
        assert prompt.index("## Operator Memory — Rules") < prompt.index("## Your Memory — Global Rules")

    async def test_long_operator_when_rules_are_filtered(
        self, dirs, read_only_dir, captured_events
    ):
        rules = _when_engrams(12)
        lines = "\n".join(f"- {e.text}" for e in rules)
        managed = read_only_dir(
            "managed", {"rules.md": f"## Always\n- Keep it short\n## When\n{lines}\n"}
        )
        llm = _FakeLLM(content="\n".join(f"- {rules[i].text}" for i in (0, 1)))

        cortex = _managed_cortex(dirs, managed, llm_client=llm)
        prompt = await cortex.build_memory_context("do the august campaign")

        assert llm.calls == 1
        assert "- Keep it short" in prompt
        assert rules[0].text in prompt and rules[1].text in prompt
        assert rules[2].text not in prompt

    async def test_an_empty_operator_tier_makes_no_model_call(
        self, dirs, read_only_dir, captured_events
    ):
        g, _ = dirs
        lines = "\n".join(f"- {e.text}" for e in _when_engrams(12))
        _write(g, {"rules.md": f"## When\n{lines}\n"})
        managed = read_only_dir("managed", {})
        llm = _FakeLLM(content="")

        await _managed_cortex(dirs, managed, llm_client=llm).build_memory_context("go")

        assert llm.calls == 1  # the person's global tier only
        assert len(captured_events) == 1

    async def test_encode_never_writes_the_managed_folder(self, dirs, managed):
        g, p = dirs
        before = _snapshot(managed)
        engrams = [
            Engram(text="Name: Ada", kind="profile", scope="global"),
            Engram(text="Timezone: PST", kind="profile", scope="project"),
            *(
                Engram(
                    text=f"{kind} rule for {scope}", kind=kind, scope=scope,
                    confidence="high", source="user",
                )
                for kind in ("always", "never", "when")
                for scope in ("global", "project")
            ),
            *(
                Engram(text=f"lesson for {scope}", kind="lesson", scope=scope)
                for scope in ("global", "project")
            ),
        ]

        await _managed_cortex(dirs, managed).encode(engrams)

        assert _snapshot(managed) == before
        profile = (g / "profile.md").read_text()
        assert "Name: Ada" in profile and "Timezone: PST" in profile
        for scope, folder in (("global", g), ("project", p)):
            rules = (folder / "rules.md").read_text()
            for kind in ("always", "never", "when"):
                assert f"{kind} rule for {scope}" in rules
            assert f"lesson for {scope}" in (folder / "lessons.md").read_text()

    async def test_compaction_and_vacuum_never_touch_the_managed_folder(
        self, dirs, read_only_dir
    ):
        g, _ = dirs
        many = "# Lessons\n" + "".join(
            f"- Managed fact {i} <!-- ts:2026-10-01 -->\n" for i in range(25)
        )
        managed = read_only_dir("managed", {"lessons.md": many})
        before = _snapshot(managed)

        cortex = _managed_cortex(dirs, managed, llm_client=_CompactionLLM(keep=[1]))
        # The operator's entries don't count toward the person's threshold.
        assert cortex.needs_compaction() is False

        _write(g, {"lessons.md": many.replace("Managed", "Person")})
        await cortex.compact_all()
        assert len(Hippocampus(g).get_lessons()) == 1  # compaction did run

        _write(g, {"lessons.md": many.replace("Managed", "Person")})
        await cortex.vacuum()
        assert len(Hippocampus(g).get_lessons()) == 1

        assert _snapshot(managed) == before

    def test_the_profile_move_targets_global_never_managed(self, dirs, managed, caplog):
        g, p = dirs
        Hippocampus(p).rewrite_identity(["Timezone: PST"])
        before = _snapshot(managed)

        with caplog.at_level(logging.WARNING, logger="anton.core.memory.cortex"):
            _managed_cortex(dirs, managed)

        assert _snapshot(managed) == before
        assert "Timezone: PST" in (g / "profile.md").read_text()
        assert not (p / "profile.md").exists()
        assert [r for r in caplog.records if r.name == "anton.core.memory.cortex"] == []

    async def test_a_cerebellum_lesson_lands_in_the_project_tier(self, dirs, managed):
        _, p = dirs
        before = _snapshot(managed)
        llm = MagicMock()
        llm.generate_object_code = AsyncMock(
            return_value=_DiffPassResult(
                lessons=[_LessonDraft(text="Check a column exists before filtering", topic="pandas")]
            )
        )
        cerebellum = Cerebellum(cortex=_managed_cortex(dirs, managed), llm=llm)

        await cerebellum.on_post_execute(
            Cell(code="df[df.x > 1]", stdout="", stderr="", error="KeyError: 'x'")
        )
        await cerebellum.flush()

        assert "Check a column exists before filtering" in (p / "lessons.md").read_text()
        assert _snapshot(managed) == before

    async def test_memorize_writes_the_persons_tiers(self, dirs, managed):
        g, p = dirs
        before = _snapshot(managed)
        session = SimpleNamespace(_cortex=_managed_cortex(dirs, managed), _memory_writes=set())
        session._track_memory_write = types.MethodType(ChatSession._track_memory_write, session)
        session.settle_memory_writes = types.MethodType(ChatSession.settle_memory_writes, session)

        result = await handle_memorize(session, {"entries": [
            {"text": "Answer in Spanish", "kind": "always", "scope": "global"},
            {"text": "Deploy on green only", "kind": "always", "scope": "project"},
            {"text": "Name: Ada", "kind": "profile", "scope": "global"},
            {"text": "Reports go out on Mondays", "kind": "lesson", "scope": "project"},
        ]})
        await session.settle_memory_writes()

        assert result.ok is True
        assert "Answer in Spanish" in (g / "rules.md").read_text()
        assert "Deploy on green only" in (p / "rules.md").read_text()
        assert "Name: Ada" in (g / "profile.md").read_text()
        assert "Reports go out on Mondays" in (p / "lessons.md").read_text()
        assert _snapshot(managed) == before
