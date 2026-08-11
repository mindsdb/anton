from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from anton.core.memory.cortex import Cortex
from anton.core.memory.hippocampus import Engram, Hippocampus


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

class TestRuleRetrievalActionName:
    """ENG-1390 — the analytics ACTION NAME, kept out of the class above on purpose.

    `TestRuleRetrievalObservability` autouse-stubs `_emit_rule_retrieval`, which is
    exactly what these tests must NOT have stubbed: they assert what reaches
    `send_event`.
    """

    async def test_the_action_name_carries_the_prefix_the_collector_requires(self):
        """The collector lambda relays only `anton_`/`ds_connect`-prefixed actions.

        It drops anything else while still returning HTTP 200, and `send_event` is
        fire-and-forget behind a bare `except`, so a wrong name is invisible from
        the client — this event shipped as `rule_retrieval` and produced nothing.
        Pinning the literal here because the constraint lives in a lambda whose
        code is in no repo, so nothing else in this codebase can enforce it.
        """
        from anton.core.memory.cortex import _RULE_RETRIEVAL_ACTION

        assert _RULE_RETRIEVAL_ACTION.startswith("anton_"), (
            f"{_RULE_RETRIEVAL_ACTION!r} would be silently dropped by the collector"
        )

    async def test_the_emitted_action_is_the_prefixed_one(self, cortex, monkeypatch):
        """End to end: the name that actually reaches `send_event`, not just the
        constant. A refactor could reintroduce a literal and this would catch it."""
        import anton.core.memory.cortex as cortex_mod

        sent = {}
        monkeypatch.setattr(cortex_mod, "_analytics_settings", object())
        monkeypatch.setattr(
            "anton.analytics.send_event",
            lambda settings, action, **props: sent.update(action=action, props=props),
        )
        cortex_mod._emit_rule_retrieval(outcome="filtered", when_rules=3)

        assert sent["action"] == "anton_rule_retrieval", sent
        assert sent["props"]["outcome"] == "filtered"
