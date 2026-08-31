"""`harness` names WHICH AGENT ran, never where (ENG-1694).

Until this change one field held both vocabularies — "anton"/"hermes" (agents)
alongside "cli"/"cloud" (places) — so it could answer neither question: a "cli"
trace could not say which agent ran, and an "anton" trace could not say where.

`surface` (ENG-1459) owns the "where" axis. This pins the split so the field
cannot quietly reacquire a second meaning, which is how it got here.
"""

import inspect

from anton.core.llm.tracing import (
    HARNESS_ANTON,
    HARNESS_HERMES,
    SURFACE_CLI,
    VALID_HARNESSES,
    VALID_SURFACES,
    TraceContext,
)


class TestTheTwoVocabulariesAreDisjoint:
    def test_no_value_means_both_an_agent_and_a_place(self):
        # The actual defect, as an assertion. `cli` was in both worlds.
        assert not (VALID_HARNESSES & VALID_SURFACES), (
            "a value that is both an agent and a surface reintroduces the "
            f"ambiguity this split removed: {VALID_HARNESSES & VALID_SURFACES}"
        )

    def test_places_are_not_harness_values(self):
        for place in ("cli", "cloud", "desktop", "web", "cowork"):
            assert place not in VALID_HARNESSES, f"{place!r} names a place, not an agent"

    def test_agents_are_not_surface_values(self):
        for agent in (HARNESS_ANTON, HARNESS_HERMES):
            assert agent not in VALID_SURFACES


class TestEveryFirstPartyCallerReportsAnAgent:
    """All three call sites named a place before this change."""

    def test_the_cli_entrypoints_report_anton_and_the_cli_surface(self):
        from anton import chat, chat_session

        for mod in (chat, chat_session):
            src = inspect.getsource(mod)
            assert "harness=HARNESS_ANTON" in src, f"{mod.__name__} does not report its agent"
            assert "surface=SURFACE_CLI" in src, f"{mod.__name__} lost its surface"
            assert 'harness="cli"' not in src, f"{mod.__name__} still names a place as its harness"

    def test_the_cloud_pod_reports_anton(self):
        # The pod image is "anton + boot" — the agent running there IS anton, so
        # "cloud" was factually wrong rather than merely overloaded.
        from anton.cloud_turn import session as cloud_session

        src = inspect.getsource(cloud_session)
        assert "harness=HARNESS_ANTON" in src
        assert 'harness="cloud"' not in src


class TestTheTraceNameIsRestored:
    """The gateway composes the display name from `harness` alone.

    So a CLI turn arrived as `cli:turn-N` while every other anton turn was
    `anton:turn-N`, splitting name-based grouping. Fixing the value fixes the
    name with no gateway change — assert it rather than assuming.
    """

    def _name_the_gateway_would_build(self, ctx: TraceContext) -> str:
        import json

        from anton.core.llm.openai import OpenAIProvider
        from anton.core.llm.tracing import reset_trace_context, set_trace_context

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._emit_trace_headers = True
        token = set_trace_context(ctx)
        try:
            md = json.loads(provider._build_trace_headers()["Langfuse-Metadata"])
        finally:
            reset_trace_context(token)
        # mindshub_inference/minds/requests/context.py: f"{harness}:turn-{turn_id}"
        return f"{md['harness']}:turn-{md['turn_id']}"

    def test_a_cli_turn_is_named_anton_again(self):
        name = self._name_the_gateway_would_build(
            TraceContext(turn_id=7, harness=HARNESS_ANTON, surface=SURFACE_CLI)
        )
        assert name == "anton:turn-7", "CLI turns must rejoin the anton:turn-N family"

    def test_a_cli_turn_stays_distinguishable_despite_the_shared_name(self):
        # Sharing the trace NAME with desktop is the point — grouping works
        # again — but the surface tag is what keeps them apart.
        import json

        from anton.core.llm.openai import OpenAIProvider
        from anton.core.llm.tracing import reset_trace_context, set_trace_context

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._emit_trace_headers = True
        token = set_trace_context(
            TraceContext(turn_id=1, harness=HARNESS_ANTON, surface=SURFACE_CLI)
        )
        try:
            headers = provider._build_trace_headers()
        finally:
            reset_trace_context(token)
        assert "surface:cli" in headers["Langfuse-Tags"]
        assert json.loads(headers["Langfuse-Metadata"])["harness"] == HARNESS_ANTON
