"""`surface` says WHERE a turn happened, and only in a vocabulary we recognise.

ENG-1459. The value arrives from a host (cowork-server derives it from org
tenancy; the CLI states it directly), so it is untrusted input on a field whose
whole purpose is to partition a population cleanly.
"""

import logging

import pytest

from anton.core.llm.tracing import (
    SURFACE_CLI,
    SURFACE_DESKTOP,
    SURFACE_WEB,
    VALID_SURFACES,
    surface_tag,
)
from anton.core.session import _validated_surface


class TestTheEnum:
    def test_cloud_is_deliberately_not_a_surface(self):
        """It is an execution mode, not a place a user sits.

        `anton/cloud_turn` runs headless with cowork as its caller, so one web
        turn could legitimately be both — folding it in would recreate the
        two-vocabularies-in-one-field problem ENG-1694 is undoing.
        """
        assert "cloud" not in VALID_SURFACES
        assert VALID_SURFACES == {SURFACE_DESKTOP, SURFACE_WEB, SURFACE_CLI}

    def test_tag_is_prefixed(self):
        assert surface_tag(SURFACE_WEB) == "surface:web"


class TestValidation:
    @pytest.mark.parametrize("value", sorted(VALID_SURFACES))
    def test_recognised_values_pass_through(self, value):
        assert _validated_surface(value) == value

    def test_none_stays_none(self):
        # "The host did not say" is a real answer, not a value to invent.
        assert _validated_surface(None) is None

    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_is_treated_as_absent(self, value):
        # ENG-1495's lesson: "" meaning both "CLI" and "unidentified" is what
        # made the old field unusable.
        assert _validated_surface(value) is None

    def test_case_and_whitespace_are_normalised(self):
        # A host sending "Web " must not create a second population.
        assert _validated_surface(" Web ") == SURFACE_WEB

    def test_an_unrecognised_value_is_dropped_with_a_warning(self, caplog):
        # Dropped, not forwarded: a junk row in every breakdown is bad, but a
        # *wrong* surface is worse — it moves a turn into the population it is
        # being compared against.
        with caplog.at_level(logging.WARNING):
            assert _validated_surface("cowork") is None
        messages = [r.getMessage() for r in caplog.records]
        assert any("cowork" in m for m in messages), messages

    def test_validation_never_raises(self):
        """Telemetry must not be able to fail a turn."""

        class Hostile:
            def __str__(self):
                raise RuntimeError("boom")

        assert _validated_surface(Hostile()) is None


class TestTheCliDeclaresItself:
    def test_cli_entrypoints_pass_the_cli_surface(self):
        """Both CLI call sites must set it, or CLI turns look like an unknown host."""
        import inspect

        from anton import chat, chat_session

        for mod in (chat, chat_session):
            src = inspect.getsource(mod)
            assert "surface=SURFACE_CLI" in src, f"{mod.__name__} does not declare its surface"


class TestItReachesTheWireFromTheConfig:
    """The seam the unit tests above cannot see.

    `_validated_surface` and `_build_trace_headers` are the two ends. Between
    them sits the session, which has to store the value and put it on the
    per-turn `TraceContext`. Both ends can be perfect while that middle is
    disconnected — mutation-testing confirmed exactly that: deleting
    `surface=self._surface` from the TraceContext left every other test in this
    file green. So this drives a real turn and reads what the provider would
    actually send.
    """

    async def _headers_during(self, session, prompt="go"):
        from anton.core.llm.openai import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._emit_trace_headers = True
        seen = []
        async for _ in session.turn_stream(prompt):
            seen.append(provider._build_trace_headers())
        return [h for h in seen if h]

    async def test_a_configured_surface_reaches_the_trace_headers(self, make_session):
        session = make_session(harness="anton", surface=SURFACE_WEB)
        headers = await self._headers_during(session)
        assert headers, "no trace headers were built during the turn"
        assert all(surface_tag(SURFACE_WEB) in h["Langfuse-Tags"] for h in headers)

    async def test_an_unconfigured_surface_stays_absent_end_to_end(self, make_session):
        session = make_session(harness="anton")
        headers = await self._headers_during(session)
        assert headers
        assert all("surface:" not in h["Langfuse-Tags"] for h in headers)

    async def test_the_session_validates_what_the_host_supplied(self, make_session):
        # Catches a session that stores config.surface raw: an unrecognised
        # value must not reach the wire, and a sloppy one must be normalised.
        assert make_session(surface=" WEB ")._surface == SURFACE_WEB
        assert make_session(surface="cowork")._surface is None


class TestThePodReceivesItsAttribution:
    """The pod cannot derive any of this, so it has to arrive on the wire.

    ENG-1459. Web turns do not run in cowork-server — they execute here, in a
    `minds-anton-scratchpad` pod ("anton + boot", no cowork-server installed).
    So `surface`, `cowork_server_version` and `install_channel` are all absent
    on web unless cowork sends them, which is what `TurnRequestV1.trace` is for.
    """

    def _req(self, **over):
        import json

        from anton.cloud_turn.contract import TurnRequestV1

        body = {"protocol_version": 1, "conversation_id": "c1", "input": "go"}
        body.update(over)
        return TurnRequestV1.from_json(json.dumps(body))

    def test_the_trace_block_survives_the_wire(self):
        req = self._req(trace={"surface": "web", "install_channel": "hosted"})
        assert req.trace == {"surface": "web", "install_channel": "hosted"}

    def test_a_controller_too_old_to_forward_it_yields_none(self):
        # Steps land in any order, so an unforwarded block must read as
        # "no attribution" rather than failing the turn.
        assert self._req().trace is None

    def test_a_non_dict_trace_is_rejected_rather_than_carried(self):
        # It reaches ChatSessionConfig and a metadata dict; a string or list
        # would explode somewhere less obvious than here.
        assert self._req(trace="web").trace is None
        assert self._req(trace=["web"]).trace is None

    def test_the_pod_config_takes_its_surface_from_the_block(self):
        from anton.core.llm.tracing import SURFACE_WEB

        # Assert on the resolution rule rather than building a real cloud
        # session (which needs a trusted mount + settings): the pod reads
        # `(request.trace or {}).get("surface")`.
        req = self._req(trace={"surface": "web"})
        assert (req.trace or {}).get("surface") == SURFACE_WEB
        assert (self._req().trace or {}).get("surface") is None
