"""Unit tests for the verifier eval's CI gate mechanics (ENG-1334).

Not the eval itself — these test the machinery that decides whether a green
`verdict-eval` means anything. That machinery is the reason the check can be
trusted, and until this file existed it had no regression net at all: it was
verified once, by hand, in a shell, and then the scratch probe was deleted.

Three things are pinned here, each of which has already been wrong once:

1. A gateway 402/429 becomes a *marked* skip. If the marker stops reaching the
   junit report, the CI guard's out-of-money branch silently becomes dead code
   and a starved run reports red for the wrong reason.
2. A throttle is retried before giving up. Skipping on the first 429 made the
   whole matrix skip and the job go GREEN having executed nothing — ENG-1334's
   own failure mode, rebuilt by its escape hatch (caught reviewing #328).
3. The marker string matches the one the workflow greps for. Two files, one
   string, previously held together by "keep in sync" comments.
4. A repointed alias is rejected rather than silently measured. The eval picks
   its models by alias NAME, and an alias is a catalog pointer that moves
   without a PR here — `mindshub_air` moved off Kimi and the eval reported
   green for a week while covering one population twice (ENG-1687). Tested here
   because the check must hold with no key and no network.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from anton.core.llm.provider import TokenLimitExceeded

import tests.test_verifier_verdict_live as ev

_WORKFLOW = Path(__file__).resolve().parent.parent / ".github/workflows/verifier-eval.yml"


class _FakeResponse:
    """Only `.headers`, because that is all `_gateway_denial` reads.

    Deliberately NOT a real `openai.APIStatusError`: the OpenAI SDK unwraps
    `exc.body` while Anthropic does not, so a hand-built SDK error asserts a
    shape that never occurs on the wire and passes for the wrong reason.
    """

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class _FakeCause(Exception):
    """The chained SDK error: `.body` and `.response.headers`, nothing more.

    Derives from Exception because `__cause__` must — Python enforces that, and
    a plain object raises TypeError on assignment.
    """

    def __init__(self, body=None, headers=None) -> None:
        super().__init__("gateway denial")
        self.body = body
        self.response = _FakeResponse(headers or {})


def _denial(message: str, *, body=None, headers=None) -> TokenLimitExceeded:
    """A TokenLimitExceeded chained to a gateway denial, as anton raises it."""
    exc = TokenLimitExceeded(message)
    exc.__cause__ = _FakeCause(body=body, headers=headers)
    return exc


class _AlwaysThrottled:
    """Every call 429s with no reason at all — the `unknown` path."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate_object_code(self, *args, **kwargs):
        self.calls += 1
        raise TokenLimitExceeded("Server returned 429 — rate limit exceeded for key.")


class _AlwaysDenied:
    """Every call fails with a specific gateway denial."""

    def __init__(self, exc_factory) -> None:
        self.calls = 0
        self._exc_factory = exc_factory

    async def generate_object_code(self, *args, **kwargs):
        self.calls += 1
        raise self._exc_factory()


class _ThrottledOnce:
    """429s once, then succeeds — a TPM velocity window that cleared."""

    def __init__(self, verdict) -> None:
        self.calls = 0
        self._verdict = verdict

    async def generate_object_code(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise TokenLimitExceeded("Server returned 429 — please slow down.")
        return self._verdict


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """The real pause is 20s per case; tests must not pay it."""
    monkeypatch.setattr(ev, "_THROTTLE_RETRY_SLEEP_S", 0.0)


@pytest.fixture(autouse=True)
def _fresh_retry_budget(monkeypatch):
    """The throttle budget is session-scoped by design, so reset it per test.

    Without this, whichever test runs first spends the budget and the rest
    silently exercise the exhausted path — passing for the wrong reason.
    """
    monkeypatch.setattr(ev, "_throttle_retries_used", 0)


@pytest.mark.asyncio
async def test_a_persistent_429_becomes_a_marked_skip():
    llm = _AlwaysThrottled()

    with pytest.raises(pytest.skip.Exception) as caught:
        await ev._verdict(llm, ev._CASES[0])

    # The marker is what the CI guard greps out of the junit report to keep a
    # starved run green-with-a-warning instead of red.
    assert ev._GATEWAY_UNAVAILABLE in str(caught.value)
    # Retried exactly once before giving up — not zero (which cascades into an
    # all-skip green) and not forever (which hangs the job on a dead wallet).
    assert llm.calls == 2, f"expected one retry then skip, got {llm.calls} calls"


@pytest.mark.asyncio
async def test_a_transient_429_is_retried_and_the_eval_still_runs():
    """The case that matters: a throttle must NOT cost us the eval.

    Skipping on the first 429 is what produced a green job with zero cases
    executed, because the next case fires into the same throttle window.
    """
    sentinel = object()
    llm = _ThrottledOnce(sentinel)

    result = await ev._verdict(llm, ev._CASES[0])

    assert result is sentinel
    assert llm.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,carrier",
    [
        ("wallet_empty", "body-code"),
        ("wallet_empty", "header"),
        ("included_allowance_exhausted", "body-code"),
        ("included_allowance_exhausted", "header"),
        ("included_allowance_exhausted", "envelope"),
    ],
)
async def test_a_starved_key_skips_without_burning_the_retry(reason, carrier):
    """wallet_empty / allowance_exhausted will not clear inside this run.

    Retrying them just adds a pause before the identical answer, so the retry is
    reserved for the one denial that is actually transient.
    """
    kwargs = {
        "body-code": {"body": {"code": reason}},
        "envelope": {"body": {"error": {"code": reason}}},
        "header": {"headers": {"x-mindshub-reason": reason}},
    }[carrier]
    llm = _AlwaysDenied(lambda: _denial(f"Server returned — {reason}", **kwargs))

    with pytest.raises(pytest.skip.Exception) as caught:
        await ev._verdict(llm, ev._CASES[0])

    assert ev._GATEWAY_UNAVAILABLE in str(caught.value)
    assert llm.calls == 1, f"starved key should not be retried, got {llm.calls} calls"


@pytest.mark.asyncio
async def test_a_velocity_throttle_honours_retry_after(monkeypatch):
    """`rate_limited` carries the server's own backoff — use it, don't guess."""
    slept: list[float] = []

    async def _record(seconds):
        slept.append(seconds)

    monkeypatch.setattr(ev.asyncio, "sleep", _record)
    sentinel = object()
    calls = {"n": 0}

    class _ThrottledThenOk:
        async def generate_object_code(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _denial(
                    "Server returned 429 — Rate limit exceeded. Please slow down.",
                    body={"code": "rate_limited"},
                    headers={"x-mindshub-reason": "rate_limited", "retry-after": "5"},
                )
            return sentinel

    assert await ev._verdict(_ThrottledThenOk(), ev._CASES[0]) is sentinel
    assert slept == [5.0], f"expected the 5s Retry-After hint, slept {slept}"


@pytest.mark.asyncio
async def test_an_absurd_retry_after_is_capped(monkeypatch):
    """A job that naps for an hour is indistinguishable from a hung one."""
    slept: list[float] = []

    async def _record(seconds):
        slept.append(seconds)

    monkeypatch.setattr(ev.asyncio, "sleep", _record)
    llm = _AlwaysDenied(
        lambda: _denial(
            "429",
            body={"code": "rate_limited"},
            headers={"retry-after": "3600"},
        )
    )

    with pytest.raises(pytest.skip.Exception):
        await ev._verdict(llm, ev._CASES[0])

    assert slept == [ev._THROTTLE_RETRY_CAP_S], f"expected the cap, slept {slept}"


@pytest.mark.asyncio
async def test_an_http_date_retry_after_falls_back_to_the_default(monkeypatch):
    """Retry-After may be an HTTP-date; a bad parse must not mask the denial."""
    slept: list[float] = []

    async def _record(seconds):
        slept.append(seconds)

    monkeypatch.setattr(ev.asyncio, "sleep", _record)
    monkeypatch.setattr(ev, "_THROTTLE_RETRY_SLEEP_S", 7.0)
    llm = _AlwaysDenied(
        lambda: _denial(
            "429",
            body={"code": "rate_limited"},
            headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"},
        )
    )

    with pytest.raises(pytest.skip.Exception):
        await ev._verdict(llm, ev._CASES[0])

    assert slept == [7.0], f"expected the fallback pause, slept {slept}"


@pytest.mark.asyncio
async def test_a_sustained_throttle_stops_pausing_once_the_session_budget_is_spent(
    monkeypatch,
):
    """The per-call retry alone let a sustained throttle sleep for ~37 minutes.

    `_verdicts` calls `_verdict` 37 times per session (4 cases x 3 + STUCK x 6,
    both models, plus the budget test), each previously entitled to its own
    pause of up to `_THROTTLE_RETRY_CAP_S`. The job then still reported green
    with zero cases executed — a slow wrong answer instead of a fast one.
    """
    slept: list[float] = []

    async def _record(seconds):
        slept.append(seconds)

    monkeypatch.setattr(ev.asyncio, "sleep", _record)
    monkeypatch.setattr(ev, "_THROTTLE_RETRY_BUDGET", 2)
    monkeypatch.setattr(ev, "_THROTTLE_RETRY_SLEEP_S", 1.0)

    # Five independent cases all throttled: only the first two may pause.
    for _ in range(5):
        llm = _AlwaysDenied(
            lambda: _denial("429 throttled", body={"code": "rate_limited"})
        )
        with pytest.raises(pytest.skip.Exception):
            await ev._verdict(llm, ev._CASES[0])

    assert len(slept) == 2, (
        f"session budget is 2, so only 2 pauses should happen; slept {slept}"
    )
    assert ev._throttle_retries_used == 2


def test_the_workflow_bounds_the_job_so_a_hang_cannot_hold_a_runner():
    """Belt for the braces: even a bug in the budget cannot burn 6 hours.

    GitHub's default job timeout is 360 minutes.
    """
    workflow = _WORKFLOW.read_text()
    found = re.search(r"timeout-minutes:\s*(\d+)", workflow)

    assert found, "verdict-eval has no timeout-minutes; the default is 360"
    assert int(found.group(1)) <= 30, (
        f"timeout-minutes={found.group(1)} is too loose for a ~90s job"
    )


def test_single_valued_cases_still_demand_that_exact_verdict():
    """The acceptable-set change must not have loosened the single-valued cases.

    Every fixture but the two hallucinated-success ones exists because the
    *wrong label is the bug* — a recovered error judged INCOMPLETE
    force-continues (ENG-1134), a genuine question judged INCOMPLETE makes the
    agent answer itself (ENG-716), an environment wall judged INCOMPLETE walks
    into the wall (ENG-836). Those must stay single-valued.
    """
    by_name = {c.name: c for c in ev._CASES}

    single = {
        "recovered_tool_error": "COMPLETE",
        "genuine_question": "WAITING",
        "environment_wall": "STUCK",
        "stopped_partway": "INCOMPLETE",
        # ENG-1633: INCOMPLETE here IS the incident — a complete reply
        # failed as "truncated" because the TRANSCRIPT was clipped, not
        # the answer.
        "long_complete_reply": "COMPLETE",
        # ENG-2686: INCOMPLETE here IS the incident — the honest N/D reply
        # was force-continued into invented fares. And the three risk
        # controls guard the over-correction: a one-shot give-up must not
        # become STUCK, a requested estimate and a computed total must not
        # become "unsourced".
        "honest_unobtainable_data": "STUCK",
        "one_attempt_give_up": "INCOMPLETE",
        "user_requested_estimate": "COMPLETE",
        "computed_total_clipped_source": "COMPLETE",
        # And the shadow-replay finding: a failed attempt followed by asking
        # the user is WAITING. INCOMPLETE here IS ENG-716's incident.
        "asks_user_after_one_failed_attempt": "WAITING",
    }
    assert set(single) | {"implied_success_data_never_arrived", "disclaimered_fabrication"} == set(by_name), (
        "a new case was added without deciding here whether it is single-valued"
    )
    for name, verdict in single.items():
        assert by_name[name].acceptable == (verdict,), (
            f"{name} must accept only {verdict}; the alternative label IS the "
            f"incident it guards against"
        )


def test_only_the_hallucinated_success_cases_accept_two_verdicts():
    """Two deliberate exceptions, both the same invariant, and it must not
    spread quietly.

    ENG-1134's safeguard is "never accept a hallucinated success as done", which
    both INCOMPLETE and STUCK satisfy. ENG-2686 added the disclaimered variant
    of the same incident (invented figures passed as COMPLETE behind an
    "indicative" label). COMPLETE and WAITING must never be acceptable in
    either — those are the failure.
    """
    multi = [c for c in ev._CASES if len(c.acceptable) > 1]

    assert [c.name for c in multi] == [
        "implied_success_data_never_arrived",
        "disclaimered_fabrication",
    ], (
        f"exactly two cases may accept multiple verdicts; found "
        f"{[c.name for c in multi]}"
    )
    for case in multi:
        acceptable = set(case.acceptable)
        assert acceptable == {"INCOMPLETE", "STUCK"}, case.name
        assert not acceptable & {"COMPLETE", "WAITING"}, (
            f"{case.name}: accepting COMPLETE or WAITING here would delete the "
            "hallucinated-success safeguard"
        )


def test_the_recovered_case_asks_one_unambiguous_question():
    """The fixture's user_message and its history turn must not drift apart.

    The verifier reads the transcript, so a stale copy in `history` would keep
    feeding it the ambiguous phrasing that made this case flake.
    """
    case = {c.name: c for c in ev._CASES}["recovered_tool_error"]
    first_user_turn = case.history[0]

    assert first_user_turn["role"] == "user"
    assert first_user_turn["content"] == case.user_message, (
        "history[0] must repeat user_message verbatim; they have drifted"
    )
    # The ambiguity was "before Tesla's IPO" trailing a single comparison
    # clause, where it could bind either one figure or both.
    assert "compared to Tesla before" not in case.user_message


def test_the_marker_matches_the_one_the_workflow_greps_for():
    """The string is a contract across two files; comments are not enforcement.

    Drift fails safe — an unmatched marker falls to the guard's red branch — but
    it silently disables the out-of-money handling, which is exactly the kind of
    quiet degradation ENG-1334 is about.
    """
    workflow = _WORKFLOW.read_text()
    found = re.search(r'MARKER\s*=\s*"([A-Z_]+)"', workflow)

    assert found, f"no MARKER assignment in {_WORKFLOW.name}; did the guard move?"
    assert found.group(1) == ev._GATEWAY_UNAVAILABLE, (
        f"marker drift: workflow greps for {found.group(1)!r} but the eval emits "
        f"{ev._GATEWAY_UNAVAILABLE!r}, so starved runs would stop being recognised"
    )


# --- The served-model pin (ENG-1687) -------------------------------------
#
# Unit-level on purpose. The live eval exercises this on every call, but only
# when a key is present — and the whole point of the pin is to be the thing that
# still works when nobody is watching. These run in the default unit suite.


@pytest.fixture(autouse=True)
def _clean_served_map():
    """`_SERVED` is module state on the eval, shared across this file's tests."""
    ev._SERVED.clear()
    yield
    ev._SERVED.clear()


def test_the_pinned_model_is_recorded_and_accepted():
    ev._check_served_model("haiku", ev._EXPECTED_SERVED["haiku"])

    assert ev._SERVED == {"haiku": ev._EXPECTED_SERVED["haiku"]}


def test_a_repointed_alias_fails_naming_both_models():
    """The failure has to be readable by someone who has never seen this file.

    "assert x == y" over two model ids does not tell them a catalog repoint is a
    normal event, that no verdict in the run is evidence any more, or that the
    fix is to re-read the slot's rationale rather than to bump the constant.
    """
    with pytest.raises(ev.AliasRepointed) as caught:
        ev._check_served_model("mindshub_air", "kimi-k2p6")

    message = str(caught.value)
    assert "mindshub_air" in message
    assert "kimi-k2p6" in message, "must name what is served now"
    assert ev._EXPECTED_SERVED["mindshub_air"] in message, "must name the pin"
    assert "ENG-1687" in message
    # Recorded before raising: the report at session end is most wanted on the
    # run that failed, and a raise that skipped the record would hide it.
    assert ev._SERVED == {"mindshub_air": "kimi-k2p6"}


def test_a_repoint_is_not_swallowed_by_the_verdict_retry_paths():
    """`_verdict` absorbs truncation and throttling. It must not absorb this.

    A repoint typed as `StructuredOutputError` would be retried at a bigger
    budget; typed as `TokenLimitExceeded` it would become a marked skip, and a
    fully-skipped matrix reports GREEN by design (ENG-1334's out-of-money
    branch). Either one restores the silent pass this check exists to break.
    """
    from anton.core.llm.provider import StructuredOutputError

    assert not issubclass(ev.AliasRepointed, (StructuredOutputError, TokenLimitExceeded))
    assert not issubclass(ev.AliasRepointed, pytest.skip.Exception)


def test_an_alias_outside_the_pin_map_is_recorded_but_not_asserted():
    """`VERIFIER_EVAL_*_MODEL` exists for one-off runs against another alias.

    Failing those would make the escape hatch unusable, so an unpinned alias is
    reported and never asserted.
    """
    ev._check_served_model("gpt-terra", "gpt-5.6-terra")

    assert ev._SERVED == {"gpt-terra": "gpt-5.6-terra"}


def test_an_alias_echo_is_recorded_as_not_disclosed_and_never_a_repoint(monkeypatch):
    """ENG-2892: the gateway now returns the alias as `model`. That is no
    information about the served model, not a repoint — the check must not
    raise, must record the blindness where the report will show it, and must
    keep failing a GENUINE repoint (a different real id)."""
    monkeypatch.setattr(ev, "_SERVED", {})
    ev._check_served_model("haiku", "haiku")  # must not raise
    assert "not disclosed" in ev._SERVED["haiku"]
    assert ev._SERVED["haiku"].startswith("haiku")
    # The guard is still armed for a real repoint of the same slot.
    with pytest.raises(ev.AliasRepointed):
        ev._check_served_model("haiku", "claude-sonnet-5")
    # And the pinned id itself is still accepted and recorded verbatim.
    ev._check_served_model("haiku", "claude-haiku-4-5-20251001")
    assert ev._SERVED["haiku"] == "claude-haiku-4-5-20251001"
    # A confirmed match wins over the echo test: when the alias IS the pinned
    # real id (a one-off `VERIFIER_EVAL_*_MODEL=<real id>` run), the gateway
    # did disclose and the report must not claim blindness (#490 self-review,
    # finding 2 — the echo branch used to run first and mislabel it).
    monkeypatch.setitem(ev._EXPECTED_SERVED, "gpt-5.6-luna", "gpt-5.6-luna")
    ev._check_served_model("gpt-5.6-luna", "gpt-5.6-luna")
    assert ev._SERVED["gpt-5.6-luna"] == "gpt-5.6-luna"


@pytest.mark.parametrize("served", [None, "", 0, object()])
def test_a_missing_served_id_is_not_treated_as_a_repoint(served):
    """A provider that omits `model` says nothing about which model ran.

    Reading that as a repoint would fail the eval on a provider property. It
    also must not be RECORDED, or the session report would claim an alias
    resolved to nothing.
    """
    ev._check_served_model("haiku", served)

    assert ev._SERVED == {}


@pytest.mark.skipif(
    bool(
        os.environ.get("VERIFIER_EVAL_FIRST_PARTY_MODEL")
        or os.environ.get("VERIFIER_EVAL_NARRATING_MODEL")
    ),
    reason="an env override deliberately runs an unpinned alias",
)
def test_every_pinned_alias_is_one_the_matrix_actually_runs():
    """A pin on an alias no longer in the matrix checks nothing and reads as
    coverage. Only fires if someone edits the matrix and forgets the map."""
    assert set(ev._EXPECTED_SERVED) <= set(ev._MODELS), (
        "_EXPECTED_SERVED pins an alias the matrix does not run: "
        f"{sorted(set(ev._EXPECTED_SERVED) - set(ev._MODELS))}"
    )


def test_the_matrix_slots_resolve_to_different_models():
    """The pin catches the repoint; this catches the state the repoint CAUSED.

    ENG-1687 is not "an alias moved" — it is "both slots ended up holding one
    model and the gate stayed green". `_EXPECTED_SERVED` reds on the move, but
    the obvious way to clear that red is to update the map to the new id, and if
    the new id is the other slot's model the eval goes back to running one model
    twice with a perfectly green pin. Distinctness is what the ticket is about,
    so assert it rather than leaving it to hold by accident.

    Two ways to arrive at one model in two slots, both covered:
      - identical aliases (someone set both `VERIFIER_EVAL_*_MODEL` the same)
      - distinct aliases pinned to the same served id
    """
    assert len(set(ev._MODELS)) == len(ev._MODELS), (
        f"the matrix runs the same alias twice: {ev._MODELS}. Unset one of the "
        "VERIFIER_EVAL_*_MODEL overrides."
    )
    pinned = [ev._EXPECTED_SERVED[a] for a in ev._MODELS if a in ev._EXPECTED_SERVED]
    assert len(set(pinned)) == len(pinned), (
        f"the matrix's slots are pinned to the same model: {pinned}. That is "
        "ENG-1687's end state, not its fix — re-pick a slot instead of pointing "
        "both at one model."
    )


@pytest.mark.parametrize("alias", ["gpt-terra", "mindshub_air"])
def test_a_served_id_cannot_inject_lines_into_the_report(alias):
    """`response.model` is remote text, and it lands in two reports.

    Unsanitized it reaches an exception message and `GITHUB_STEP_SUMMARY`, so a
    value carrying a newline writes extra markdown lines into a CI artifact —
    e.g. a bogus "✅" row for an alias that was never checked. anton already
    sanitizes this exact field before it reaches a prompt
    (`identity.sanitize_model_name`); this asserts the eval reuses it rather than
    hand-rolling the check.

    Both branches are covered because they format the value in different places:
    an unpinned alias only records it, a pinned one also interpolates it into the
    raise.
    """
    poisoned = "evil-model\n- `haiku` -> `totally-fine` OK"

    try:
        ev._check_served_model(alias, poisoned)
    except ev.AliasRepointed as exc:
        assert "\n" not in str(exc), "the raise carries a raw newline"

    recorded = ev._SERVED.get(alias)
    assert recorded is not None, "a poisoned id must still be recorded, not dropped"
    assert "\n" not in recorded, f"newline survived sanitisation: {recorded!r}"
    assert len(recorded) <= 80, "the sanitiser's length cap did not apply"


def test_only_the_one_attempt_control_is_recorded_not_gated_on_a_model():
    """The skip path must not spread quietly.

    One control carries a measured label slip on haiku and is recorded there
    rather than gated. Every incident case stays gated on every model: a
    fabrication fixture that is only "recorded" somewhere ships the
    fabrication there.
    """
    skipping = [c for c in ev._CASES if c.skip_models]
    assert [c.name for c in skipping] == ["one_attempt_give_up"], (
        f"only the one-attempt control may skip a model; found "
        f"{[c.name for c in skipping]}"
    )
    case = skipping[0]
    assert case.skip_models == ("haiku",)
    # Still gated somewhere in the matrix, at the trade-off guard's count.
    assert ev._NARRATING_MODEL not in case.skip_models
    assert ev._runs_for(case) == 6


def test_run_overrides_are_only_the_premature_give_up_guard():
    overridden = [c for c in ev._CASES if c.runs is not None]
    assert [c.name for c in overridden] == ["one_attempt_give_up"]


def test_fabrication_guards_run_at_the_higher_count():
    """The two hallucinated-success cases guard a low-rate laundering
    regression, and N=3 misses a 1-in-6 COMPLETE rate 58% of the time."""
    by_name = {c.name: c for c in ev._CASES}
    for name in ("implied_success_data_never_arrived", "disclaimered_fabrication"):
        assert ev._runs_for(by_name[name]) == ev._STUCK_RUNS, name
    # And the plain single-valued controls still run at the default.
    assert ev._runs_for(by_name["stopped_partway"]) == ev._RUNS


def test_recorded_not_gated_pairs_are_excluded_at_parametrization_not_skipped():
    """verifier-eval.yml's out-of-money branch passes only when every skip in
    the junit carries GATEWAY_UNAVAILABLE, so a design skip would turn a
    starved run from a warning into a red misconfiguration.
    """
    ids = {p.id for p in ev._MATRIX}
    assert "one_attempt_give_up-haiku" not in ids
    assert "one_attempt_give_up-mindshub_air" in ids
    assert len(ev._MATRIX) == len(ev._CASES) * len(ev._MODELS) - 1
    import inspect

    assert "pytest.skip(" not in inspect.getsource(ev.test_verdict)


# --- ENG-2863: the two-tier scope decision in verifier-eval.yml ------------

import importlib.util as _ilu

_SCOPE_PATH = Path(__file__).resolve().parent.parent / "scripts/verifier_eval_scope.py"
_spec = _ilu.spec_from_file_location("verifier_eval_scope", _SCOPE_PATH)
scope = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(scope)


def test_every_scope_marker_resolves_to_a_span_in_the_code():
    """A renamed function must fail here, not silently turn the matrix off.

    The workflow decides full-matrix vs guard-only by intersecting the PR's
    hunks with these symbols' line spans. A marker that resolves to nothing
    would make every PR touching that code run the one-call guard, and the
    rubric would be unguarded again (the ENG-1334 failure in a new coat).
    """
    root = Path(__file__).resolve().parent.parent
    for path, names in scope.SPAN_MARKERS.items():
        spans = scope.marker_spans((root / path).read_text(), names)
        missing = [n for n in names if n not in spans]
        assert not missing, f"{path}: scope markers resolve to nothing: {missing}"
    # Pinned by EQUALITY, not subset: review of #486 deleted six entries one
    # at a time and the subset asserts passed every time. Removing a marker is
    # a deliberate edit to this test, with a reason.
    assert set(scope.SPAN_MARKERS) == {"anton/core/session.py", "anton/core/llm/client.py"}
    assert set(scope.SPAN_MARKERS["anton/core/session.py"]) == {
        "_VerifierVerdict", "_VERIFIER_TOKEN_BUDGETS", "_VERIFIER_NO_PREAMBLE",
        "_VERIFIER_JUDGMENT_RUBRIC", "_build_verify_request", "_render_verify_transcript",
        "_render_tool_result_content", "_clip_keep_cause",
    }
    # The forced-tool-call body, not only the delegators that call it: a
    # tool_choice mutation inside _generate_object_with read "guard" before.
    assert set(scope.SPAN_MARKERS["anton/core/llm/client.py"]) == {
        "_generate_object_with", "_call_with_auth_confirmation",
        "generate_object", "generate_object_code",
    }
    assert set(scope.ALWAYS_FULL) == {
        "tests/test_verifier_verdict_live.py", "anton/core/llm/structured.py",
        "scripts/verifier_eval_scope.py",
    }


def test_scope_counts_a_body_edit_inside_a_marker_and_ignores_one_outside():
    src = (
        "X = 1\n"
        "def _render_verify_transcript(h):\n"
        "    a = 1\n"
        "    b = 2\n"
        "    return a + b\n"
        "\n"
        "async def turn_stream(self):\n"
        "    y = _VERIFIER_TOKEN_BUDGETS\n"
        "    return y\n"
    )
    spans = scope.marker_spans(src, ("_render_verify_transcript", "_VERIFIER_TOKEN_BUDGETS"))
    assert spans == {"_render_verify_transcript": (2, 5)}  # the constant is only *used* here
    inside = scope.parse_hunks("@@ -4 +4 @@\n-    b = 2\n+    b = 3\n")
    outside = scope.parse_hunks("@@ -8 +8 @@\n-    y = _VERIFIER_TOKEN_BUDGETS\n+    y = 0\n")
    assert scope.touched(inside, spans, spans) == {"_render_verify_transcript"}
    assert scope.touched(outside, spans, spans) == set()
    # A pure insertion (zero-length old side) inside the span counts too.
    insertion = scope.parse_hunks("@@ -3,0 +4 @@\n+    c = 9\n")
    assert scope.touched(insertion, spans, spans) == {"_render_verify_transcript"}
    # And a pure DELETION whose zero-length new side lands on the span's first
    # line: old side is outside the span, so only the max(length, 1) treatment
    # of the new side can catch it (review of #486: the insertion case above
    # passed via the non-empty new side and never exercised that guard).
    deletion = scope.parse_hunks("@@ -7 +2,0 @@\n-    gone = 1\n")
    assert scope.touched(deletion, spans, spans) == {"_render_verify_transcript"}


def test_guard_mode_selects_exactly_the_truncation_guard():
    workflow = _WORKFLOW.read_text()
    assert "scripts/verifier_eval_scope.py" in workflow
    assert "${{ steps.scope.outputs.pytest_args }}" in workflow
    assert "fetch-depth: 0" in workflow
    # The -k expression must select an existing test, and only one.
    matches = [n for n in dir(ev) if n.startswith("test_") and scope.GUARD_K in n]
    assert matches == ["test_narrating_model_reaches_a_verdict_at_shipped_budgets"]
    # And the scope step never reaches for pytest.skip.
    step = workflow.split("Decide eval scope")[1].split("Run verdict-quality eval")[0]
    assert "pytest.skip" not in step


def test_a_failing_scope_decision_falls_back_to_the_full_matrix(tmp_path, monkeypatch):
    """Fail CLOSED. If the decision cannot be made (bad revision, git error,
    unparsable file), the answer is the full matrix — never a red job for an
    unrelated reason, and never the one-call guard."""
    out = tmp_path / "gh_output"
    monkeypatch.setattr(scope, "decide", lambda b, h: (_ for _ in ()).throw(RuntimeError("boom")))
    rc = scope.main(["--base", "x", "--head", "y", "--github-output", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "scope=full" in text and "pytest_args=\n" in text


def test_decide_diffs_from_the_merge_base_not_the_base_tip(tmp_path, monkeypatch):
    """pull_request.base.sha is the base branch TIP. After the PR branched, the
    base gained a commit touching an ALWAYS_FULL file; the PR itself touched only
    turn_stream. Against the tip that reads "full" (the base's own change shows
    up as a reverse hunk); against the merge base it is "guard"."""
    import subprocess

    def git(*a):
        return subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q", "-b", "staging")
    git("config", "user.email", "t@example.com"); git("config", "user.name", "t")
    (tmp_path / "anton/core/llm").mkdir(parents=True); (tmp_path / "tests").mkdir(); (tmp_path / "scripts").mkdir()
    session = tmp_path / "anton/core/session.py"
    session.write_text(
        "def _render_verify_transcript(h):\n    return h\n\n"
        "async def turn_stream(self):\n    return 1\n"
    )
    (tmp_path / "anton/core/llm/client.py").write_text("def generate_object():\n    pass\n")
    (tmp_path / "tests/test_verifier_verdict_live.py").write_text("# eval\n")
    git("add", "."); git("commit", "-q", "-m", "base")
    branch_point = git("rev-parse", "HEAD")
    # The PR: touches turn_stream only.
    git("checkout", "-q", "-b", "pr")
    session.write_text(session.read_text().replace("return 1", "return 2"))
    git("commit", "-q", "-am", "pr edit")
    pr_head = git("rev-parse", "HEAD")
    # The base moves on and touches an ALWAYS_FULL file.
    git("checkout", "-q", "staging")
    (tmp_path / "tests/test_verifier_verdict_live.py").write_text("# eval changed on staging\n")
    git("commit", "-q", "-am", "verifier work landed on staging")
    base_tip = git("rev-parse", "HEAD")
    assert base_tip != branch_point

    monkeypatch.chdir(tmp_path)
    verdict, reasons = scope.decide(base_tip, pr_head)
    assert verdict == "guard", reasons
    # And a PR that really touches a marker still reads full from the same base tip.
    git("checkout", "-q", "pr")
    session.write_text(session.read_text().replace("return h", "return list(h)"))
    git("commit", "-q", "-am", "renderer edit")
    verdict, reasons = scope.decide(base_tip, git("rev-parse", "HEAD"))
    assert verdict == "full" and any("_render_verify_transcript" in r for r in reasons), reasons
