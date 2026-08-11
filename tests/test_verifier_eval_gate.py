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
"""

from __future__ import annotations

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
    """The acceptable-set change must not have loosened the other four cases.

    Four of the five fixtures exist because the *wrong label is the bug* — a
    recovered error judged INCOMPLETE force-continues (ENG-1134), a genuine
    question judged INCOMPLETE makes the agent answer itself (ENG-716), an
    environment wall judged INCOMPLETE walks into the wall (ENG-836). Those
    must stay single-valued.
    """
    by_name = {c.name: c for c in ev._CASES}

    single = {
        "recovered_tool_error": "COMPLETE",
        "genuine_question": "WAITING",
        "environment_wall": "STUCK",
        "stopped_partway": "INCOMPLETE",
    }
    for name, verdict in single.items():
        assert by_name[name].acceptable == (verdict,), (
            f"{name} must accept only {verdict}; the alternative label IS the "
            f"incident it guards against"
        )


def test_only_the_hallucinated_success_case_accepts_two_verdicts():
    """One deliberate exception, and it must not spread quietly.

    ENG-1134's safeguard is "never accept a hallucinated success as done", which
    both INCOMPLETE and STUCK satisfy. COMPLETE and WAITING must never be
    acceptable there — those are the failure.
    """
    multi = [c for c in ev._CASES if len(c.acceptable) > 1]

    assert [c.name for c in multi] == ["implied_success_data_never_arrived"], (
        f"exactly one case may accept multiple verdicts; found "
        f"{[c.name for c in multi]}"
    )
    acceptable = set(multi[0].acceptable)
    assert acceptable == {"INCOMPLETE", "STUCK"}
    assert not acceptable & {"COMPLETE", "WAITING"}, (
        "accepting COMPLETE or WAITING here would delete the ENG-1134 safeguard"
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
