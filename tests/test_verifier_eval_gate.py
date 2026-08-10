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


class _AlwaysThrottled:
    """Every call 429s — a wallet that is genuinely empty for this run."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate_object_code(self, *args, **kwargs):
        self.calls += 1
        raise TokenLimitExceeded("Server returned 429 — rate limit exceeded for key.")


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
