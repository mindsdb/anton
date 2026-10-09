"""Live eval: does the finish-authorization check read user intent correctly?

When a turn reaches its step or token limit, anton asks the cheap coding model
whether the turn's message already says "keep going" (see
`_build_finish_authorization_request`). A false positive spends the user's
allowance without asking; a false negative only costs one extra question. So
the negatives here are the ones that matter most, and the first fixture is the
measured failure: a pacing instruction read as "keep going".

Each case is fired at the real MindsHub gateway with the production prompt.
Gating, client wiring and model pinning come from the verifier eval: requires
``MINDSHUB_API_KEY`` (environment or repo-root ``.env``), auto-skips without it.
"""

from __future__ import annotations

import pytest

from anton.core.session import (
    _VERIFIER_TOKEN_BUDGETS,
    _FinishAuthorization,
    _build_finish_authorization_request,
    _finish_authorized,
)

from tests.test_verifier_verdict_live import _KEY, _client

pytestmark = pytest.mark.skipif(
    not _KEY, reason="MINDSHUB_API_KEY not set — live finish-authorization eval skipped"
)

MODELS = ("mindshub_air", "haiku")

#: A case whose answer may go either way. Only for messages where a "no" is the
#: safe miss (one extra question) and pushing the prompt toward "yes" made a
#: harmful miss elsewhere: listing a bare "finish" as an example made
#: "can you finish the report section?" read as "keep going".
EITHER = None

# (label, message, should the turn continue past its limit without asking?)
CASES: list[tuple[str, str, bool | None]] = [
    # The incident's own replies.
    ("incident-keep-going", "Keep going, don't stop until you are finished.", True),
    ("incident-go", "Go and don't stop until you hae something for me to review!!!!", True),
    ("incident-finish", "Finish", EITHER),
    ("keep-going", "keep going", True),
    ("just-finish", "Just finish it, no need to check with me.", True),
    ("until-done", "Work on this until it's completely done, don't ask me in between.", True),
    (
        "task-then-dont-stop",
        "Convert the attached RFP to match the strategy with tracked changes, plus a "
        "change summary and gaps list. Don't stop until all three are saved.",
        True,
    ),
    ("yes-continue", "Yes, continue", True),
    # Pacing and order are not authorization.
    (
        "primes-pacing",
        "Do this one step at a time, with a separate scratchpad run for each step and "
        "never combining steps: compute the 1st through 16th prime numbers, one prime "
        "per run, and print each before moving on.",
        False,
    ),
    ("step-by-step", "Walk me through it step by step.", False),
    ("check-each", "Check each file before moving on to the next one.", False),
    ("carefully", "Take your time and be thorough.", False),
    # Ordinary task requests.
    (
        "rfp-request",
        "Please convert the whole Riverside Remediation RFP to match the Strategy using "
        "real Word tracked changes. Deliver a tracked-changes RFP plus a change summary "
        "and a gaps list.",
        False,
    ),
    ("report", "Build a quarterly sales report from the CSV and chart revenue by region.", False),
    ("finish-the-report", "Can you finish the report section on Q3 revenue?", False),
    # Continue, but differently / not now.
    ("different-approach", "Keep going but try a different approach, this one isn't working.", False),
    ("continue-tomorrow", "Let's continue tomorrow.", False),
    ("stop", "Stop here and summarize what you have.", False),
    ("pause", "Pause for now, I need to check something.", False),
    ("question", "How much longer will this take?", False),
    ("should-you-continue", "Should you keep going or is this good enough?", False),
    ("only-section-2", "Only finish section 2, leave the rest.", False),
]


#: Every run of a case must give the expected answer: a case that flips between
#: runs is a coin flip, not a guard (same policy as the verifier eval).
RUNS = 3


@pytest.mark.parametrize("run", range(RUNS))
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("label,message,expected", CASES, ids=[c[0] for c in CASES])
async def test_finish_authorization(model, label, message, expected, run):
    system, messages = _build_finish_authorization_request(message)
    verdict = await _client(model).generate_object_code(
        _FinishAuthorization,
        system=system,
        messages=messages,
        max_tokens=_VERIFIER_TOKEN_BUDGETS[0],
    )
    got = _finish_authorized(verdict, message)
    if expected is EITHER:
        return
    assert got is expected, (
        f"{model} read {label!r} as authorized={verdict.authorized} quote={verdict.quote!r}"
    )
