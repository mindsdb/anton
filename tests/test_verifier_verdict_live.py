"""Live behavioural eval for the completion verifier's verdict quality (ENG-1211).

Every other verifier test in this repo asserts *wording* — that a rubric string
reaches the model. None of them asserts the judgment the model then makes, and
the rubric changed four times in a week (ENG-716, ENG-1081, ENG-1134,
ENG-1155). This module closes that gap: each case is a committed transcript
fixture drawn from a real incident, fired at the real MindsHub gateway through
the exact production code path (``_build_verify_request`` +
``generate_object_code`` with the real ``_VerifierVerdict`` schema and the
shipped ``_VERIFIER_TOKEN_BUDGETS``), asserting the returned ``status``.

Ground truth is anchored to the fixtures, not to live behaviour or drifting
real-world facts (ENG-381 lesson): the transcript is the input, the expected
status is the label.

Gating: requires ``MINDSHUB_API_KEY`` in the environment (or repo-root
``.env``). Without it the module auto-skips, so the default CI unit run is
unaffected — **unless** ``VERIFIER_EVAL_REQUIRE_LIVE=1``, which turns a missing
key into a hard collection error. ``.github/workflows/verifier-eval.yml`` sets
that for first-party PRs, so the gate can no longer report success without
executing (ENG-1334). Fork PRs cannot receive the secret, so they still skip.

Model matrix: one first-party alias (``haiku`` — enforces the forced
``tool_choice`` structurally) and one narrating alias (``mindshub_air`` —
narrates into ``content`` before the tool call). ENG-1081 measured a clean
9-vs-3 split between those populations; an eval that only ran on haiku would
have missed the entire ENG-1081 incident class.

Non-determinism policy (decided up front, per the ticket): every case asserts
N-of-N identical verdicts — a case that can't produce the same verdict N times
in a row is not a regression guard, it's a coin flip. N defaults to 3; the
STUCK case runs 6, because its base rate under the pre-ENG-836 rubric was
measured at ~1-in-5 (Kiranam session: 4 COMPLETE / 4 INCOMPLETE / 1 STUCK on
one blocker), so a single run passes ~20% of the time by luck. The STUCK case
therefore also doubles as ENG-836's before/after measurement.

Cost note: a full run is ~50 verdict calls over ~2-4k-token transcripts on
cheap aliases, ~4 minutes wall. Because the key is also picked up from the
repo-root ``.env``, a plain ``pytest tests/`` on a machine with a keyed
``.env`` runs this live — same behaviour as ``test_web_tools_live.py``.
Deselect with ``-k "not verdict_live"`` or unset the key to skip.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

# Load .env once so plain os.environ reads pick up keys the developer put in
# the repo-root .env (same pattern as tests/test_web_tools_live.py).
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except Exception:
    pass

from anton.config.settings import AntonSettings
from anton.core.llm.client import LLMClient
from anton.core.llm.provider import StructuredOutputError, TokenLimitExceeded
from anton.core.session import (
    _VERIFIER_TOKEN_BUDGETS,
    _VerifierVerdict,
    _build_verify_request,
)

_KEY = os.environ.get("MINDSHUB_API_KEY")
_BASE = os.environ.get("MINDSHUB_BASE_URL", "https://api.mindshub.ai")

# Two different callers want opposite things from a missing key, and conflating
# them is what let this suite report success while testing nothing (ENG-1334).
#
#   A developer running `pytest tests/` wants it to skip. They have no key, they
#   are not testing the verifier, and failing their whole run would be rude.
#
#   CI wants it to FAIL. A gate that goes green when it cannot run is worse than
#   no gate: the board looks guarded. Measured 2026-08-10 — eight consecutive
#   green `verdict-eval` runs, 13-24s each, every one the skip path, while a real
#   run takes ~4 minutes. Eight PRs merged past a check that asserted nothing.
#
# So the skip stays the default and CI opts out of it explicitly. Raising at
# import time fails collection, which surfaces as a red job with this message
# rather than a green one with a skip note nobody reads.
_REQUIRE_LIVE = os.environ.get("VERIFIER_EVAL_REQUIRE_LIVE") == "1"

if _REQUIRE_LIVE and not _KEY:
    raise RuntimeError(
        "VERIFIER_EVAL_REQUIRE_LIVE=1 but MINDSHUB_API_KEY is empty, so this "
        "eval would skip every case and report success. Refusing to pass "
        "without running. Either provide the key or stop requiring live mode. "
        "See ENG-1334."
    )

pytestmark = pytest.mark.skipif(
    not _KEY, reason="MINDSHUB_API_KEY not set — live verdict eval skipped"
)

# Marker the CI guard greps for in the junit report, to tell "the key ran out of
# money" apart from every other reason a case did not run. The string is a
# contract with .github/workflows/verifier-eval.yml; tests/test_verifier_eval_gate.py
# asserts the two match, so drift fails the suite rather than relying on this
# comment. If it ever did drift, it fails SAFE — an unrecognised marker falls to
# the guard's red branch, never to a false green.
_GATEWAY_UNAVAILABLE = "GATEWAY_UNAVAILABLE"

# Pause before the single throttle retry. Long enough for a TPM window to
# clear, short enough that a genuinely dead wallet does not stall the job for
# long (one sleep per case, not per call). Overridable so tests can set it to 0.
_THROTTLE_RETRY_SLEEP_S = float(os.environ.get("VERIFIER_EVAL_THROTTLE_RETRY_SLEEP_S", "20"))

# One structural-tool_choice alias and one narrating alias — the two behaviour
# populations ENG-1081 measured. Overridable for one-off runs against other
# aliases without editing the file.
_FIRST_PARTY_MODEL = os.environ.get("VERIFIER_EVAL_FIRST_PARTY_MODEL", "haiku")
_NARRATING_MODEL = os.environ.get("VERIFIER_EVAL_NARRATING_MODEL", "mindshub_air")
_MODELS = [_FIRST_PARTY_MODEL, _NARRATING_MODEL]

_RUNS = int(os.environ.get("VERIFIER_EVAL_RUNS", "3"))
_STUCK_RUNS = int(os.environ.get("VERIFIER_EVAL_STUCK_RUNS", "6"))


def _client(model: str) -> LLMClient:
    """Build an LLMClient the way a MindsHub install does.

    Goes through ``AntonSettings`` + ``LLMClient.from_settings`` (minds-cloud →
    openai-compatible normalisation, minds_url → openai_base_url derivation)
    rather than hand-constructing a provider, so the eval rides whatever
    provider wiring production rides. Every field that affects the call is
    passed explicitly so a developer's ``~/.cowork/.env`` can't leak in.
    """
    settings = AntonSettings(
        planning_provider="minds-cloud",
        coding_provider="minds-cloud",
        planning_model=model,
        coding_model=model,
        minds_api_key=_KEY,
        minds_url=_BASE,
        openai_api_key=None,
        openai_base_url=None,
        anthropic_api_key=None,
        planning_reasoning_effort=None,
        coding_reasoning_effort=None,
    )
    return LLMClient.from_settings(settings)


async def _verdict(llm: LLMClient, case: Case) -> _VerifierVerdict:
    """One verdict call, replicating session.py's budget-escalation loop:
    first budget, retry once on truncation with the bigger one (ENG-1081).
    Anything else propagates — a hard failure here is a test failure, which is
    the point.

    Except running out of money, which is not a test failure. See
    ``_GATEWAY_UNAVAILABLE`` below.
    """
    system, messages = _build_verify_request(case.history, case.user_message)
    retried_after_throttle = False
    attempt = 0
    while attempt < len(_VERIFIER_TOKEN_BUDGETS):
        budget = _VERIFIER_TOKEN_BUDGETS[attempt]
        try:
            return await llm.generate_object_code(
                _VerifierVerdict, system=system, messages=messages, max_tokens=budget
            )
        except StructuredOutputError as exc:
            if exc.truncated and attempt + 1 < len(_VERIFIER_TOKEN_BUDGETS):
                attempt += 1
                continue
            raise
        except TokenLimitExceeded as exc:
            # TokenLimitExceeded conflates two situations that need OPPOSITE
            # handling, and the type cannot tell them apart:
            #
            #   a TPM velocity throttle  — transient, clears in seconds
            #   a dead wallet / exhausted allowance — permanent for this run
            #
            # llm/openai.py maps ANY 429 carrying a string `detail` to this type
            # (`if exc.status_code == 429 and isinstance(detail, str)`), and that
            # branch sits BEFORE the wallet-code check — and the gateway's own
            # 429 dialect is FastAPI-style `{"detail": ...}`. So a throttle
            # arrives here indistinguishable from an empty wallet.
            #
            # Skipping immediately on the first one was wrong: the next case
            # fires straight into the same throttle window, so the whole matrix
            # skips and the job goes GREEN having executed nothing — the exact
            # outcome ENG-1334 exists to make impossible, rebuilt by its own
            # escape hatch. Caught in review on #328.
            #
            # Time is the discriminator, not the exception type. Retry once
            # after a pause: a throttle clears and the eval runs for real; an
            # empty wallet persists and the honest starved path fires.
            #
            # Deliberately NOT catching ConnectionError, which anton also raises
            # for a 401 invalid key — a misconfiguration somebody must fix, so
            # that has to stay red.
            if not retried_after_throttle:
                retried_after_throttle = True
                await asyncio.sleep(_THROTTLE_RETRY_SLEEP_S)
                continue  # same budget: this is a retry, not an escalation
            pytest.skip(f"{_GATEWAY_UNAVAILABLE}: {exc}")
    raise AssertionError("unreachable: budget loop exhausted without raising")


async def _verdicts(llm: LLMClient, case: Case, n: int) -> list[_VerifierVerdict]:
    # Sequential, not gathered: the gateway rate-limits per key (ENG-878
    # counts full context per request), and a burst of parallel verdict calls
    # 429ing would fail the eval for reasons that say nothing about the rubric.
    return [await _verdict(llm, case) for _ in range(n)]


# ---------------------------------------------------------------------------
# Fixtures — committed transcripts, one per real incident.
#
# History entries use anton's own internal shapes (assistant tool_use blocks,
# user-role tool_result blocks) so `_render_verify_transcript` sees exactly
# what it sees in production.
# ---------------------------------------------------------------------------


def _tool_call(name: str, tool_id: str, **input_) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": input_}],
    }


def _tool_result(tool_id: str, text: str) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [{"type": "text", "text": text}],
            }
        ],
    }


@dataclass
class Case:
    name: str
    user_message: str
    history: list[dict]
    expected: str
    source: str


# --- 1. Tool errored early, model recovered another way → COMPLETE ----------
# ENG-1134's originating incident shape (the SpaceX-vs-Tesla run): the first
# search errored, the model got the data elsewhere, the final answer is
# correct and complete — and the pre-ENG-1134 rubric still forced a redundant
# continuation because *an* errored tool result existed in the transcript.
_RECOVERED = Case(
    name="recovered_tool_error",
    user_message="How much total funding did SpaceX raise compared to Tesla before Tesla's IPO?",
    history=[
        {
            "role": "user",
            "content": "How much total funding did SpaceX raise compared to Tesla before Tesla's IPO?",
        },
        _tool_call("web_search", "toolu_01", query="SpaceX total funding raised"),
        _tool_result(
            "toolu_01",
            "[error] SearchProviderError: upstream returned 502 Bad Gateway; no results",
        ),
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "The search provider errored — let me fetch the funding data directly instead.",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_02",
                    "name": "web_fetch",
                    "input": {"url": "https://en.wikipedia.org/wiki/SpaceX"},
                },
            ],
        },
        _tool_result(
            "toolu_02",
            "SpaceX — funding history: total equity funding raised approximately "
            "$9.8 billion across funding rounds through 2023. Early rounds: 2002 "
            "founding capital $100M (Musk), 2008 NASA CRS contract $1.6B, ...",
        ),
        _tool_call(
            "web_fetch",
            "toolu_03",
            url="https://en.wikipedia.org/wiki/Tesla,_Inc.",
        ),
        _tool_result(
            "toolu_03",
            "Tesla, Inc. — before its June 2010 IPO, Tesla raised approximately "
            "$783 million in private funding: Series A-E rounds ($187M), a $465M "
            "DOE loan (2009), and a $50M Daimler investment. The IPO itself raised "
            "$226 million.",
        ),
        {
            "role": "assistant",
            "content": (
                "Here's the comparison you asked for:\n\n"
                "- **SpaceX**: roughly $9.8B in total equity funding raised to date.\n"
                "- **Tesla (pre-IPO)**: roughly $783M in private funding before its "
                "June 2010 IPO — about $187M across Series A-E, a $465M DOE loan, "
                "and a $50M Daimler investment.\n\n"
                "So SpaceX has raised on the order of 12x what Tesla had raised "
                "before going public."
            ),
        },
    ],
    expected="COMPLETE",
    source="ENG-1134 (recovered-error force-continuation)",
)

# --- 2. Implied success, but the data never arrived → INCOMPLETE ------------
# ENG-1134's preserved safeguard: the rewording must NOT open the door to
# hallucinated success. The assistant's answer depends on data whose tool call
# errored and was never recovered.
_IMPLIED_SUCCESS = Case(
    name="implied_success_data_never_arrived",
    user_message="Pull last month's revenue from the analytics database and tell me how we did.",
    history=[
        {
            "role": "user",
            "content": "Pull last month's revenue from the analytics database and tell me how we did.",
        },
        _tool_call(
            "scratchpad",
            "toolu_11",
            code="df = query_db('SELECT SUM(amount) FROM revenue ...')",
        ),
        _tool_result(
            "toolu_11",
            "[error] OperationalError: connection to server at analytics-db.internal "
            "(10.0.4.12), port 5432 failed: Connection refused",
        ),
        {
            "role": "assistant",
            "content": (
                "Done! Last month's revenue came in at $1.24M, up 15% "
                "month-over-month — a strong result driven mostly by the "
                "enterprise tier. Let me know if you want the per-product split."
            ),
        },
    ],
    expected="INCOMPLETE",
    source="ENG-1134 (hallucinated-success safeguard)",
)

# --- 3. Assistant asks a question it genuinely needs answered → WAITING -----
# ENG-716's incident class: the pre-716 verifier treated a legitimate
# clarifying question as unfinished work and pushed the agent to answer its
# own question.
_WAITING = Case(
    name="genuine_question",
    user_message="Delete the old backup directories from the project.",
    history=[
        {
            "role": "user",
            "content": "Delete the old backup directories from the project.",
        },
        _tool_call("scratchpad", "toolu_21", code="import os; os.listdir('.')"),
        _tool_result(
            "toolu_21",
            "['backups_2024', 'backups_2025', 'backups_old', 'data', 'src', 'README.md']",
        ),
        {
            "role": "assistant",
            "content": (
                "I found three backup directories: `backups_2024/`, "
                "`backups_2025/`, and `backups_old/`. Deleting is irreversible, "
                "and `backups_2025/` contains this year's snapshots — which of "
                "these should I delete? All three, or just the older two?"
            ),
        },
    ],
    expected="WAITING",
    source="ENG-716 (verifier answers the agent's own question)",
)

# --- 4. Environment wall — repeated failed workarounds → STUCK --------------
# ENG-836's Kiranam incident: an OS-level library the task requires, no root
# to install it, and the assistant narrating "let me try another approach"
# after each failed workaround. Under the pre-ENG-836 rubric this measured
# 4 COMPLETE / 4 INCOMPLETE / 1 STUCK — the ~1-in-5 base rate that makes this
# case's N-of-N assertion the before/after measurement for that fix.
#
# The traceback fixture is deliberately >400 characters with the decisive
# line LAST: `_render_tool_result_content` clips tool results at tool_cap=400,
# and pre-ENG-836 that clip was head-only (`content[:cap]`), which discarded
# exactly the line naming the blocker. A short tidy error string here would
# reach the verifier intact and pass against the live bug (see the 2026-08-04
# comment on ENG-1211).
_PYODBC_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/workspace/.anton/scratchpad/cell_7.py", line 12, in <module>\n'
    "    conn = connect_azure_sql()\n"
    "           ^^^^^^^^^^^^^^^^^^^\n"
    '  File "/workspace/.anton/scratchpad/cell_7.py", line 6, in connect_azure_sql\n'
    "    import pyodbc\n"
    '  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load\n'
    '  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked\n'
    '  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked\n'
    '  File "<frozen importlib._bootstrap_external>", line 1287, in exec_module\n'
    '  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed\n'
    '  File "/home/user/.venv/lib/python3.12/site-packages/pyodbc/__init__.py", line 5, in <module>\n'
    "    from .pyodbc import *\n"
    "ImportError: libodbc.so.2: cannot open shared object file: No such file or directory"
)
# The trap this fixture exists to spring (see ENG-1211's 2026-08-04 comment):
# `_render_tool_result_content` clips at tool_cap=400, and pre-ENG-836 that
# clip was head-only — so the blocker's name must sit entirely OUTSIDE the
# first 400 rendered characters, or a rubric judged on a truncation bug would
# still see the cause and the case would pass against the live bug. Asserted
# on the exact rendered form (the "[error] " prefix shifts the window).
assert "libodbc" not in f"[error] {_PYODBC_TRACEBACK}"[:400], (
    "STUCK fixture's decisive line must fall outside a 400-char head clip"
)

_APT_FAILURE = (
    "Reading package lists...\n"
    "E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)\n"
    "E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), "
    "are you root?\n"
    "$ sudo apt-get install unixodbc\n"
    "bash: sudo: command not found"
)

_STUCK = Case(
    name="environment_wall",
    user_message="Connect to our Azure SQL database and build the sales dashboard from the orders table.",
    history=[
        {
            "role": "user",
            "content": "Connect to our Azure SQL database and build the sales dashboard from the orders table.",
        },
        _tool_call("scratchpad", "toolu_31", code="conn = connect_azure_sql()"),
        _tool_result("toolu_31", f"[error] {_PYODBC_TRACEBACK}"),
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "The ODBC system library is missing. Let me try installing it.",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_32",
                    "name": "scratchpad",
                    "input": {"code": "!apt-get install -y unixodbc"},
                },
            ],
        },
        _tool_result("toolu_32", f"[error] {_APT_FAILURE}"),
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "No root access for apt. Trying a pure-Python driver instead.",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_33",
                    "name": "scratchpad",
                    "input": {"code": "import pymssql"},
                },
            ],
        },
        _tool_result(
            "toolu_33",
            "[error] ModuleNotFoundError: No module named 'pymssql'\n"
            "$ pip install pymssql\n"
            "ERROR: Failed building wheel for pymssql\n"
            "src/pymssql/_pymssql.c:611:10: fatal error: sqlfront.h: No such file "
            "or directory — the FreeTDS development headers (freetds-dev) are not "
            "installed and cannot be installed without root.",
        ),
        {
            "role": "assistant",
            "content": (
                "The pymssql build also failed — it needs the FreeTDS system "
                "headers, which I can't install without root either. I'll try "
                "yet another connection approach next."
            ),
        },
    ],
    expected="STUCK",
    source="ENG-836 (Kiranam environment wall, base rate 1-in-5)",
)

# --- 5. Stopped partway through a multi-part request → INCOMPLETE -----------
_PARTIAL = Case(
    name="stopped_partway",
    user_message=(
        "Three things: (1) list the CSV files in data/, (2) compute the total "
        "row count across them, (3) build a bar chart of rows per file."
    ),
    history=[
        {
            "role": "user",
            "content": (
                "Three things: (1) list the CSV files in data/, (2) compute the "
                "total row count across them, (3) build a bar chart of rows per file."
            ),
        },
        _tool_call("scratchpad", "toolu_41", code="import glob; glob.glob('data/*.csv')"),
        _tool_result(
            "toolu_41", "['data/orders.csv', 'data/refunds.csv', 'data/customers.csv']"
        ),
        {
            "role": "assistant",
            "content": (
                "There are three CSV files in `data/`: `orders.csv`, "
                "`refunds.csv`, and `customers.csv`. Next I'll compute the row "
                "counts and build the chart."
            ),
        },
    ],
    expected="INCOMPLETE",
    source="baseline (multi-part request, stopped after step 1)",
)

_CASES = [_RECOVERED, _IMPLIED_SUCCESS, _WAITING, _STUCK, _PARTIAL]


def _runs_for(case: Case) -> int:
    return _STUCK_RUNS if case.expected == "STUCK" else _RUNS


# ---------------------------------------------------------------------------
# The eval
# ---------------------------------------------------------------------------


# If this reds on a PR whose diff can't plausibly change a verdict: re-run the
# failing case against origin/staging with the same key. Still failing -> this
# is live gateway/model drift, i.e. the PRODUCTION verifier now judges this
# fixture differently — a real regression that deserves its own ticket, not an
# eval flake to be quieted (the per-run `status: reason` strings in the failure
# message say what the model now thinks). Passing on staging -> suspect the PR
# after all, or a transient (#307 review: the accepted lever for transients is
# a one-retry wrapper on non-StructuredOutputError failures in `_verdict`,
# never looser assertions).
@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
async def test_verdict(model: str, case: Case):
    llm = _client(model)
    n = _runs_for(case)
    verdicts = await _verdicts(llm, case, n)
    statuses = [v.status for v in verdicts]
    # The verifier's own `reason` strings are the diagnostic for a red run —
    # "why did the rubric flip" — so surface them in the failure message
    # instead of just the statuses.
    detail = "; ".join(f"{v.status}: {v.reason}" for v in verdicts)
    assert statuses == [case.expected] * n, (
        f"{case.name} on {model}: expected {case.expected} x{n} "
        f"(source: {case.source}), got [{detail}]"
    )


async def test_narrating_model_reaches_a_verdict_at_shipped_budgets():
    """ENG-1081 regression guard: a narrating alias must produce *a* verdict
    through the shipped budget-escalation loop — the failure mode was 98.6% of
    mindshub_air verdicts silently returning no tool call at max_tokens=256,
    which the fail-safe upstream turned into silent task death. Verdict
    *quality* on this alias is covered by the matrix above; this asserts only
    that the call survives the model's narration.

    Behaviourally this overlaps the matrix (a truncation escape on the
    recovered fixture would fail `test_verdict` too) — kept as one cheap,
    named call so the ENG-1081 regression class stays traceable in the test
    report even if the matrix's cases or models are later reshaped.
    """
    llm = _client(_NARRATING_MODEL)
    verdict = await _verdict(llm, _RECOVERED)
    assert verdict.status in ("COMPLETE", "WAITING", "INCOMPLETE", "STUCK")
