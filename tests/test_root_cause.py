"""Root-cause classification and the session ledger (ENG-1492).

The safety property under test is not "does it classify well" — it is that
`self_inflicted` and `transient` failures can NEVER become trip-eligible, no
matter how often they repeat. That is the whole reason the tiers exist, and it
is what makes ENG-836's probes G and H structural rather than hoped-for.
"""

from __future__ import annotations

import pytest

from anton.core.root_cause import (
    TIER_SELF,
    TIER_TRANSIENT,
    TIER_UNCLASSIFIED,
    TIER_WALL,
    RootCauseLedger,
    classify,
)


# ── The motivating case ────────────────────────────────────────────────────


# Verbatim-shaped reasons from the ENG-836 runaway: four approaches at one wall
# (no ODBC driver, no root). `tool_handlers.py` sets `reason` to the traceback's
# LAST line, which is what these are.
KIRANAM_ATTEMPTS = [
    "ModuleNotFoundError: No module named 'pyodbc'",
    "package_install_failed",
    "ModuleNotFoundError: No module named 'pyodbc.drivers'",
    "ModuleNotFoundError: No module named 'pymssql'",
]


def test_the_kiranam_ping_pong_collapses_to_one_class():
    """Four identifiers, one class — the case an exact match cannot catch.

    This is why the key has two rungs. Normalised error TEXT would see four
    unrelated problems here and never fire, which is exactly what happened.
    """
    causes = [classify(r) for r in KIRANAM_ATTEMPTS]

    assert {c.cls for c in causes} == {"missing_dependency"}
    assert all(c.trip_eligible for c in causes)

    ledger = RootCauseLedger()
    for c in causes:
        ledger.add(c)

    # The coarse rung sees one wall hit four times…
    assert ledger.max_class == 4
    # …while the exact rung sees the approach-swapping and stays low, which is
    # why a breaker keyed only on the exact rung would have missed this.
    assert ledger.max_exact < 4
    assert ledger.top_class == "missing_dependency"


def test_a_submodule_counts_as_its_top_level_package():
    """`pandas.io.parsers` and `pandas` are one missing dependency."""
    a = classify("ModuleNotFoundError: No module named 'pandas.io.parsers'")
    b = classify("ModuleNotFoundError: No module named 'pandas'")
    assert a.key == b.key == "missing_dependency:pandas"


# ── The safety property ────────────────────────────────────────────────────


SELF_INFLICTED_REASONS = [
    "NameError: name 'wb' is not defined",
    "SyntaxError: unterminated string literal (detected at line 250)",
    "IndentationError: unexpected unindent (backend.py, line 822)",
    "AttributeError: 'LLMResponse' object has no attribute 'text'",
    "UnicodeEncodeError: 'utf-8' codec can't encode character",
    "PyCompileError: Sorry: IndentationError: unexpected unindent",
    "TypeError: unsupported operand type(s)",
    "KeyError: 'total'",
]


@pytest.mark.parametrize("reason", SELF_INFLICTED_REASONS)
def test_self_inflicted_failures_can_never_trip(reason):
    """Probe H, encoded structurally.

    These are the types that DOMINATED the heaviest turns in the 2026-08-12
    harvest — UnicodeEncodeError 11, AttributeError 8, NameError 8,
    SyntaxError 8, PyCompileError 5. An agent iterating on its own bug must not
    be interrupted, and this must not depend on the error happening to change
    between attempts.
    """
    rc = classify(reason)
    assert rc.tier == TIER_SELF
    assert not rc.trip_eligible


def test_a_self_inflicted_failure_repeated_forever_still_cannot_trip():
    """The property that matters: repetition does not promote a tier."""
    ledger = RootCauseLedger()
    for _ in range(100):
        ledger.add(classify("NameError: name 'wb' is not defined"))

    assert ledger.failures == 100
    assert ledger.tiers[TIER_SELF] == 100
    # Nothing reached a rung a trip could read.
    assert ledger.max_exact == 0
    assert ledger.max_class == 0
    assert ledger.top_class == ""


TRANSIENT_REASONS = [
    "HTTPError: 429 Too Many Requests",
    "HTTPError: 503 Service Unavailable",
    "ConnectionResetError: [Errno 54] Connection reset by peer",
    "ReadTimeout: HTTPSConnectionPool read timed out",
    "Cell timed out after 120s",
    "Killed by liveness watchdog after inactivity",
]


@pytest.mark.parametrize("reason", TRANSIENT_REASONS)
def test_transients_can_never_trip(reason):
    """Probe G: a 429 with a Retry-After twice is not a wall."""
    rc = classify(reason)
    assert rc.tier == TIER_TRANSIENT
    assert not rc.trip_eligible


def test_a_status_code_beats_the_exception_name():
    """An HTTPError is transient or a wall depending on its STATUS, not its type.

    Checked before the type table on purpose — the class name alone cannot tell
    a rate limit from a permission denial.
    """
    assert classify("HTTPError: 429 Too Many Requests").tier == TIER_TRANSIENT
    assert classify("HTTPError: 403 Forbidden").tier == TIER_WALL
    assert classify("HTTPError: 403 Forbidden").cls == "permission_denied"
    assert classify("HTTPError: 401 Unauthorized").cls == "auth_missing"


# ── Walls ──────────────────────────────────────────────────────────────────


def test_wall_classes_and_their_identifiers():
    cases = [
        ("PermissionError: [Errno 13] Permission denied: '/etc/hosts'",
         "permission_denied", "/etc/hosts"),
        ("ConnectionRefusedError: [Errno 61] Connection refused to db.internal:5432",
         "connection_refused", "db.internal:5432"),
        ("OperationalError: could not connect to server postgres.local:5432",
         "db_unavailable", "postgres.local:5432"),
    ]
    for reason, cls, ident in cases:
        rc = classify(reason)
        assert rc.tier == TIER_WALL, reason
        assert rc.cls == cls, reason
        assert rc.identifier == ident, reason


def test_missing_file_keeps_its_path_so_exploration_does_not_look_like_a_wall():
    """The ambiguous class: a wrong path the agent wrote vs a missing binary.

    Three DIFFERENT paths is an agent exploring and must stay spread across the
    exact rung; the same path three times is a wall. (ENG-1531 will additionally
    refuse to count this class on the coarse rung for the same reason.)
    """
    ledger = RootCauseLedger()
    for p in ("/tmp/a.csv", "/tmp/b.csv", "/tmp/c.csv"):
        ledger.add(classify(f"FileNotFoundError: [Errno 2] No such file or directory: '{p}'"))
    assert ledger.max_exact == 1

    ledger2 = RootCauseLedger()
    for _ in range(3):
        ledger2.add(classify(
            "FileNotFoundError: [Errno 2] No such file or directory: '/usr/bin/odbcinst'"))
    assert ledger2.max_exact == 3


def test_a_bare_oserror_is_not_treated_as_a_wall_without_evidence():
    """OSError is the catch-all base class — too coarse to key on by itself."""
    assert classify("OSError: [Errno 5] Input/output error").tier == TIER_UNCLASSIFIED
    assert classify("OSError: [Errno 28] No space left on device").tier == TIER_WALL


# ── Coverage, the live writer inventory ────────────────────────────────────


def test_a_failure_with_no_reason_is_counted_but_never_trip_eligible():
    """Unmigrated handlers must not silently create walls.

    Falling back to result TEXT is weaker evidence — it is prose the model can
    influence, which is the ENG-1276 defect one level up — so it lands in
    `unclassified` and is excluded from every trip rung.
    """
    rc = classify("", result_text="Tool 'web_fetch' failed: something went wrong")
    assert rc.tier == TIER_UNCLASSIFIED
    assert not rc.trip_eligible
    assert not rc.from_reason


def test_reason_coverage_reports_the_share_with_a_usable_reason():
    ledger = RootCauseLedger()
    ledger.add(classify("ModuleNotFoundError: No module named 'pyodbc'"))
    ledger.add(classify("NameError: name 'x' is not defined"))
    ledger.add(classify("", result_text="opaque failure"))
    ledger.add(classify("", result_text="another opaque failure"))

    assert ledger.failures == 4
    assert ledger.reason_coverage == 0.5
    assert ledger.event_fields()["rc_reason_coverage"] == 0.5


def test_event_fields_are_flat_and_complete():
    """The event carries per-tier counts, not just the trip-eligible rungs.

    Without the tier split the distribution cannot answer the question this
    ticket exists for — whether a wall-repeat population large enough to justify
    a breaker exists at all.
    """
    ledger = RootCauseLedger()
    for r in KIRANAM_ATTEMPTS:
        ledger.add(classify(r))
    ledger.add(classify("NameError: name 'x' is not defined"))
    ledger.add(classify("HTTPError: 429 Too Many Requests"))

    f = ledger.event_fields()
    assert f["rc_failures"] == 6
    assert f["rc_wall"] == 4
    assert f["rc_self_inflicted"] == 1
    assert f["rc_transient"] == 1
    assert f["rc_max_class"] == 4
    assert f["rc_top_class"] == "missing_dependency"
    assert all(not isinstance(v, (dict, list)) for v in f.values())


def test_the_ledger_never_resets_on_success():
    """There is no reset API at all — the absence IS the design.

    The existing per-tool streak resets on success, and interleaved false
    successes are why it stayed asleep through ENG-836. This counts occurrences.
    """
    assert not any(
        hasattr(RootCauseLedger(), name) for name in ("reset", "clear", "on_success")
    )


# ── The writer inventory, kept honest by the suite ─────────────────────────


def test_every_sentinel_reason_is_mapped():
    """Every `reason="..."` literal in the tree has a deliberate tier.

    Convention 9's writer inventory, as a test rather than a one-off audit. A
    new handler sentinel that nobody classifies would land in `unclassified` and
    quietly depress the wall counts — a wrong answer in the safe-looking
    direction, which is the failure mode this whole ticket exists to avoid.

    If this fails: add the new reason to `_SENTINEL_REASONS` with a tier, and
    resolve ambiguity AWAY from `external_wall` (a false wall interrupts a
    working agent; a missed one is caught later by the spend ceiling).
    """
    import pathlib
    import re

    from anton.core.root_cause import _SENTINEL_REASONS

    root = pathlib.Path(__file__).resolve().parents[1] / "anton"
    found: set[str] = set()
    for path in root.rglob("*.py"):
        if path.name == "root_cause.py":
            continue
        found |= set(re.findall(r'reason="([a-z_]+)"', path.read_text()))

    unmapped = found - set(_SENTINEL_REASONS)
    assert not unmapped, (
        f"unclassified handler sentinels: {sorted(unmapped)} — add them to "
        "_SENTINEL_REASONS with a deliberate tier"
    )


def test_ambiguous_sentinels_resolve_away_from_tripping():
    """The asymmetry that decides every judgement call in the table above."""
    from anton.core.root_cause import classify

    for reason in ("artifact_not_found", "unknown_datasource", "launch_failed",
                   "missing_name", "invalid_port"):
        assert not classify(reason).trip_eligible, reason
    # …while the unambiguous walls stay trip-eligible.
    for reason in ("package_install_failed", "store_unavailable"):
        assert classify(reason).trip_eligible, reason
