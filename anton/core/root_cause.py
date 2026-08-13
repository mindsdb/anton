"""Root-cause classification for tool failures (ENG-1492).

Turns ENG-1276's per-failure ``ToolOutcome.reason`` into a **comparable key**,
so "is this the same wall as last time?" becomes a lookup instead of a guess.
This module only measures. Nothing here trips, hands back, or touches the
existing per-tool error streak — the control that consumes it is ENG-1531, and
it is deliberately not built until this has reported real numbers.

Why a key at all
----------------
The ENG-836 runaway ping-ponged ``pyodbc`` -> ``apt`` -> ``.deb`` -> ``pymssql``:
four approaches, four different error texts, **one wall** (no ODBC driver, no
root). Matching normalised error *text* would see four unrelated problems, which
is why the key has two levels — a ``class`` that collapses all four, and an
``identifier`` that keeps them apart when that matters.

The tiers are the safety argument
---------------------------------
Not the thresholds. A breaker keyed on repetition is one bad classification away
from interrupting an agent that is productively fixing its own bug, and the
2026-08-12 harvest showed the heaviest turns are *dominated* by exactly that:
UnicodeEncodeError, NameError, SyntaxError, AttributeError, PyCompileError. So
``self_inflicted`` failures are excluded **structurally** — they are counted, and
they can never become trip-eligible however often they repeat.

ENG-1286's original argument was that self-repair escapes because the error
changes on each attempt. That is probabilistic and fails exactly when an agent
retries the same broken line. This does not.

``transient`` is the same move for ENG-836's probe G (a 429 twice is not a wall),
and it closes a real gap: ENG-673's ``classify_transient`` lives in the LLM
clients and is unreachable from a tool outcome, so tiering by exception type buys
transient exclusion without building tool-level transient typing first.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Reused as a PURE FUNCTION. Deliberately not the ACC subsystem it lives in:
# `_acc_observe` returns early when ANTON_ACC_MODE == "off", and a signal that
# disappears when a memory feature flag flips is convention 9's state-gated trap.
from anton.core.memory.acc import _normalise_error_signature

TIER_SELF = "self_inflicted"
TIER_TRANSIENT = "transient"
TIER_WALL = "external_wall"
TIER_UNCLASSIFIED = "unclassified"

#: Only this tier may ever contribute to a trip. Everything else is measured and
#: ignored — see the module docstring for why that is structural, not a default.
TRIP_ELIGIBLE_TIERS = frozenset({TIER_WALL})

#: The agent's own bugs. It can fix these by writing better code, so repetition
#: is iteration, not a wall — however many times it repeats.
_SELF_INFLICTED = frozenset({
    "NameError", "SyntaxError", "IndentationError", "TabError", "TypeError",
    "AttributeError", "KeyError", "IndexError", "ValueError", "UnboundLocalError",
    "ZeroDivisionError", "PyCompileError", "UnicodeEncodeError",
    "UnicodeDecodeError", "UnicodeError", "AssertionError", "StopIteration",
    "RecursionError", "NotImplementedError",
})

#: Retryable by nature. Two of these is a provider blip, not a blocked task.
_TRANSIENT = frozenset({
    "ConnectionResetError", "ConnectionAbortedError", "BrokenPipeError",
    "TimeoutError", "ReadTimeout", "ReadTimeoutError", "ConnectTimeout",
    "SSLError", "ChunkedEncodingError", "IncompleteRead", "RemoteDisconnected",
})

#: Exception type -> wall class. The identifier is pulled separately below.
_WALL_TYPES = {
    "ModuleNotFoundError": "missing_dependency",
    "ImportError": "missing_dependency",
    "PermissionError": "permission_denied",
    "FileNotFoundError": "missing_file",
    "NotADirectoryError": "missing_file",
    "IsADirectoryError": "missing_file",
    "ConnectionRefusedError": "connection_refused",
    "OperationalError": "db_unavailable",
    "InterfaceError": "db_unavailable",
    "DatabaseError": "db_unavailable",
    "MemoryError": "resource_exhausted",
    "OSError": "resource_exhausted",
}

#: Sentinel reasons handlers emit instead of an exception name. These are the
#: handler's own verdict, so they are the most trustworthy input this has —
#: every one is enumerated from the tree by `test_every_sentinel_reason_is_mapped`,
#: which fails when a new one is added without a decision here.
#:
#: **Ambiguous sentinels are deliberately NOT trip-eligible.** A false wall
#: interrupts a working agent; a missed wall only means the spend ceiling catches
#: it later. The costs are not symmetric, so ambiguity resolves away from the
#: tier that can trip.
_SENTINEL_REASONS = {
    # The agent passed bad or missing arguments — it can fix these itself.
    "scratchpad_empty_code": (TIER_SELF, "empty_code"),
    "scratchpad_missing_name": (TIER_SELF, "missing_name"),
    "missing_name": (TIER_SELF, "missing_argument"),
    "missing_slug": (TIER_SELF, "missing_argument"),
    "missing_description": (TIER_SELF, "missing_argument"),
    "invalid_type": (TIER_SELF, "invalid_argument"),
    "invalid_port": (TIER_SELF, "invalid_argument"),
    "invalid_health_timeout": (TIER_SELF, "invalid_argument"),
    "invalid_datasources": (TIER_SELF, "invalid_argument"),
    # The agent named something that does not exist. Ambiguous — it could be a
    # genuinely absent resource — but the agent chose the identifier and can
    # list the real ones, so it resolves to the non-tripping side.
    "artifact_not_found": (TIER_SELF, "unknown_resource"),
    "unknown_datasource": (TIER_SELF, "unknown_resource"),
    # Genuine walls: the environment is missing something the agent cannot add.
    "package_install_failed": (TIER_WALL, "missing_dependency"),
    "store_unavailable": (TIER_WALL, "service_unavailable"),
    # Could be environment or config and the sentinel does not say which, so it
    # stays out of every trip rung until something distinguishes them.
    "launch_failed": (TIER_UNCLASSIFIED, "unclassified"),
}

# Identifier extraction, per class. Kept narrow on purpose: a wrong identifier
# splits one wall into several and hides it, which is the failure mode that
# matters here — the opposite error merely groups two walls together, which is
# visible in the data rather than silent.
_MODULE_RE = re.compile(r"No module named ['\"]([\w.]+)['\"]")
_IMPORT_NAME_RE = re.compile(r"cannot import name ['\"](\w+)['\"]")
_PATH_RE = re.compile(r"['\"]([^'\"]{1,120})['\"]")
_WINERR_PATH_RE = re.compile(r"\[(?:Errno|WinError) \d+\][^:]*: ['\"]?([^'\"]{1,120})")
_HOSTPORT_RE = re.compile(r"([\w.-]+):(\d{2,5})")
_STATUS_RE = re.compile(r"\b(4\d\d|5\d\d)\b")

#: HTTP statuses that are walls rather than blips, mapped to their class.
_STATUS_WALLS = {"401": "auth_missing", "403": "permission_denied", "404": "missing_file"}
#: …and the ones that are always transient, whatever else the text says.
_STATUS_TRANSIENT = frozenset({"429", "500", "502", "503", "504"})

_EXC_LINE_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning))\b")


@dataclass(frozen=True)
class RootCause:
    """One failure, classified.

    ``key`` is the exact rung (``class:identifier``); ``cls`` alone is the
    coarse rung that collapses the ENG-836 ping-pong. ``trip_eligible`` is the
    whole safety contract in one boolean — a consumer must never re-derive it
    from the tier, or the exclusion stops being structural.
    """

    tier: str
    cls: str
    identifier: str
    #: False when nothing usable was available and the key came from normalised
    #: result text. Counted separately so the coverage share is measurable
    #: rather than assumed (convention 9's writer-inventory, as a live metric).
    from_reason: bool = True

    @property
    def key(self) -> str:
        return f"{self.cls}:{self.identifier}" if self.identifier else self.cls

    @property
    def trip_eligible(self) -> bool:
        return self.tier in TRIP_ELIGIBLE_TIERS


def _exception_name(reason: str) -> str:
    """The exception type at the head of a reason, or ''.

    `reason` is usually a traceback's LAST line (`tool_handlers.py`), which is
    where the type lives, but a bare `type(exc).__name__` also arrives here from
    the dispatcher's own except-clause.
    """
    reason = (reason or "").strip()
    if not reason:
        return ""
    head = reason.split(":", 1)[0].strip()
    if head and head.replace("_", "").isalnum() and head[0].isupper():
        return head
    m = _EXC_LINE_RE.search(reason)
    return m.group(1) if m else ""


def _identifier_for(cls: str, text: str) -> str:
    """The salient noun for a wall class — the module, path, host or service.

    Normalised through `_normalise_error_signature` when nothing more specific
    is found, so two spellings of the same failure still land together.
    """
    if cls == "missing_dependency":
        m = _MODULE_RE.search(text) or _IMPORT_NAME_RE.search(text)
        if m:
            # Top-level package only: `pandas.io.parsers` and `pandas` are one
            # missing dependency, and the breaker must see them as such.
            return m.group(1).split(".")[0]
    if cls in ("missing_file", "permission_denied"):
        m = _WINERR_PATH_RE.search(text) or _PATH_RE.search(text)
        if m:
            return m.group(1)
    if cls in ("connection_refused", "db_unavailable", "auth_missing"):
        m = _HOSTPORT_RE.search(text)
        if m:
            return f"{m.group(1)}:{m.group(2)}"
    return _normalise_error_signature(text)[:80]


def classify(reason: str, result_text: str = "") -> RootCause:
    """Classify one tool failure.

    `reason` is ENG-1276's `ToolOutcome.reason` — the handler's own machine
    comparable cause. `result_text` is the fallback for handlers that set none;
    keying on it is weaker (it is prose the model can influence) so it is
    recorded as `from_reason=False` and lands in `unclassified`, which is not
    trip-eligible.
    """
    reason = (reason or "").strip()
    if not reason:
        sig = _normalise_error_signature(result_text or "")
        return RootCause(TIER_UNCLASSIFIED, "unclassified", sig[:80], from_reason=False)

    sentinel = _SENTINEL_REASONS.get(reason)
    if sentinel:
        tier, cls = sentinel
        return RootCause(tier, cls, "")

    exc = _exception_name(reason)

    # Status codes are checked BEFORE the exception type: an HTTPError carrying
    # a 429 is transient no matter what its class name suggests, and one
    # carrying a 403 is a wall. The type alone cannot tell those apart.
    status = _STATUS_RE.search(reason)
    if status:
        code = status.group(1)
        if code in _STATUS_TRANSIENT:
            return RootCause(TIER_TRANSIENT, f"http_{code}", "")
        wall_cls = _STATUS_WALLS.get(code)
        if wall_cls:
            return RootCause(TIER_WALL, wall_cls, _identifier_for(wall_cls, reason))

    if exc in _SELF_INFLICTED:
        return RootCause(TIER_SELF, exc, "")
    if exc in _TRANSIENT:
        return RootCause(TIER_TRANSIENT, exc, "")

    wall_cls = _WALL_TYPES.get(exc)
    if wall_cls:
        # OSError is the catch-all base class — only treat it as a resource wall
        # when the text actually says so, else it is too coarse to key on.
        if exc == "OSError" and not re.search(
            r"No space|Disk quota|Too many open files|Cannot allocate", reason, re.I
        ):
            return RootCause(TIER_UNCLASSIFIED, "unclassified",
                             _normalise_error_signature(reason)[:80])
        return RootCause(TIER_WALL, wall_cls, _identifier_for(wall_cls, reason))

    # Timeouts and kills the runtime words rather than raises.
    low = reason.lower()
    if "timed out" in low or "timeout" in low or "inactivity" in low or "liveness" in low:
        return RootCause(TIER_TRANSIENT, "timeout", "")
    if "permission denied" in low:
        return RootCause(TIER_WALL, "permission_denied",
                         _identifier_for("permission_denied", reason))
    if "connection refused" in low:
        return RootCause(TIER_WALL, "connection_refused",
                         _identifier_for("connection_refused", reason))

    return RootCause(TIER_UNCLASSIFIED, "unclassified",
                     _normalise_error_signature(reason)[:80])


@dataclass
class RootCauseLedger:
    """Session-scoped tally of classified failures.

    **Occurrences, never a streak.** A success must not reset anything — that is
    ENG-1276's lesson one level up. The existing per-tool streak reset on
    success, and interleaved false successes (a 404 HTML page that parsed as a
    valid archive) are precisely why it never counted to five through the
    ENG-836 runaway.

    **Lifetime is one ChatSession, and that means different things per host —
    check before reading the numbers as "per conversation".**

    | host | ChatSession built | ledger effectively |
    |------|-------------------|--------------------|
    | CLI (`chat.py`) | once, then loops `turn_stream` | spans the conversation |
    | Cowork (`anton_harness`) | inside `stream_response()`, i.e. per HTTP turn | **per turn** |

    So on the primary product this resets every turn. That was NOT the original
    intent — the design note said session-scoped, on the reasoning that
    ENG-1531 fires once per root cause per session — and it is recorded here
    because the difference is invisible from this file.

    What it still measures correctly: a wall hit repeatedly **within one turn**,
    which is the ENG-836 shape (all four pyodbc/apt/.deb/pymssql attempts were
    in a single turn of 29 calls). What it misses: a wall that persists across
    turns because the user said "try again", where each turn starts from zero.

    Consequence for ENG-1531: its "at most once per root cause per session"
    cannot be built on this object alone — it needs cowork-server to hold the
    ledger per conversation, or to accept per-turn scope deliberately. Sizing
    thresholds from this data is still valid; claiming session coverage is not.
    """

    exact: Counter = field(default_factory=Counter)
    classes: Counter = field(default_factory=Counter)
    tiers: Counter = field(default_factory=Counter)
    failures: int = 0
    with_reason: int = 0

    def add(self, rc: RootCause) -> None:
        self.failures += 1
        self.tiers[rc.tier] += 1
        if rc.from_reason:
            self.with_reason += 1
        if not rc.trip_eligible:
            # Counted in `tiers` for the coverage picture, but kept out of the
            # rungs a trip could ever read. This is where "structurally cannot
            # trip" actually happens.
            return
        self.exact[rc.key] += 1
        self.classes[rc.cls] += 1

    @property
    def max_exact(self) -> int:
        return max(self.exact.values(), default=0)

    @property
    def max_class(self) -> int:
        return max(self.classes.values(), default=0)

    @property
    def top_class(self) -> str:
        return self.classes.most_common(1)[0][0] if self.classes else ""

    @property
    def reason_coverage(self) -> float:
        """Share of failures that carried a usable `ToolOutcome.reason`.

        The live form of the writer inventory: a key computed over a third of
        failures would set thresholds wrong in the safe-looking direction, and
        this is the number that says whether that is happening.
        """
        return (self.with_reason / self.failures) if self.failures else 0.0

    def event_fields(self) -> dict:
        """Flat properties for the turn's analytics event and log line."""
        return {
            "rc_failures": self.failures,
            "rc_distinct": len(self.exact),
            "rc_max_exact": self.max_exact,
            "rc_max_class": self.max_class,
            "rc_top_class": self.top_class,
            "rc_reason_coverage": round(self.reason_coverage, 3),
            "rc_self_inflicted": self.tiers.get(TIER_SELF, 0),
            "rc_transient": self.tiers.get(TIER_TRANSIENT, 0),
            "rc_wall": self.tiers.get(TIER_WALL, 0),
            "rc_unclassified": self.tiers.get(TIER_UNCLASSIFIED, 0),
        }
