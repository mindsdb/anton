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

#: Wall classes that count on the EXACT rung only, never the coarse one.
#:
#: `missing_file` is genuinely ambiguous — a wrong path the agent invented is
#: self-inflicted, a missing binary is a wall — and the two are indistinguishable
#: from the error alone. The exact rung tells them apart by construction: the
#: same path three times is a wall, three different paths is an agent exploring.
#: Counting it on the coarse rung merges those, and worse, drowns real walls:
#: measured, five file probes plus the four-attempt ENG-836 dependency wall in
#: one turn emitted `max_class=5, top_class="missing_file"` — the wall this
#: whole ticket exists to see, masked by exploration.
#:
#: The cost, stated because ENG-1492 already accepted it: a genuine wall spread
#: over several paths (one missing ODBC install surfacing as three absent files)
#: scores 1 on both rungs and is invisible.
_EXACT_ONLY_CLASSES = frozenset({"missing_file"})

#: Classes `classify` returns as literals rather than through one of the tables
#: below — the runtime WORDS these failures instead of raising a mapped type,
#: so there is no exception name to look up. Named rather than inlined so
#: `ALL_CLASSES` is a real derivation: a consumer enumerating the vocabulary
#: from the tables alone silently misses them.
#:
#: `timeout` is the one that bites. It is reachable in production on every
#: scratchpad cell timeout (`backends/local.py` builds "Cell timed out after
#: {N}s total", which arrives here as the tool's `reason`) and appears in no
#: table, so a closed-set guard built from the tables reports a legitimate
#: value as a novel class. `permission_denied` and `connection_refused` were
#: covered only by coincidence — they happen to also be `_WALL_TYPES` values.
CLS_UNCLASSIFIED = "unclassified"
CLS_TIMEOUT = "timeout"
CLS_PERMISSION_DENIED = "permission_denied"
CLS_CONNECTION_REFUSED = "connection_refused"

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
    # A refused package spec (flag/URL/path-shaped entry) is the agent's own
    # bad argument, not an environment wall (ENG-1635).
    "package_install_rejected": (TIER_SELF, "invalid_argument"),
    # `read_image`, ENG-2248: both are the agent's own file choice, and both
    # messages tell it what to do instead (use a real image / resize).
    "not_an_image": (TIER_SELF, "invalid_argument"),
    "image_too_large": (TIER_SELF, "invalid_argument"),
    # The agent named something that does not exist. Ambiguous — it could be a
    # genuinely absent resource — but the agent chose the identifier and can
    # list the real ones, so it resolves to the non-tripping side.
    "artifact_not_found": (TIER_SELF, "unknown_resource"),
    "unknown_datasource": (TIER_SELF, "unknown_resource"),
    # Same shape for a path: `read_image` reports the file the agent named as
    # absent. Genuinely absent files exist, but the agent supplied the path and
    # can list the directory, so it resolves to the non-tripping side.
    "missing_file": (TIER_SELF, "unknown_resource"),
    # Genuine walls: the environment is missing something the agent cannot add.
    "package_install_failed": (TIER_WALL, "missing_dependency"),
    "store_unavailable": (TIER_WALL, "service_unavailable"),
    # Could be environment or config and the sentinel does not say which, so it
    # stays out of every trip rung until something distinguishes them.
    "launch_failed": (TIER_UNCLASSIFIED, "unclassified"),
    # `read_image`'s catch-all read failure wraps a bare `except Exception`, so
    # one sentinel covers a permissions wall, a corrupt file the agent itself
    # wrote, and a decode bug. Nothing here distinguishes them, so it stays out
    # of every trip rung (ENG-2248).
    "read_failed": (TIER_UNCLASSIFIED, "unclassified"),
    # A bare `except Exception` around a PIL round-trip: a missing Pillow is a
    # wall, a corrupt BMP is not, and the sentinel cannot tell them apart.
    "bmp_convert_failed": (TIER_UNCLASSIFIED, "unclassified"),
}

# Identifier extraction, per class. Kept narrow on purpose: a wrong identifier
# splits one wall into several and hides it, which is the failure mode that
# matters here — the opposite error merely groups two walls together, which is
# visible in the data rather than silent.
_MODULE_RE = re.compile(r"No module named ['\"]([\w.]+)['\"]")
# Captures the MODULE, not the symbol. Keying on the symbol split one broken
# package into as many keys as symbols the agent tried to import from it —
# three failures on `pandas` scoring max_exact=1, distinct=3 — and made
# `cannot import name 'read_excel' from 'pandas'` a different wall from
# `No module named 'pandas'`. The `.split(".")[0]` at the use site is a no-op on
# a symbol, which is the evidence the module was always intended.
_IMPORT_NAME_RE = re.compile(
    r"cannot import name ['\"]\w+['\"] from "
    r"(?:partially initialized module )?['\"]([\w.]+)['\"]"
)
_PATH_RE = re.compile(r"['\"]([^'\"]{1,120})['\"]")
_WINERR_PATH_RE = re.compile(r"\[(?:Errno|WinError) \d+\][^:]*: ['\"]?([^'\"]{1,120})")
_HOSTPORT_RE = re.compile(r"([\w.-]+):(\d{2,5})")
_STATUS_RE = re.compile(r"\b(4\d\d|5\d\d)\b")

#: HTTP statuses that are walls rather than blips, mapped to their class.
_STATUS_WALLS = {"401": "auth_missing", "403": "permission_denied", "404": "missing_file"}
#: …and the ones that are always transient, whatever else the text says.
_STATUS_TRANSIENT = frozenset({"429", "500", "502", "503", "504"})

_EXC_LINE_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning))\b")

#: Hard cap on what any regex here is allowed to see.
#:
#: `_HOSTPORT_RE` is O(n^2) on input containing no colon — `[\w.-]+` matches
#: greedily from every start position and then fails. Measured: **1,051 ms** on
#: 20,000 characters. Nothing reaches it that long today (the scratchpad handler
#: caps its reason at 160 chars and every other producer emits a short sentinel
#: or an exception name), but that guarantee lives in another file and would be
#: silently lost the first time a handler passes a long reason. Capping here
#: makes it local, and covers every pattern in this module rather than the one
#: that happens to be quadratic.
#:
#: 500 is well past the longest real reason: a traceback's last line arrives
#: pre-truncated to 160.
_MAX_REASON_CHARS = 500


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
    # Bounded before any regex sees it — see `_MAX_REASON_CHARS`.
    reason = (reason or "").strip()[:_MAX_REASON_CHARS]
    if not reason:
        sig = _normalise_error_signature((result_text or "")[:_MAX_REASON_CHARS])
        return RootCause(TIER_UNCLASSIFIED, CLS_UNCLASSIFIED, sig[:80], from_reason=False)

    sentinel = _SENTINEL_REASONS.get(reason)
    if sentinel:
        tier, cls = sentinel
        return RootCause(tier, cls, "")

    exc = _exception_name(reason)

    # The explicit type tables win over the status heuristic, and the ORDER here
    # is the safety property — not a style choice.
    #
    # The status check used to run first, on the reasoning that an HTTPError
    # carrying 429 is transient whatever its class name says. True, but it
    # searched `\b(4\d\d|5\d\d)\b` against the WHOLE reason with no HTTP-shape
    # precondition, so any self-inflicted exception whose message happened to
    # contain a bare 401/403/404 was promoted to `external_wall`, trip-eligible.
    # Measured on real raised exceptions:
    #
    #     {200:..,404:..}[403]        -> KeyError: 403          -> wall, trip=True
    #     assert 404 == 200           -> AssertionError         -> wall, trip=True
    #     SyntaxError … (line 404)    -> wall, trip=True
    #     …the same error at line 405 -> self_inflicted, trip=False
    #
    # That is exactly the "agent retries the same broken line" case this tier
    # exists to exclude, so the guarantee in the module docstring was false for
    # 8 specific integers. `HTTPError` is in neither table, so it still falls
    # through to the status branch — which keeps the legitimate case working.
    if exc in _SELF_INFLICTED:
        return RootCause(TIER_SELF, exc, "")
    if exc in _TRANSIENT:
        return RootCause(TIER_TRANSIENT, exc, "")

    # RESIDUAL, by construction — the guarantee above is scoped to types IN the
    # table, and this branch is what a non-enumerated type falls through to. An
    # agent bug raising a type nobody enumerated (`RuntimeError`, `LookupError`,
    # a bare `Exception`) whose message happens to carry 401/403/404 still mints
    # a trip-eligible `external_wall` — `RuntimeError: record 404 not found` is
    # the realistic shape.
    #
    # Left as-is deliberately: widening `_SELF_INFLICTED` to catch it would have
    # to include `Exception`, which would swallow every genuine wall raised as a
    # generic type. The direction is the safe one — it INFLATES wall counts, so
    # it is visible as noise in the ENG-1492 distribution rather than hiding a
    # wall — and this is measurement-only until ENG-1531 reads any of it.
    #
    # SECOND, OPPOSITE INTERACTION — this branch also runs BEFORE the
    # `_WALL_TYPES` lookup below, so an ENUMERATED wall type whose path or host
    # happens to contain a bare 3-digit 4xx/5xx token is demoted to transient
    # and never reaches `root_cause_wall` or any trip rung. Measured:
    #
    #     FileNotFoundError … '/data/500/report.csv'  -> transient http_500
    #     ConnectionRefusedError … db.internal:503    -> transient http_503
    #     …'/data/report.csv'                         -> external_wall  (control)
    #     …db.internal:5432                           -> external_wall  (control,
    #                                                    4 digits, `\b` blocks it)
    #
    # So the status branch errs BOTH ways: it inflates walls from non-enumerated
    # types (above) and deflates walls from enumerated ones (here). Both are
    # tolerable while nothing consumes the tier, and the motivating dependency
    # walls carry no such tokens (`pyodbc`/`pymssql` are unaffected).
    #
    # For ENG-1531: do not trip on `_STATUS_WALLS` classes without either an
    # HTTP-shape precondition on the reason or a check that the exception type
    # was enumerated — that single precondition fixes both directions at once.
    # And when sizing thresholds from the ENG-1492 distribution, know that walls
    # with numeric ports or paths read LOW. Frequency here is UNMEASURED; the
    # combinatorial floor is 3/900 of uniformly distributed 3-digit tokens.
    status = _STATUS_RE.search(reason)
    if status:
        code = status.group(1)
        if code in _STATUS_TRANSIENT:
            return RootCause(TIER_TRANSIENT, f"http_{code}", "")
        wall_cls = _STATUS_WALLS.get(code)
        if wall_cls:
            return RootCause(TIER_WALL, wall_cls, _identifier_for(wall_cls, reason))

    wall_cls = _WALL_TYPES.get(exc)
    if wall_cls:
        # OSError is the catch-all base class — only treat it as a resource wall
        # when the text actually says so, else it is too coarse to key on.
        if exc == "OSError" and not re.search(
            r"No space|Disk quota|Too many open files|Cannot allocate", reason, re.I
        ):
            return RootCause(TIER_UNCLASSIFIED, CLS_UNCLASSIFIED,
                             _normalise_error_signature(reason)[:80])
        return RootCause(TIER_WALL, wall_cls, _identifier_for(wall_cls, reason))

    # Timeouts and kills the runtime words rather than raises.
    low = reason.lower()
    if "timed out" in low or "timeout" in low or "inactivity" in low or "liveness" in low:
        return RootCause(TIER_TRANSIENT, CLS_TIMEOUT, "")
    if "permission denied" in low:
        return RootCause(TIER_WALL, CLS_PERMISSION_DENIED,
                         _identifier_for(CLS_PERMISSION_DENIED, reason))
    if "connection refused" in low:
        return RootCause(TIER_WALL, CLS_CONNECTION_REFUSED,
                         _identifier_for(CLS_CONNECTION_REFUSED, reason))

    return RootCause(TIER_UNCLASSIFIED, CLS_UNCLASSIFIED,
                     _normalise_error_signature(reason)[:80])


#: Every value `classify` can put in `RootCause.cls`, and every tier. THE
#: authoritative enumeration — a consumer must read these rather than
#: reassembling the tables, which is how `timeout` went missing (ENG-2247).
#:
#: Adding a class means adding it here. That keeps the SET honest; it does not
#: by itself keep a closed-set guard honest, because such a guard only catches
#: an unregistered value if some input actually reaches the branch returning
#: it. Two of ENG-2247's own adversarial inputs missed their branches (an
#: `OSError: ` prefix sent them down `_WALL_TYPES` instead) and the guard
#: still passed. So a new literal needs BOTH an entry here and an input that
#: reaches it.
ALL_CLASSES: frozenset[str] = frozenset(
    set(_SELF_INFLICTED)
    | set(_TRANSIENT)
    | set(_WALL_TYPES.values())
    | {cls for _, cls in _SENTINEL_REASONS.values()}
    | set(_STATUS_WALLS.values())
    | {f"http_{code}" for code in _STATUS_TRANSIENT}
    | {CLS_UNCLASSIFIED, CLS_TIMEOUT, CLS_PERMISSION_DENIED, CLS_CONNECTION_REFUSED}
)

ALL_TIERS: frozenset[str] = frozenset(
    {TIER_SELF, TIER_TRANSIENT, TIER_WALL, TIER_UNCLASSIFIED}
)


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
    #: Times the recorder's guard swallowed an exception instead of booking a
    #: failure. Emitted so a BROKEN instrument is distinguishable from a QUIET
    #: one — without it the two are byte-identical:
    #:
    #:     3 wall failures, classifier raising -> failures=0 … wall=0
    #:     3 successes, genuinely clean turn   -> failures=0 … wall=0
    #:
    #: That ambiguity matters because "the wall-repeat population is too small
    #: to justify the control" is one of ENG-1492's own sanctioned conclusions,
    #: so a silent instrument failure reads as a legitimate answer and would
    #: cancel ENG-1531 on the strength of a bug. Same defect `reason_coverage`
    #: was built to fix, one level down: a metric must not exclude its own
    #: failures from its own denominator.
    #:
    #: Deliberately NOT counted in `failures` — that would corrupt the very
    #: distribution this exists to protect. It is a separate signal.
    classify_errors: int = 0

    def note_error(self) -> None:
        """Record that the guard caught something. Must never raise."""
        self.classify_errors += 1

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
        if rc.cls not in _EXACT_ONLY_CLASSES:
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

        **Read this together with `classify_errors`.** A classification that
        RAISED reaches neither counter, so it leaves this ratio entirely — on
        partial failure the number reads high:

            3 classified (2 with reason) + 2 raised
              -> reported 0.667, true 2/5 = 0.4

        The systematic case is safe rather than flattering (if every
        classification raises, `failures` is 0 and this reads 0.0, not 1.0), and
        `classify_errors` is non-zero in both cases — so a dashboard should
        treat any `classify_errors > 0` as "this ratio is an over-estimate"
        rather than trusting it at face value.
        """
        return (self.with_reason / self.failures) if self.failures else 0.0

    def event_fields(self) -> dict:
        """Flat properties for the turn's analytics event and log line.

        **Emitted per turn; the VALUES are cumulative for the ledger's lifetime**
        (see the class docstring — one turn under Cowork, the whole conversation
        under the CLI). The names read as per-turn and are not.

        This matters for ENG-1492's actual deliverable, the distribution that
        sets ENG-1286/ENG-1531's thresholds: on a host where the ledger spans
        turns, a 10-turn session emits 10 events whose counts only climb, so a
        naive per-EVENT percentile is dominated by long sessions and reads high.
        Take the LAST event per session, or compute per-session maxima — never
        a percentile over the raw event stream.
        """
        return {
            "root_cause_failures": self.failures,
            "root_cause_classify_errors": self.classify_errors,
            "root_cause_distinct": len(self.exact),
            "root_cause_max_exact": self.max_exact,
            "root_cause_max_class": self.max_class,
            "root_cause_top_class": self.top_class,
            "root_cause_reason_coverage": round(self.reason_coverage, 3),
            "root_cause_self_inflicted": self.tiers.get(TIER_SELF, 0),
            "root_cause_transient": self.tiers.get(TIER_TRANSIENT, 0),
            "root_cause_wall": self.tiers.get(TIER_WALL, 0),
            "root_cause_unclassified": self.tiers.get(TIER_UNCLASSIFIED, 0),
        }
