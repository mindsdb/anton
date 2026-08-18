"""STATE API errors — identical across both drivers."""


class StateError(Exception):
    """Base class for all STATE errors."""


class StateValidationError(StateError):
    """Item/key failed validation (empty key, type, size, TTL format)."""


class ConditionalCheckFailed(StateError):
    """Conditional write not applied (if_not_exists / if_version)."""


class StateThrottled(StateError):
    """Operation rate limit exceeded (DynamoDB throttle / OnDemandThroughput cap)."""


class StateUnavailable(StateError):
    """Broker unreachable / 5xx / timeout. For a mutation the outcome is unknown
    (do not blindly retry — see the shared-table design spec §2.6)."""


# --- runner-visible outage signal (mindshub_services PR #164 item 11) -------
#
# When the generated backend's own FastAPI/Mangum handler lets one of these
# escape, Starlette's ServerErrorMiddleware turns it into an in-band 500
# response — it does NOT propagate out as a Lambda FunctionError. That means
# the gateway's runner_state_error/runner_error split (keyed off FunctionError)
# never fires for this, the actual common case. artifact_runner records the
# outage here so it can tag the 5xx response with a header the gateway reads
# and strips before it reaches the client (see functions/artifact_runner and
# functions/artifact_gateway/lambda_function.py in mindshub_services).
#
# Only StateThrottled/StateUnavailable are runner-visible: those are
# infrastructure-side failures, genuinely "look at the state plane, not the
# artifact's code" per README's promise. StateValidationError/
# ConditionalCheckFailed are the artifact's OWN logic (bad input, lost an
# optimistic-concurrency race) — tagging those as a state outage would send an
# operator to the wrong place. Kept in sync with artifact_gateway's
# _STATE_ERROR_TYPES (mindshub_services), which makes the same distinction for
# the FunctionError path.
_RUNNER_VISIBLE_ERRORS = (StateThrottled, StateUnavailable)

# Module-level, not a contextvar: a Lambda execution environment runs one
# request at a time, and mindshub_services' artifact_runner resets this (via
# pop_last_state_error()) immediately before every invoke of backend.handler —
# a co-tenant sharing the warm container can't inherit a stale value, because
# artifact_runner's _evict_stale_artifact_dirs already evicts this whole
# module (by its /tmp/<artifact>_<md5>/ origin path) before a *different*
# artifact's code runs, and a repeat invoke of the *same* artifact gets an
# explicit pop-and-clear regardless.
_last_state_error: str | None = None


def _record(exc: Exception) -> None:
    """Record a runner-visible outage. No-op for artifact-logic errors."""
    global _last_state_error
    if isinstance(exc, _RUNNER_VISIBLE_ERRORS):
        _last_state_error = type(exc).__name__


def pop_last_state_error() -> str | None:
    """Read-and-clear. artifact_runner calls this both immediately before
    invoking backend.handler (discarding whatever is there, so a value from a
    previous invocation never bleeds into this one) and immediately after (to
    get this invocation's own value, if any)."""
    global _last_state_error
    val = _last_state_error
    _last_state_error = None
    return val
