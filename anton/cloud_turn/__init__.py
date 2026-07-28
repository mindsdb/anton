"""Cloud turn runner: run one full anton turn inside a sandbox pod.

`python -m anton.cloud_turn` reads a :class:`TurnRequestV1` as JSON on stdin,
runs the turn to completion against the mounted workspace, and emits versioned
:class:`TurnEventV1` records as JSONL on stdout (diagnostics go to stderr). It is
the cloud counterpart of the in-process host harness: the same ChatSession, but
built cloud-safe (see :mod:`anton.cloud_turn.session`) and driven headlessly.
"""

from anton.cloud_turn.protocol import (
    PROTOCOL_VERSION,
    CapabilitiesV1,
    ErrorCodeV1,
    MessageV1,
    TurnCompletedV1,
    TurnErrorV1,
    TurnEventV1,
    TurnFailedV1,
    TurnRequestV1,
    TurnStartedV1,
    event_line,
    parse_request,
)

__all__ = [
    "PROTOCOL_VERSION",
    "CapabilitiesV1",
    "TurnRequestV1",
    "MessageV1",
    "TurnEventV1",
    "TurnStartedV1",
    "TurnCompletedV1",
    "TurnFailedV1",
    "TurnErrorV1",
    "ErrorCodeV1",
    "parse_request",
    "event_line",
]
