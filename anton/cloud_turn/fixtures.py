"""Canonical V1 protocol JSON fixtures for downstream consumers.

Derived from the real Pydantic models so they can't drift from the wire
contract. Committed under ``schemas/fixtures/`` for scratchpad-controller and
cowork-server tests to consume. ``tests/test_cloud_turn_schemas.py`` regenerates
and diffs them, and validates each against the exported JSON Schema.

Run ``python -m anton.cloud_turn.fixtures`` to rewrite the committed files.
"""

from __future__ import annotations

import json
from pathlib import Path

from anton.cloud_turn.protocol import (
    ErrorCodeV1,
    MessageV1,
    TurnCompletedV1,
    TurnFailedV1,
    TurnRequestV1,
    TurnStartedV1,
    TurnErrorV1,
)
from anton.cloud_turn.schema_export import SCHEMA_DIR

FIXTURE_DIR = SCHEMA_DIR / "fixtures"


def _request_text_only() -> TurnRequestV1:
    return TurnRequestV1(
        run_id="run_1", attempt_id="att_1", conversation_id="conv_1",
        organization_id="acme", user_id="u1", workspace_id="proj_1",
        input="What is 2 + 2?",
    )


def _request_with_history() -> TurnRequestV1:
    return TurnRequestV1(
        run_id="run_2", attempt_id="att_1", conversation_id="conv_1",
        input="And multiply that by 10.",
        history=[
            MessageV1(role="user", content="What is 2 + 2?"),
            MessageV1(role="assistant", content="2 + 2 = 4."),
        ],
    )


def _started() -> TurnStartedV1:
    return TurnStartedV1(run_id="run_1", attempt_id="att_1", sequence=1)


def _completed_text_only() -> TurnCompletedV1:
    return TurnCompletedV1(
        run_id="run_1", attempt_id="att_1", sequence=2,
        final_text="2 + 2 = 4.",
        output_messages=[MessageV1(role="assistant", content="2 + 2 = 4.")],
    )


def _completed_tool_using() -> TurnCompletedV1:
    return TurnCompletedV1(
        run_id="run_3", attempt_id="att_1", sequence=2,
        final_text="The file has 3 lines.",
        output_messages=[
            MessageV1(role="assistant", content=[
                {"type": "text", "text": "Let me count the lines."},
                {"type": "tool_use", "id": "t1", "name": "scratchpad",
                 "input": {"action": "exec", "name": "main",
                           "code": "print(sum(1 for _ in open('data.txt')))"}},
            ]),
            MessageV1(role="user", content=[
                {"type": "tool_result", "tool_use_id": "t1", "content": "[output]\n3"},
            ]),
            MessageV1(role="assistant", content="The file has 3 lines."),
        ],
    )


def _failed() -> TurnFailedV1:
    return TurnFailedV1(
        run_id="run_4", attempt_id="att_1", sequence=2,
        error=TurnErrorV1(
            code=ErrorCodeV1.MODEL_PROVIDER_FAILURE,
            message="ConnectionError: the model provider is momentarily overloaded.",
            retryable=True,
        ),
    )


def _failed_invalid_request_null_ids() -> TurnFailedV1:
    return TurnFailedV1(
        run_id=None, attempt_id=None, sequence=1,
        error=TurnErrorV1(
            code=ErrorCodeV1.INVALID_REQUEST,
            message="InvalidRequestError: not valid JSON",
            retryable=False,
        ),
    )


#: name → model instance. The filename is ``<name>.json``.
FIXTURES = {
    "request-text-only": _request_text_only,
    "request-with-history": _request_with_history,
    "event-turn-started": _started,
    "event-turn-completed-text": _completed_text_only,
    "event-turn-completed-tool-using": _completed_tool_using,
    "event-turn-failed": _failed,
    "event-turn-failed-invalid-request-null-ids": _failed_invalid_request_null_ids,
}


def expected_files() -> dict[Path, str]:
    """Map of committed fixture path → expected serialized JSON."""
    out: dict[Path, str] = {}
    for name, factory in FIXTURES.items():
        model = factory()
        payload = json.loads(model.model_dump_json())
        out[FIXTURE_DIR / f"{name}.json"] = (
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    return out


def write_fixtures() -> list[Path]:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for path, content in expected_files().items():
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    for p in write_fixtures():
        print(f"wrote {p}")
