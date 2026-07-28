"""Cloud turn: run one full anton turn inside a sandbox pod.

`python -m anton.cloud_turn` reads a :class:`TurnRequestV1` as one JSON line on
stdin, runs the turn to completion against the mounted workspace with a
cloud-safe session, and emits `delta` / `turn_completed` / `turn_failed` JSONL
events on stdout (diagnostics to stderr). It is the headless counterpart of the
desktop CLI host: the same ChatSession, built cloud-safe.
"""

from anton.cloud_turn.contract import TurnRequestV1
from anton.cloud_turn.session import build_cloud_chat_session

__all__ = ["TurnRequestV1", "build_cloud_chat_session"]
