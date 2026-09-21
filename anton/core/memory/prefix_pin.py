"""Pin the memory text that enters the cached prompt prefix for one conversation.

Memory rules and lessons are rendered into the system prompt and into the
scratchpad tool description. Both sit in front of the whole conversation in the
request, so if their bytes differ from the previous turn the entire conversation
is re-written to the prompt cache. They did differ on most turns: end-of-turn
consolidation adds rules between turns, and rule retrieval is keyed on the
current user message.

The pin records the rendered text the first time a conversation needs it and
serves that exact text for the rest of the conversation. New memories still get
written; they reach the prompt from the next conversation on. The host builds a
fresh session per turn, so the pin lives on disk next to the conversation's
scratchpad snapshots; without a conversation scope it lives in memory, which
still covers a long-lived CLI session.
"""

from __future__ import annotations

import json
from pathlib import Path


class MemoryPrefixPin:
    def __init__(self, path: Path | None):
        self._path = path
        self._values: dict[str, str] | None = None

    def get(self, key: str) -> str | None:
        self._load()
        assert self._values is not None
        return self._values.get(key)

    def set(self, key: str, value: str) -> None:
        self._load()
        assert self._values is not None
        self._values[key] = value
        self._save()

    def _load(self) -> None:
        if self._values is not None:
            return
        self._values = {}
        if self._path is None:
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            self._values = {k: v for k, v in data.items() if isinstance(v, str)}

    def _save(self) -> None:
        # Best-effort: losing the pin only returns the prompt to being rebuilt.
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._values), encoding="utf-8")
            tmp.replace(self._path)
        except OSError:
            pass
