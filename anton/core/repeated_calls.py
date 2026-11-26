"""Breaking identical-call loops.

A model can send the same tool call again and again, get the same result each
time, and keep going until the round cap: one cell 236 times in a single turn.
From `REPEATS_BEFORE_NOTICE` identical (call, result) pairs in a turn the result
says so, and from `REPEATS_BEFORE_WITHHOLD` the tool is left out of the next
round, so the model has to change course or answer.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from anton.core.llm.prompts import ANTON_NOTE

REPEATS_BEFORE_NOTICE = 3
REPEATS_BEFORE_WITHHOLD = 5

# An object's repr carries its memory address ("at 0x7f..."), which differs on every run of the
# same cell. Other hex values (hashes, ids) are real changes and stay as they are.
_ADDRESS = re.compile(r"(?<=\bat )0x[0-9a-fA-F]+")


class RepeatedCalls:
    """Counts identical (tool, input, result) triples over one turn."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def record(self, name: str, tool_input: Any, result_text: str) -> int:
        """Record one call and return how many times this turn it has run with this result."""
        key = json.dumps(
            [name, tool_input, _ADDRESS.sub("0x", result_text)], sort_keys=True, default=str,
        )
        digest = hashlib.sha256(key.encode()).hexdigest()
        self._counts[digest] = count = self._counts.get(digest, 0) + 1
        return count


def repeat_notice(name: str, count: int, withheld: bool) -> str:
    notice = (
        f"\n\n{ANTON_NOTE} You have made exactly this '{name}' call {count} times this turn "
        "and got exactly this result each time. Running it again will not change it. "
        "Change the code or the approach, or stop and say what is blocking you. "
        "If you are waiting for something to finish, wait inside one call instead."
    )
    if withheld:
        notice += f" The '{name}' tool is not available for your next step."
    return notice


def without_tools(tools: list[dict], names: set[str]) -> list[dict]:
    """`tools` minus `names`, or `tools` unchanged when nothing else would be left."""
    kept = [tool for tool in tools if tool.get("name") not in names]
    return kept or tools
