"""Breaking the idle-scratchpad loop.

Some models (MindsHub Air, when it served Kimi) sometimes announce a
connector call in a scratchpad cell — `print('Querying HubSpot next.')` —
instead of making it, then repeat that every round until the spend ceiling,
about 1.2M tokens later. After `IDLE_ROUNDS_BEFORE_WITHHOLD` such rounds in a
row the tool loop sends the next round without the scratchpad tool, so the
model has to call a real tool or answer.
"""
from __future__ import annotations

import ast
from typing import Any, Iterable

from anton.core.llm.prompts import ANTON_NOTE

SCRATCHPAD = "scratchpad"
IDLE_ROUNDS_BEFORE_WITHHOLD = 2

IDLE_SCRATCHPAD_NOTICE = (
    f"{ANTON_NOTE} Your last scratchpad calls did nothing: they only printed fixed messages "
    "or ran no code. The scratchpad is not available for this step. Call the tool you "
    "actually need now (a connected service's tools are named <service>__<tool>), "
    "or answer the user."
)


def _is_inert_statement(node: ast.stmt) -> bool:
    if isinstance(node, ast.Pass):
        return True
    if not isinstance(node, ast.Expr):
        return False
    value = node.value
    if isinstance(value, ast.Constant):
        return True
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "print"
        and all(isinstance(arg, ast.Constant) for arg in value.args)
        and all(isinstance(kw.value, ast.Constant) for kw in value.keywords)
    )


def is_idle_scratchpad_call(name: str, tool_input: Any) -> bool:
    """A scratchpad call that does no work: removing a pad, or running code
    that is empty, comments only, or prints only literals. Anything that
    computes, reads a variable or fails to parse counts as real work."""
    if name != SCRATCHPAD or not isinstance(tool_input, dict):
        return False
    action = tool_input.get("action")
    if action == "remove":
        return True
    if action != "exec":
        return False
    code = tool_input.get("code") or ""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return all(_is_inert_statement(node) for node in tree.body)


def round_is_idle(tool_calls: Iterable[Any]) -> bool:
    calls = list(tool_calls)
    return bool(calls) and all(is_idle_scratchpad_call(tc.name, tc.input) for tc in calls)


def without_scratchpad(tools: list[dict]) -> list[dict]:
    return [tool for tool in tools if tool.get("name") != SCRATCHPAD]
