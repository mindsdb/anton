"""Registered MCP tool names: `namespaced_tool_name` builds one, and
`parse_namespaced_tool_name` splits one back into engine, connection and
tool.

They live apart from `registry`, which imports the MCP SDK through `client`,
so a caller that only reads a stored tool row's name, such as cowork-server,
splits it without loading the SDK. `registry`'s module docstring says why
every MCP tool name is namespaced.
"""

from __future__ import annotations

from dataclasses import dataclass


def namespaced_tool_name(engine: str, tool_name: str, *, connection_name: str | None = None) -> str:
    """`{engine}__{tool_name}` in the common single-connection-per-engine
    case; `{engine}__{connection_name}__{tool_name}` when `connection_name`
    is given (multiple connections for this engine in the same turn, see
    `registry`'s module docstring). Left out by default so the common case's
    tool names stay exactly what every existing test/doc example already
    shows."""
    if connection_name is None:
        return f"{engine}__{tool_name}"
    return f"{engine}__{connection_name}__{tool_name}"


@dataclass(frozen=True)
class ToolNameParts:
    """A registered MCP tool name split back into what built it."""

    engine: str
    connection_name: str | None
    tool_name: str


def parse_namespaced_tool_name(*, name: str) -> ToolNameParts | None:
    """The inverse of `namespaced_tool_name`, or None when `name` is not a
    namespaced MCP tool name (a built-in tool such as `web_fetch`), or when
    its engine, tool or connection is empty.

    The engine is the text before the first `__` and the tool is the text
    after the last `__`. Anything between is the connection name. A
    connection's name is free text from the vault and may start or end with
    `_` or hold `__`, while tool names come from the MCP server's fixed list,
    so the split trusts the tool side: a tool name that holds `__` never
    parses back, and one that starts with `_` parses back only when no
    connection name is present.
    """
    engine, _, rest = name.partition("__")
    connection_name, separator, tool_name = rest.rpartition("__")
    if not engine or not tool_name or (separator and not connection_name):
        return None
    return ToolNameParts(engine=engine, connection_name=connection_name or None, tool_name=tool_name)
