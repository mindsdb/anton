"""A cloud turn's custom MCP tools: built from the job's ``custom_mcp`` block, called through the gateway.

The pod never holds a custom server's token or address. Each tool is a native
tool whose handler posts one call to the inference gateway with this turn's
key, and the gateway resolves, checks and dials the server. Building the tools
makes no request.

The tool list is part of the model's cached prefix, so it is put in one fixed
form here, the last hop before the model: tools sorted by registered name,
every schema rebuilt with sorted keys. Nothing of the connection's label,
version or id enters a definition, so the same switched-on tools give the
same bytes on every request and every turn.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx2 as httpx

from anton.cloud_turn.contract import (
    CUSTOM_MCP_PROTOCOL_VERSION,
    CustomMcpBlockV1,
    CustomMcpConnectionV1,
    CustomMcpToolV1,
)
from anton.core.mcp.names import registrable_tool_name
from anton.core.mcp.wiring import CUSTOM_MCP_ENGINE
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.tool_defs import ToolDef

logger = logging.getLogger(__name__)

#: The gateway's call route, relative to the turn's inference base, which ends in ``/v1``.
CALL_PATH = "/datasources/custom-mcp/call"
#: Over the gateway's own 45 second deadline, so its answer arrives first.
CALL_TIMEOUT_SECONDS = 50.0
#: The most characters of one answer the model sees.
MAX_RESULT_CHARS = 60_000
CUT_LINE = (
    "[Cut: this tool returned more text than can be shown. The first 60000 characters are above. "
    "Ask for less, for example with a filter or a smaller page.]"
)
DROPPED_LINE = "[Part of this answer, such as an image, could not be shown here.]"
_ERROR_BODY_LIMIT = 65_536
_ANSWER_KEYS = frozenset(
    {"protocol_version", "connection_id", "credential_version", "content", "is_error", "dropped_content_types"}
)

#: A fixed message per gateway code; the gateway's own text is never relayed.
_MESSAGES = {
    "needs_reconnect": "The connection to {server} needs to be reconnected before this tool can run again.",
    "tool_not_allowed": "This tool is switched off for {server}, so it cannot run.",
    "response_too_large": (
        "This tool returned more text than can be shown, so none of it is shown. "
        "Ask for less, for example with a filter or a smaller page."
    ),
    "rate_limited": "{Server} has had too many calls in a short time. Wait a moment, then try again.",
    "destination_forbidden": "{Server} is at an address that is not allowed, so this tool cannot run.",
    "invalid_request": "{Server} was not called: the arguments are not a valid JSON object or are too large.",
    "invalid_connection": "{Server} is not available in this conversation.",
}
_UNAVAILABLE = "{Server} is temporarily unavailable. Try again shortly."
_UNREADABLE = "{Server} sent an answer that could not be read."
#: The outcome reason per gateway code, each one tiered in `anton.core.root_cause`.
#: Codes the desktop MCP path shares keep its reasons, so both paths count alike.
_REASONS = {
    "needs_reconnect": "mcp_permanent_error",
    "destination_forbidden": "mcp_egress_blocked",
    "rate_limited": "mcp_transient_error",
    "tool_not_allowed": "custom_mcp_tool_switched_off",
    "invalid_connection": "custom_mcp_unknown_connection",
    "response_too_large": "custom_mcp_too_large",
    "invalid_request": "custom_mcp_invalid_request",
}


def build_custom_mcp_tools(
    block: CustomMcpBlockV1, *, gateway_base_url: str, turn_key: str, correlation_id: str
) -> list[ToolDef]:
    """The block's tools as native ``ToolDef``s in their fixed form, with no request made.

    Input: the parsed block, the turn's inference base (``https://host/v1``), its
    key and its correlation id. Output: one tool per definition, sorted by
    registered name, each schema with sorted keys. A tool with no valid
    registered name is left out with a warning, and a second tool under a name
    already taken is skipped, so the output never depends on block order.
    """
    url = gateway_base_url.rstrip("/") + CALL_PATH
    entries: list[tuple[str, int, CustomMcpConnectionV1, CustomMcpToolV1]] = []
    for connection in block.connections:
        for tool in connection.tools:
            name = registrable_tool_name(CUSTOM_MCP_ENGINE, tool.name, connection_name=connection.name)
            if name is None:
                logger.warning(
                    "custom MCP tool left out: no valid tool name for it",
                    extra={"event": "custom_mcp_tool_left_out", "connection_id": connection.connection_id},
                )
                continue
            entries.append((name, connection.connection_id, connection, tool))
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    tool_defs: list[ToolDef] = []
    taken: set[str] = set()
    for name, _, connection, tool in entries:
        if name in taken:
            continue
        taken.add(name)
        tool_defs.append(
            ToolDef(
                name=name,
                description=tool.description,
                input_schema=fixed_form(tool.input_schema),
                handler=_make_handler(
                    url=url, turn_key=turn_key, correlation_id=correlation_id, connection=connection, tool=tool.name
                ),
            )
        )
    return tool_defs


def fixed_form(value: Any) -> Any:
    """``value`` rebuilt with every object's keys sorted, at every depth; arrays keep their order.

    The parser has bounded the depth, so the recursion stays shallow. No side effects.
    """
    if isinstance(value, dict):
        return {key: fixed_form(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [fixed_form(item) for item in value]
    return value


def render_answer(content: list[dict], dropped: bool) -> str:
    """One string for the model: text parts, then structured content as JSON, cut at the cap.

    A longer answer is cut at exactly ``MAX_RESULT_CHARS`` characters and
    ``CUT_LINE`` follows. ``DROPPED_LINE`` follows when the gateway left a part out.
    """
    parts = [
        part["text"] if part["type"] == "text" else json.dumps(part["value"], ensure_ascii=False)
        for part in content
    ]
    text = "\n".join(parts)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + "\n" + CUT_LINE
    if dropped:
        text = f"{text}\n{DROPPED_LINE}" if text else DROPPED_LINE
    return text


def _server_words(label: str | None) -> dict[str, str]:
    """How a message names the server, in sentence case and mid-sentence."""
    server = f'the custom server "{label}"' if label else "the custom server"
    return {"server": server, "Server": server[0].upper() + server[1:]}


def _content_part_ok(part: object) -> bool:
    """Whether ``part`` is a text part or a structured part of the call contract."""
    if not isinstance(part, dict):
        return False
    if part.get("type") == "text":
        return set(part) == {"type", "text"} and isinstance(part["text"], str)
    return part.get("type") == "structured" and set(part) == {"type", "value"}


def _read_answer(body: bytes, connection: CustomMcpConnectionV1) -> dict | None:
    """The gateway's 200 answer, or None when it is not this call's answer under the contract."""
    try:
        answer = json.loads(body.decode("utf-8"))
    except (ValueError, RecursionError):
        return None
    if not isinstance(answer, dict) or set(answer) != _ANSWER_KEYS:
        return None
    binding = {
        "protocol_version": CUSTOM_MCP_PROTOCOL_VERSION,
        "connection_id": connection.connection_id,
        "credential_version": connection.credential_version,
    }
    # An answer that does not echo this exact binding is not this call's answer.
    if not all(type(answer[key]) is int and answer[key] == value for key, value in binding.items()):
        return None
    content, dropped = answer["content"], answer["dropped_content_types"]
    if (
        not isinstance(content, list)
        or not all(_content_part_ok(part) for part in content)
        or not isinstance(answer["is_error"], bool)
        or not isinstance(dropped, list)
        or not all(isinstance(kind, str) for kind in dropped)
    ):
        return None
    return answer


def _error_code(body: bytes) -> str | None:
    """The gateway's error code from a refusal body, or None when the body carries none."""
    if len(body) > _ERROR_BODY_LIMIT:
        return None
    try:
        error = json.loads(body.decode("utf-8"))
    except (ValueError, RecursionError):
        return None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else None


def _make_handler(*, url: str, turn_key: str, correlation_id: str, connection: CustomMcpConnectionV1, tool: str):
    """The async handler for one tool: one gateway call per model call, on a client of its own.

    Each call opens and closes its own client, so nothing outlives it and no
    event loop is shared. Arguments and answers are never logged.
    """
    words = _server_words(connection.label)

    async def handle_custom_mcp_call(session, tc_input: dict) -> ToolOutcome:
        arguments = {} if tc_input is None else tc_input
        if not isinstance(arguments, dict):
            return ToolOutcome(
                content=_MESSAGES["invalid_request"].format(**words), ok=False, reason="custom_mcp_invalid_request"
            )
        body = {
            "protocol_version": CUSTOM_MCP_PROTOCOL_VERSION,
            "correlation_id": correlation_id,
            "connection_id": connection.connection_id,
            "credential_version": connection.credential_version,
            "tool": tool,
            "arguments": arguments,
        }
        try:
            payload = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        except (ValueError, TypeError, RecursionError):
            return ToolOutcome(
                content=_MESSAGES["invalid_request"].format(**words), ok=False, reason="custom_mcp_invalid_request"
            )
        try:
            async with httpx.AsyncClient(timeout=CALL_TIMEOUT_SECONDS, follow_redirects=False) as client:
                response = await client.post(
                    url,
                    content=payload,
                    headers={
                        "Authorization": f"Bearer {turn_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
        except httpx.HTTPError as exc:
            logger.warning(
                "custom MCP call did not reach the gateway: %s",
                type(exc).__name__,
                extra={"event": "custom_mcp_call_failed", "connection_id": connection.connection_id},
            )
            return ToolOutcome(content=_UNAVAILABLE.format(**words), ok=False, reason="mcp_transient_error")
        if response.status_code != 200:
            code = _error_code(response.content)
            if code not in _MESSAGES:
                return ToolOutcome(content=_UNAVAILABLE.format(**words), ok=False, reason="mcp_transient_error")
            return ToolOutcome(content=_MESSAGES[code].format(**words), ok=False, reason=_REASONS[code])
        answer = _read_answer(response.content, connection)
        if answer is None:
            return ToolOutcome(content=_UNREADABLE.format(**words), ok=False, reason="custom_mcp_invalid_answer")
        text = render_answer(answer["content"], bool(answer["dropped_content_types"]))
        return ToolOutcome(content=text, ok=not answer["is_error"], reason="mcp_tool_error" if answer["is_error"] else "")

    return handle_custom_mcp_call
