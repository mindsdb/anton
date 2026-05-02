"""Permission policy — Cowork-style safety gates per agent group.

Anton Dispatch wraps every inbound message with a :class:`PermissionPolicy`.
The policy is a small, declarative description of what an agent group is
allowed to do; the router consults it before forwarding gated tool calls
to the runtime, and emits an :class:`ActionCard` when user approval is
required.

Modeled on Claude Cowork's safety primitives (selective file access,
network allowlist, "Act without asking" mode, deletion protection,
scheduled-task caution). The mapping:

  - File scopes ↔ Cowork's "be selective about file access"
  - Network allowlist ↔ Cowork's network egress restrictions
  - Act-without-asking flag ↔ Cowork's "Act without asking" mode
  - Destructive-action gate ↔ Cowork's deletion permission prompt
  - Scheduled-task constraints ↔ Cowork's scheduled-task guidance

Policies are *per agent group*, not per session — they describe the
agent's standing capability, not a particular conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class GateDecision(str, Enum):
    """Outcome of evaluating a policy against a proposed action."""

    ALLOW = "allow"
    """Action is permitted; runtime may proceed."""

    DENY = "deny"
    """Action is forbidden; runtime is told the call failed (closed)."""

    PROMPT = "prompt"
    """Action is gated; router emits an ActionCard and pauses the runtime."""


@dataclass(frozen=True)
class FileScope:
    """A filesystem permission grant.

    Attributes:
        path: Directory or file the agent may access. Subpaths inherit
            unless an explicit deny scope shadows them.
        mode: ``"read"``, ``"write"``, or ``"read-write"``. ``"write"``
            implies the ability to create new files but **not** to delete
            existing ones — deletion always goes through the destructive
            gate.
    """

    path: str
    mode: Literal["read", "write", "read-write"] = "read"


@dataclass
class PermissionPolicy:
    """Declarative capability set for an agent group.

    Defaults are conservative — a freshly-created agent group with no
    explicit policy can only read its own workspace and respond in chat.
    Every escalation is opt-in.

    Attributes:
        file_scopes: Filesystem grants. Empty = workspace only.
        network_allowlist: Hostnames/domains the agent may reach via
            network egress tools. Empty = none. Matches both exact host
            and subdomains (``"github.com"`` matches ``"api.github.com"``).
            Note: web fetch / web search run server-side and are not
            governed here, matching Cowork's carve-out.
        mcp_allowlist: MCP server names this agent group may invoke.
            Empty = none.
        act_without_asking: When ``True``, the runtime does not pause
            between tool calls for confirmation. Off by default. Should
            only be enabled when the user is actively supervising and
            working with trusted inputs.
        require_approval_for_destructive: When ``True`` (the default),
            destructive actions (file deletion, irreversible API calls)
            emit an ActionCard regardless of ``act_without_asking``.
        scheduled_dispatch_allowed: Whether this agent group may be
            triggered by scheduled inbound. Off by default per Cowork's
            "start simple" guidance.
        scheduled_destructive_blocked: When ``True`` (the default),
            scheduled inbound can never escalate destructive gates —
            they auto-deny without prompting, since the user isn't there
            to click.
    """

    file_scopes: list[FileScope] = field(default_factory=list)
    network_allowlist: list[str] = field(default_factory=list)
    mcp_allowlist: list[str] = field(default_factory=list)
    act_without_asking: bool = False
    require_approval_for_destructive: bool = True
    scheduled_dispatch_allowed: bool = False
    scheduled_destructive_blocked: bool = True


@dataclass
class ProposedAction:
    """An action the runtime wants to take, evaluated by the policy.

    The dispatcher inspects pending tool calls before they execute and
    constructs a :class:`ProposedAction`. The policy gate returns a
    :class:`GateDecision`. For ``PROMPT`` decisions, the gate also
    supplies a human-readable reason that becomes the ActionCard prompt.
    """

    kind: Literal[
        "file_read",
        "file_write",
        "file_delete",
        "network_egress",
        "mcp_call",
        "shell_exec",
        "scheduled_trigger",
    ]
    target: str
    """What is being acted on — a path, hostname, MCP name, command, etc."""

    is_destructive: bool = False
    is_scheduled_context: bool = False


@dataclass
class GateResult:
    """Outcome of :func:`evaluate`."""

    decision: GateDecision
    reason: str = ""
    """Human-readable explanation. Surfaced in ActionCard prompts and
    in deny messages returned to the agent."""


def evaluate(policy: PermissionPolicy, action: ProposedAction) -> GateResult:
    """Evaluate a proposed action against a policy.

    Returns one of:
      - :attr:`GateDecision.ALLOW` — runtime proceeds silently.
      - :attr:`GateDecision.DENY` — runtime is told the call failed.
      - :attr:`GateDecision.PROMPT` — router emits an ActionCard.

    Pure function with no side effects; safe to call repeatedly.
    """

    # Scheduled-context destructive actions auto-deny when the user
    # explicitly asked to block them — there's nobody to click.
    if (
        action.is_scheduled_context
        and action.is_destructive
        and policy.scheduled_destructive_blocked
    ):
        return GateResult(
            GateDecision.DENY,
            "Destructive actions blocked in scheduled-dispatch context.",
        )

    if action.kind == "file_delete":
        # Deletion always prompts unless explicitly disabled — mirrors
        # Cowork's deletion-protection gate.
        if policy.require_approval_for_destructive:
            return GateResult(
                GateDecision.PROMPT,
                f"Allow deletion of `{action.target}`?",
            )
        return GateResult(GateDecision.ALLOW)

    if action.kind in ("file_read", "file_write"):
        if not _path_in_scopes(action.target, policy.file_scopes, action.kind):
            return GateResult(
                GateDecision.DENY,
                f"Path `{action.target}` is outside allowed file scopes.",
            )
        return GateResult(GateDecision.ALLOW)

    if action.kind == "network_egress":
        if not _host_in_allowlist(action.target, policy.network_allowlist):
            return GateResult(
                GateDecision.DENY,
                f"Host `{action.target}` is not in the network allowlist.",
            )
        return GateResult(GateDecision.ALLOW)

    if action.kind == "mcp_call":
        if action.target not in policy.mcp_allowlist:
            return GateResult(
                GateDecision.DENY,
                f"MCP server `{action.target}` is not allowlisted.",
            )
        return GateResult(GateDecision.ALLOW)

    if action.kind == "shell_exec":
        if action.is_destructive and policy.require_approval_for_destructive:
            return GateResult(
                GateDecision.PROMPT,
                f"Run command `{action.target}`?",
            )
        if not policy.act_without_asking:
            return GateResult(
                GateDecision.PROMPT,
                f"Run command `{action.target}`?",
            )
        return GateResult(GateDecision.ALLOW)

    if action.kind == "scheduled_trigger":
        if not policy.scheduled_dispatch_allowed:
            return GateResult(
                GateDecision.DENY,
                "Scheduled dispatch is not enabled for this agent group.",
            )
        return GateResult(GateDecision.ALLOW)

    return GateResult(GateDecision.DENY, f"Unknown action kind: {action.kind}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _path_in_scopes(path: str, scopes: list[FileScope], kind: str) -> bool:
    """Check whether ``path`` falls under any scope with sufficient mode."""
    needs_write = kind == "file_write"
    for scope in scopes:
        if not (path == scope.path or path.startswith(scope.path.rstrip("/") + "/")):
            continue
        if needs_write and scope.mode == "read":
            continue
        return True
    return False


def _host_in_allowlist(host: str, allowlist: list[str]) -> bool:
    """Check whether ``host`` matches any entry, including subdomain matches."""
    for entry in allowlist:
        if host == entry or host.endswith("." + entry):
            return True
    return False
