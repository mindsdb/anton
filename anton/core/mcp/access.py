"""Per-connection tool access governance: fail-closed read/write/none
filtering.

This filtering step — deciding which of an MCP server's tools even get
registered for a given connection — *is* the complete governance mechanism
for this feature. It's a static, pre-declared per-connection setting, not a
live approval flow, so it needs none of the machinery (prompts, per-action
dynamic gating) that ``anton/core/dispatch/``'s deleted orchestration layer
would have provided — that layer was never wired into any production
entrypoint in this repo or in cowork-server and was removed outright
(commit ``59bae27d``, "Cleanup dispatch"). There is no stub to extend here,
and no need to rebuild one.

A naming heuristic (``get_``/``search_`` -> read, ``manage_`` -> write) was
considered and rejected: HubSpot's own tool names break it in both
directions (``read_campaign_data`` uses a prefix in neither bucket,
``render_landing_page_ui``/``render_asset`` are display verbs that say
nothing about data access, ``tool_guidance`` has no verb prefix at all, and
``manage_onboarding``'s description reads as partly read despite its
write-shaped name). A hardcoded table is the right shape instead — kept with
a **fail-closed default**: any tool name not in the table, including ones a
provider adds later (HubSpot's own list has grown twice since GA), is
treated as write-tier and only surfaces for a `write`-mode connection. It
never falls through to read just because it looks harmless.
"""

from __future__ import annotations

from typing import Literal

AccessMode = Literal["read", "write", "none"]

#: engine -> {tool_name: "read" | "write"}. Doc-derived from HubSpot's public
#: integration page, not yet checked against a real `tools/list` call (no MCP
#: client existed to make one when this table was drafted) — confirm it
#: against the live response once Stage 1's client can, before Stage 2
#: testing relies on it being accurate. `query_crm_data` (SQL-shaped) and
#: `manage_onboarding` (its own description contradicts its write-shaped
#: name) are the two the source doc itself left ambiguous; both are kept on
#: the safe side of the line they're closest to.
TOOL_ACCESS: dict[str, dict[str, str]] = {
    "hubspot": {
        # READ (18)
        "get_user_details": "read",
        "get_organization_details": "read",
        "discover_hubspot_schema": "read",
        "search_crm_objects": "read",
        "get_crm_objects": "read",
        "query_crm_data": "read",
        "search_properties": "read",
        "get_properties": "read",
        "search_owners": "read",
        "get_campaign_attribution_reports": "read",
        "read_campaign_data": "read",
        "search_conversations": "read",
        "get_conversation_channel_metadata": "read",
        "get_marketing_email_analytics": "read",
        "get_content_analytics_report": "read",
        "render_landing_page_ui": "read",
        "render_asset": "read",
        "tool_guidance": "read",
        # WRITE (6)
        "manage_crm_objects": "write",
        "manage_campaign_objects": "write",
        "manage_marketing_email": "write",
        "manage_landing_page": "write",
        "manage_blog_post": "write",
        "submit_feedback": "write",
        # WRITE-GATED, AMBIGUOUS (1) — name says write, description reads as
        # partly read ("assess... and guide"); fail closed to write.
        "manage_onboarding": "write",
    },
}


def allowed_for_mode(engine: str, tool_name: str, access_mode: AccessMode) -> bool:
    """Whether `tool_name` should be registered for a connection at `access_mode`.

    `none` allows nothing (and callers should skip listing tools remotely at
    all for it — see `registry.discover_tool_defs`). `read` allows only
    tools this engine's table classifies "read"; an unrecognized name is
    never treated as read. `write` allows everything this engine advertises.
    """
    if access_mode == "none":
        return False
    if access_mode == "write":
        return True
    return TOOL_ACCESS.get(engine, {}).get(tool_name) == "read"
