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
``tool_guidance`` has no verb prefix at all, and ``manage_onboarding``'s
description reads as partly read despite its write-shaped name). A
hardcoded table is the right shape instead — kept with a **fail-closed
default**: any tool name not in the table, including ones a provider adds
later, is treated as write-tier and only surfaces for a `write`-mode
connection. It never falls through to read just because it looks harmless.

**Live-verified 2026-09-15** (ENG-1816) against a real HubSpot MCP Auth App
+ test account (`anton/core/mcp/client.py`'s `McpSession` pointed directly
at `https://mcp.hubspot.com`, bypassing Stage 2/3 entirely) — the table
below reflects that live `tools/list()` response, not just the original
doc. Confirmed the fail-closed default works as intended: 5 tools this
account's server advertised that weren't yet in this table (AEO/intent
features added since the table was first drafted) were correctly excluded
from a read-mode connection's registered tools until classified here.
"""

from __future__ import annotations

from typing import Literal

AccessMode = Literal["read", "write", "none"]

#: engine -> {tool_name: "read" | "write"}. Live-verified 2026-09-15 against
#: a real HubSpot MCP Auth App (see module docstring) — not just doc-derived
#: anymore, though two things from that live run are still worth tracking:
#:
#: 1. `render_landing_page_ui`, `render_asset`, and `manage_blog_post` (all
#:    in the original doc-derived table) did not appear in the live
#:    `tools/list()` response at all — removed here as stale. If a future
#:    live check shows them again (e.g. under a different scope grant),
#:    re-add with their original classification (read/read/write).
#: 2. `manage_crm_objects` — the primary "write a CRM record" tool — also
#:    did not appear live. `get_user_details`'s own response showed most
#:    write-capable CRM object types (CONTACT, DEAL, COMPANY, TICKET, ...)
#:    as `"write": "REQUIRES_REAUTHORIZATION"`, so this reads as: the test
#:    OAuth grant only requested read-level access, and the server omits
#:    write tools from `tools/list()` entirely until a broader grant exists
#:    — not a permanent removal. Kept in the table (harmless: a tool the
#:    server never returns is never filtered, it's just absent), but
#:    **before Stage 2 assumes write mode can create/update basic CRM
#:    records, re-verify with a re-auth that actually grants write scope.**
#:
#: `query_crm_data` (SQL-shaped) and `manage_onboarding` (its own
#: description contradicts its write-shaped name) are the two the original
#: source doc left ambiguous; both are kept on the safe side of the line
#: they're closest to.
TOOL_ACCESS: dict[str, dict[str, str]] = {
    "hubspot": {
        # READ
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
        "tool_guidance": "read",
        # Confirmed live 2026-09-15: pure-query tools (AGGREGATE/SEARCH/LIST
        # operations only, no create/update shape in their schema).
        "get_aeo_metrics": "read",
        "search_intent_signals": "read",
        # WRITE
        "manage_crm_objects": "write",  # not seen live yet — see note above
        "manage_campaign_objects": "write",
        "manage_marketing_email": "write",
        "manage_landing_page": "write",
        "submit_feedback": "write",
        # Confirmed live 2026-09-15: each has an explicit create/mutate
        # operation (manage_segment: CreateStaticSegmentOperation /
        # AddMembersOperation; manage_aeo_prompts: CREATE;
        # manage_aeo_recommendations: START_ACTION, e.g. "publish a blog post").
        "manage_segment": "write",
        "manage_aeo_prompts": "write",
        "manage_aeo_recommendations": "write",
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
