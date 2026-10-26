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

Two verification passes, same day: the first grant only carried read-level
CRM scopes, so `manage_crm_objects` (the primary "write a CRM record" tool)
was entirely absent from `tools/list()` — `get_user_details` showed
`CONTACT`/`DEAL`/`COMPANY`/etc. as `"write": "REQUIRES_REAUTHORIZATION"`.
After enabling the `crm.objects.{contacts,companies,deals}.write` scopes on
the app (HubSpot gates optional scopes per-app before they're even
selectable at consent time) and re-authorizing with those three checked,
`manage_crm_objects` appeared live, correctly write-tier, and
`get_user_details` confirmed `CONTACT`/`COMPANY`/`DEAL` all flipped to
`"write": "AVAILABLE"`. The live tool list now matches this table exactly
in both directions (no unclassified extras, no stale entries).

`query_crm_data` (SQL-shaped) and `manage_onboarding` (its own description
contradicts its write-shaped name) are the two the original source doc left
ambiguous; both are kept on the safe side of the line they're closest to.

A second table, ``TOOL_OUTPUT``, says what each read-tier tool returns:
``"records"`` (the customer's own data, such as CRM objects, campaign or
email analytics) or ``"metadata"`` (docs, schema, property definitions,
account or user details). The access tier answers "may this connection call
it"; this table answers "is its output evidence for a figure in an answer".
A caller that attributes an answer's numbers to tool calls, such as
cowork-server's figure sources, should count only ``returns_records``
tools, so a limit quoted from ``tool_guidance`` or a ``displayOrder`` from
``get_properties`` never reads as a sourced figure. Write-tier tools stay
out of it too. Some also view data (``manage_aeo_prompts`` can view tracked
prompts), but a tool's name can't tell a view from a write, so none of
their output counts. The same fail-closed default applies: an unlisted tool
or engine returns no records.
"""

from __future__ import annotations

from typing import Literal

AccessMode = Literal["read", "write", "none"]
ToolOutput = Literal["records", "metadata"]

#: engine -> {tool_name: "read" | "write"}. Live-verified 2026-09-15 against
#: a real HubSpot MCP Auth App with both read- and write-scoped grants (see
#: module docstring) — not doc-derived anymore. `render_landing_page_ui`,
#: `render_asset`, and `manage_blog_post` (all in the original doc-derived
#: table) never appeared in either live `tools/list()` run — removed here as
#: stale. If a future live check shows them again, re-add with their
#: original classification (read/read/write). A new read-tier tool also
#: needs a TOOL_OUTPUT entry (below) saying whether it returns records.
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
        "manage_crm_objects": "write",  # confirmed live 2026-09-15 with a write-scoped grant
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


#: engine -> {tool_name: "records" | "metadata"} for every read-tier tool in
#: TOOL_ACCESS (a test keeps the two in step). "records" means the output is
#: the customer's own data; "metadata" means docs, schema or account details
#: whose numbers (limits, display orders, ids) are not figures about the
#: business. Write-tier tools are not listed (see module docstring).
TOOL_OUTPUT: dict[str, dict[str, ToolOutput]] = {
    "hubspot": {
        # RECORDS
        "search_crm_objects": "records",  # CRM objects matching filters
        "get_crm_objects": "records",  # CRM objects by id
        "query_crm_data": "records",  # SQL-shaped query over CRM objects
        "read_campaign_data": "records",  # campaign analytics, asset metrics, contact ids
        "get_campaign_attribution_reports": "records",  # attributed revenue/contacts
        "get_marketing_email_analytics": "records",  # sends, opens, clicks
        "get_content_analytics_report": "records",  # page/blog views, sessions
        "get_aeo_metrics": "records",  # AI-answer (AEO) visibility metrics
        "search_intent_signals": "records",  # buyer intent signals
        "search_conversations": "records",  # inbox threads and messages
        # METADATA
        "tool_guidance": "metadata",  # HubSpot's usage docs for its own tools
        "discover_hubspot_schema": "metadata",  # object types and associations
        "get_properties": "metadata",  # property definitions
        "search_properties": "metadata",  # property definitions
        "get_user_details": "metadata",  # the connected user and their scopes
        "get_organization_details": "metadata",  # the HubSpot account itself
        "search_owners": "metadata",  # the portal's users (owner ids, names)
        "get_conversation_channel_metadata": "metadata",  # inbox channel config
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


def returns_records(*, engine: str, tool: str) -> bool:
    """Whether `tool`'s output is the customer's own records, the only kind
    of output that counts as evidence for a figure in an answer.

    True only for a read-tier tool this engine's TOOL_OUTPUT marks
    "records". False for metadata tools, write-tier tools, and any tool or
    engine TOOL_OUTPUT does not list. `tool` is the bare tool name; split a
    registered name with `names.parse_namespaced_tool_name` first.
    """
    if not allowed_for_mode(engine, tool, "read"):
        return False
    return TOOL_OUTPUT.get(engine, {}).get(tool) == "records"
