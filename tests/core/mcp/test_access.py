from __future__ import annotations

from anton.core.mcp.access import TOOL_ACCESS, TOOL_OUTPUT, allowed_for_mode, returns_records


def test_none_mode_allows_nothing():
    assert allowed_for_mode("hubspot", "get_user_details", "none") is False
    assert allowed_for_mode("hubspot", "manage_crm_objects", "none") is False


def test_read_mode_allows_only_read_classified_tools():
    assert allowed_for_mode("hubspot", "search_crm_objects", "read") is True
    assert allowed_for_mode("hubspot", "manage_crm_objects", "read") is False


def test_write_mode_allows_everything_the_engine_advertises():
    assert allowed_for_mode("hubspot", "search_crm_objects", "write") is True
    assert allowed_for_mode("hubspot", "manage_crm_objects", "write") is True


def test_unrecognized_tool_name_fails_closed_to_write_only():
    """A tool the classification table has never seen (a future HubSpot
    addition, or any other unlisted name) must never be treated as read just
    because it looks harmless."""
    assert allowed_for_mode("hubspot", "some_new_tool_added_later", "read") is False
    assert allowed_for_mode("hubspot", "some_new_tool_added_later", "write") is True


def test_unrecognized_engine_fails_closed_to_write_only():
    assert allowed_for_mode("an-engine-with-no-table", "anything", "read") is False
    assert allowed_for_mode("an-engine-with-no-table", "anything", "write") is True


def test_the_two_doc_ambiguous_tools_are_classified_on_the_safe_side():
    # query_crm_data: SQL-shaped, doc left it ambiguous — classified read.
    assert allowed_for_mode("hubspot", "query_crm_data", "read") is True
    # manage_onboarding: write-shaped name, read-shaped description — fails
    # closed to write.
    assert allowed_for_mode("hubspot", "manage_onboarding", "read") is False
    assert allowed_for_mode("hubspot", "manage_onboarding", "write") is True


def test_tools_added_since_the_original_table_was_drafted_are_classified():
    """Confirmed live 2026-09-15 against a real HubSpot MCP server — these 5
    tools weren't in the original doc-derived table at all."""
    # Pure-query tools confirmed live: no create/mutate operation in their schema.
    assert allowed_for_mode("hubspot", "get_aeo_metrics", "read") is True
    assert allowed_for_mode("hubspot", "search_intent_signals", "read") is True
    # Tools with an explicit create/mutate operation confirmed live.
    for write_tool in ("manage_segment", "manage_aeo_prompts", "manage_aeo_recommendations"):
        assert allowed_for_mode("hubspot", write_tool, "read") is False
        assert allowed_for_mode("hubspot", write_tool, "write") is True


def test_tools_no_longer_seen_live_are_not_in_the_table():
    """render_landing_page_ui/render_asset/manage_blog_post were in the
    original doc-derived table but didn't appear in a real tools/list() call
    (2026-09-15) — removed as stale. An unrecognized name still fails closed
    to write-tier (see test_unrecognized_tool_name_fails_closed_to_write_only),
    it's just no longer specifically classified read."""
    for removed_tool in ("render_landing_page_ui", "render_asset", "manage_blog_post"):
        assert allowed_for_mode("hubspot", removed_tool, "read") is False


def test_a_tool_that_returns_the_customers_records_returns_records():
    for data_tool in (
        "search_crm_objects",
        "get_crm_objects",
        "query_crm_data",
        "read_campaign_data",
        "get_campaign_attribution_reports",
        "get_marketing_email_analytics",
        "get_content_analytics_report",
        "get_aeo_metrics",
        "search_intent_signals",
        "search_conversations",
    ):
        assert returns_records(engine="hubspot", tool=data_tool) is True, data_tool


def test_a_read_tool_that_returns_docs_schema_or_account_metadata_does_not_return_records():
    for metadata_tool in (
        "tool_guidance",
        "discover_hubspot_schema",
        "get_properties",
        "search_properties",
        "get_user_details",
        "get_organization_details",
        "search_owners",
        "get_conversation_channel_metadata",
    ):
        assert returns_records(engine="hubspot", tool=metadata_tool) is False, metadata_tool


def test_a_write_tier_tool_never_returns_records():
    """Some write tools also view data (`manage_aeo_prompts` can view tracked
    prompts), but a tool's name can't tell a view from a write, so no write
    tool's output counts as the customer's records."""
    for write_tool, access in TOOL_ACCESS["hubspot"].items():
        if access == "write":
            assert returns_records(engine="hubspot", tool=write_tool) is False, write_tool


def test_an_unknown_engine_or_tool_does_not_return_records():
    assert returns_records(engine="hubspot", tool="some_new_tool_added_later") is False
    assert returns_records(engine="linear", tool="list_issues") is False


def test_a_read_tool_with_no_output_classification_does_not_return_records(monkeypatch):
    """A read tool added to TOOL_ACCESS before TOOL_OUTPUT says what it
    returns counts as no records, so an unclassified tool's numbers never
    read as sourced figures."""
    monkeypatch.setitem(TOOL_ACCESS, "hubspot", {**TOOL_ACCESS["hubspot"], "new_read_tool": "read"})
    assert returns_records(engine="hubspot", tool="new_read_tool") is False


def test_every_read_tier_tool_has_an_output_classification():
    """Adding a read tool to TOOL_ACCESS without saying what it returns fails
    here, so the two tables stay in step. Write-tier tools stay out of
    TOOL_OUTPUT because they never return records, and an engine TOOL_ACCESS
    does not list has no read tier, so its TOOL_OUTPUT entries would do
    nothing."""
    assert set(TOOL_OUTPUT) <= set(TOOL_ACCESS)
    for engine, tools in TOOL_ACCESS.items():
        read_tools = {name for name, access in tools.items() if access == "read"}
        assert set(TOOL_OUTPUT.get(engine, {})) == read_tools, engine


def test_a_write_tier_tool_listed_as_records_still_does_not_return_records(monkeypatch):
    """The read-tier check holds even when TOOL_OUTPUT wrongly lists a write
    tool, so a table mistake never turns a write tool's output into evidence."""
    monkeypatch.setitem(TOOL_OUTPUT, "hubspot", {**TOOL_OUTPUT["hubspot"], "manage_crm_objects": "records"})
    assert returns_records(engine="hubspot", tool="manage_crm_objects") is False
