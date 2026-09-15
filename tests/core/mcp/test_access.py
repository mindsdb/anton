from __future__ import annotations

from anton.core.mcp.access import allowed_for_mode


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
