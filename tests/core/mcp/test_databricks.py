from __future__ import annotations

import pytest

from anton.core.mcp.databricks import InvalidWorkspaceHost, canonical_workspace_host


@pytest.mark.parametrize(
    "value,host",
    [
        ("https://dbc-143742e4-8e44.cloud.databricks.com", "dbc-143742e4-8e44.cloud.databricks.com"),
        ("https://DBC-1.cloud.databricks.com/", "dbc-1.cloud.databricks.com"),
        ("https://dbc-1.cloud.databricks.com/?o=3683687545921416", "dbc-1.cloud.databricks.com"),
        ("dbc-1.cloud.databricks.com", "dbc-1.cloud.databricks.com"),
        ("https://adb-123.7.azuredatabricks.net", "adb-123.7.azuredatabricks.net"),
        ("https://1234.5.gcp.databricks.com", "1234.5.gcp.databricks.com"),
    ],
)
def test_a_workspace_address_is_reduced_to_its_host(value, host):
    assert canonical_workspace_host(value) == host


@pytest.mark.parametrize(
    "value",
    [
        "http://dbc-1.cloud.databricks.com",
        "https://dbc-1.cloud.databricks.com:8443",
        "https://dbc-1.cloud.databricks.com/sql/1.0",
        "https://user@dbc-1.cloud.databricks.com",
        "https://cloud.databricks.com",
        "https://dbc-1.cloud.databricks.com.evil.example",
        "https://evilcloud.databricks.com",
        "https://evil.example;.cloud.databricks.com",
        "https://evil%2fx.cloud.databricks.com",
        "https://evil\\x.cloud.databricks.com",
        "https://dbc-1.cloud.databricks.com.",
        "https://dbc-1.cloud.databricks.com/?o=1&x=2",
        "https://10.0.0.5",
        "https://169.254.169.254",
        "",
        "   ",
    ],
)
def test_anything_else_is_refused(value):
    with pytest.raises(InvalidWorkspaceHost):
        canonical_workspace_host(value)
