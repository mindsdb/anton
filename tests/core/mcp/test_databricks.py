from __future__ import annotations

import pytest

from anton.core.mcp.databricks import (
    GENIE_ONE_PATH,
    PER_SPACE_PATH,
    InvalidGenieSpace,
    InvalidWorkspaceHost,
    canonical_workspace_host,
    genie_mcp_url,
)

HOST = "dbc-1.cloud.databricks.com"
SPACE = "01f0a1b2c3d4e5f60718293a4b5c6d7e"


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


def test_no_space_id_gives_the_genie_one_address():
    assert genie_mcp_url(f"https://{HOST}/") == f"https://{HOST}{GENIE_ONE_PATH}"


def test_a_space_id_gives_the_per_space_address():
    url = genie_mcp_url(HOST, SPACE)
    assert url == f"https://{HOST}" + PER_SPACE_PATH.format(space_id=SPACE)
    assert url.endswith(SPACE)


def test_a_space_id_is_stripped_of_surrounding_whitespace():
    assert genie_mcp_url(HOST, f"  {SPACE}\n") == genie_mcp_url(HOST, SPACE)


@pytest.mark.parametrize(
    "space_id",
    [
        "../x",
        "a/b",
        "x?y",
        "",
        " ",
        "a" * 200,
        SPACE.upper(),
        SPACE[:-1],
        SPACE + "0",
        SPACE[:-1] + "g",
        SPACE[:-1] + "٣",
    ],
)
def test_a_malformed_space_id_is_refused(space_id):
    with pytest.raises(InvalidGenieSpace) as raised:
        genie_mcp_url(HOST, space_id)
    assert space_id.strip() == "" or space_id.strip() not in str(raised.value)


def test_a_bad_host_is_refused_before_the_space_id_is_looked_at():
    with pytest.raises(InvalidWorkspaceHost):
        genie_mcp_url("https://evil.example", "../x")
