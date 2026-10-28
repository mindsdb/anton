"""How the Connected Data Sources section describes MCP-method connections."""

from anton.core.datasources.data_vault import LocalDataVault
from anton.core.mcp.servers import MCP_METHODS
from anton.utils import datasources
from anton.utils.datasources import build_datasource_context

_GENIE_FIELDS = {
    "workspace_url": "https://acme-workspace.cloud.databricks.com",
    "client_id": "sp-client-id-1234",
    "_client_secret": "GENIESECRET999",
    "_method": "genie-mcp",
}


def _block(ctx: str, slug: str) -> list[str]:
    """The lines of one connection's block, from its slug header to the next blank line."""
    lines = ctx.splitlines()
    start = next(i for i, line in enumerate(lines) if f"Slug: `{slug}`" in line)
    end = next((i for i in range(start + 1, len(lines)) if not lines[i].strip()), len(lines))
    return lines[start:end]


class TestGenieConnectionBlock:
    def test_genie_points_at_tools_without_credential_vars(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save("databricks", "genie", dict(_GENIE_FIELDS))

        block = _block(build_datasource_context(v), "databricks-genie")
        text = "\n".join(block)

        assert "Engine: Databricks" in text
        assert "databricks__genie_ask" in text
        assert "databricks__genie_poll_response" in text
        assert "databricks__genie_get_query_result" in text
        assert "no SQL driver connection" in text
        assert "Credential env vars:" not in text
        assert "DS_" not in text
        for stored in ("acme-workspace", "sp-client-id-1234", "GENIESECRET999"):
            assert stored not in text

    def test_pat_databricks_connection_renders_as_before(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save("databricks", "genie", dict(_GENIE_FIELDS))
        v.save("databricks", "wh", {"host": "dbc.example.com", "access_token": "PATSECRET"})

        block = _block(build_datasource_context(v), "databricks-wh")

        assert "Credential env vars:" in block
        assert " - DS_DATABRICKS_WH__HOST" in block
        assert " - DS_DATABRICKS_WH__ACCESS_TOKEN" in block
        assert "databricks__genie_ask" not in "\n".join(block)

    def test_hubspot_mcp_block_is_unchanged(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save("hubspot", "crm", {"access_token": "HUBTOK", "_method": "mcp"})

        block = _block(build_datasource_context(v), "hubspot-crm")

        assert block[1:] == [
            "Engine: HubSpot",
            "Credential env vars:",
            " - DS_HUBSPOT_CRM__ACCESS_TOKEN",
        ]

    def test_two_genie_connections_name_the_tools_registration_gives_them(self, tmp_path):
        """Two Genie connections register their tools under the connection
        name, so each block must name those, not the single-connection ones."""
        v = LocalDataVault(tmp_path)
        v.save("databricks", "sales", dict(_GENIE_FIELDS))
        v.save("databricks", "ops", dict(_GENIE_FIELDS))

        ctx = build_datasource_context(v)

        for name in ("sales", "ops"):
            text = "\n".join(_block(ctx, f"databricks-{name}"))
            assert f"`databricks__{name}__genie_ask`" in text
            assert f"`databricks__{name}__genie_poll_response`" in text
            assert f"`databricks__{name}__genie_get_query_result`" in text
            assert f"`databricks__{name}__genie_*`" in text
            assert "databricks__genie_" not in text

    def test_a_genie_connection_beside_a_pat_one_keeps_the_plain_tool_names(self, tmp_path):
        """A PAT connection registers no MCP tools, so it does not make Genie's names longer."""
        v = LocalDataVault(tmp_path)
        v.save("databricks", "genie", dict(_GENIE_FIELDS))
        v.save("databricks", "wh", {"host": "dbc.example.com", "access_token": "PATSECRET"})

        text = "\n".join(_block(build_datasource_context(v), "databricks-genie"))

        assert "`databricks__genie_ask`" in text
        assert "databricks__genie__" not in text

    def test_a_genie_connection_with_tools_off_names_no_tools(self, tmp_path):
        """Access mode "none" registers no Genie tools, so the block must not point at any."""
        v = LocalDataVault(tmp_path)
        v.save("databricks", "genie", {**_GENIE_FIELDS, "_access_mode": "none"})

        block = _block(build_datasource_context(v), "databricks-genie")

        assert block[1:] == ["Engine: Databricks"]


def test_the_prompt_counts_the_methods_discovery_counts():
    assert datasources._MCP_METHODS == MCP_METHODS
