"""How the Connected Data Sources section describes MCP-method connections."""

from anton.core.datasources.data_vault import LocalDataVault
from anton.utils.datasources import build_datasource_context

_GENIE_FIELDS = {
    "workspace_url": "https://acme-workspace.cloud.databricks.com",
    "client_id": "sp-client-id-1234",
    "client_secret": "GENIESECRET999",
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
