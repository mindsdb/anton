"""build_datasource_context surfaces a non-secret identity per connection.

ENG-508: the LLM must be able to tell connections apart (which Gmail, which DB)
without exposing secrets — so the system-prompt section shows the account email
or the DB host/name, never the credential, and never a dump of opaque/config
fields.
"""
import json

from anton.core.datasources.data_vault import LocalDataVault
from anton.utils.datasources import _connection_identity, _parse_picked_files, build_datasource_context


class TestConnectionIdentity:
    def test_email_wins(self):
        assert _connection_identity({"email": "support@acme.com"}) == "support@acme.com"

    def test_host_and_database(self):
        assert _connection_identity({"host": "db.acme.com", "database": "sales"}) == "db.acme.com/sales"

    def test_host_only(self):
        assert _connection_identity({"host": "db.acme.com"}) == "db.acme.com"

    def test_no_identity_field(self):
        assert _connection_identity({"client_id": "opaque", "ssl_mode": "require"}) is None
        assert _connection_identity({}) is None

    def test_respects_secure_keys(self):
        # Defensive: a field a record marks secret is never surfaced as identity.
        assert _connection_identity({"email": "u@x.com"}, ["email"]) is None
        assert _connection_identity({"email": "u@x.com"}, []) == "u@x.com"
        assert _connection_identity({"host": "h", "database": "d"}, ["database"]) == "h"
        assert _connection_identity({"host": "h"}, ["host"]) is None

    def test_oauth_account_email(self):
        # OAuth flows store the address under `account_email`.
        assert _connection_identity({"account_email": "u@acme.com"}) == "u@acme.com"
        assert _connection_identity({"account_email": "u@acme.com"}, ["account_email"]) is None


class TestBuildContext:
    def test_shows_identity_not_secrets_or_opaque(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save("gmail", "support", {"email": "support@acme.com", "app_password": "SECRETVAL123"})
        v.save("postgres", "prod", {"host": "db.acme.com", "database": "sales", "password": "PGSECRET"})
        v.save("asana", "team", {"client_id": "opaque-guid", "access_token": "TOKSECRET"})

        ctx = build_datasource_context(v)

        # Identity is shown so the agent can pick the right account.
        assert "support@acme.com" in ctx
        assert "Host: db.acme.com" in ctx
        assert "Database: sales" in ctx
        # Secrets are never in the prompt (only their DS_* var name).
        assert "SECRETVAL123" not in ctx
        assert "PGSECRET" not in ctx
        assert "TOKSECRET" not in ctx
        assert "DS_GMAIL_SUPPORT__APP_PASSWORD" in ctx
        # Opaque/config field values are not surfaced as identity.
        assert "opaque-guid" not in ctx

    def test_empty_vault_returns_empty(self, tmp_path):
        assert build_datasource_context(LocalDataVault(tmp_path)) == ""

    def test_meta_fields_not_listed_as_env_vars(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save(
            "gmail", "support",
            {"email": "u@x.com", "app_password": "p", "_connector_id": "gmail",
             "_method": "app-password", "_label": "Support"},
        )
        ctx = build_datasource_context(v)
        # `_`-prefixed bookkeeping must not appear as DS_* env vars.
        assert "__CONNECTOR_ID" not in ctx
        assert "__METHOD" not in ctx
        assert "__LABEL" not in ctx
        assert "DS_GMAIL_SUPPORT__EMAIL" in ctx  # real fields still listed

    def test_label_and_email_both_shown(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save(
            "gmail", "support",
            {"email": "regtr@mail.com", "app_password": "x", "_label": "Support"},
        )
        ctx = build_datasource_context(v)
        assert "Label: Support" in ctx
        assert "Account: regtr@mail.com" in ctx


class TestUserLabelInPrompt:
    def test_user_label_preferred_over_legacy_label(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save(
            "gmail", "support",
            {"email": "reg@mail.com", "app_password": "x", "_label": "Old", "_user_label": "Support"},
        )
        ctx = build_datasource_context(v)
        assert "Label: Support" in ctx
        assert "Old" not in ctx

    def test_legacy_label_still_shown_when_no_user_label(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save(
            "gmail", "support",
            {"email": "reg@mail.com", "app_password": "x", "_label": "Support"},
        )
        ctx = build_datasource_context(v)
        assert "Label: Support" in ctx

    def test_no_label_shows_none(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save("postgres", "a1b2c3", {"host": "db.example.com", "database": "demo"})
        ctx = build_datasource_context(v)
        assert "Label: (none)" in ctx

    def test_slug_leads_the_block_in_backticks(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save("postgres", "a1b2c3", {"host": "db.example.com", "database": "demo", "_user_label": "prod-db"})
        ctx = build_datasource_context(v)
        assert "### Slug: `postgres-a1b2c3` — Label: prod-db" in ctx

    def test_per_field_lines_present(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save("postgres", "a1b2c3", {"host": "db.example.com", "database": "demo", "_user_label": "prod-db"})
        ctx = build_datasource_context(v)
        assert "Host: db.example.com" in ctx
        assert "Database: demo" in ctx
        assert "DS_POSTGRES_A1B2C3__HOST" in ctx

    def test_user_label_not_listed_as_env_var(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save("postgres", "a1b2c3", {"host": "x", "_user_label": "prod-db"})
        ctx = build_datasource_context(v)
        assert "__USER_LABEL" not in ctx

    def test_reference_by_slug_wording(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save("postgres", "a1b2c3", {"host": "x"})
        ctx = build_datasource_context(v)
        assert "Reference it by slug;" in ctx
        assert "Reference it by name;" not in ctx


class TestParsePickedFiles:
    def test_valid_list(self):
        raw = json.dumps([{"id": "1", "name": "a"}, {"id": "2", "name": "b"}])
        assert _parse_picked_files(raw) == [{"id": "1", "name": "a"}, {"id": "2", "name": "b"}]

    def test_drops_malformed_entries(self):
        raw = json.dumps([
            {"id": "1", "name": "keep"},
            "not-a-dict",
            {"name": "missing-id"},
            {"id": None, "name": "null-id"},
        ])
        assert _parse_picked_files(raw) == [{"id": "1", "name": "keep"}]

    def test_non_list_json_returns_empty(self):
        assert _parse_picked_files(json.dumps({"id": "1"})) == []

    def test_invalid_json_returns_empty(self):
        assert _parse_picked_files("not json") == []

    def test_empty_or_none_returns_empty(self):
        assert _parse_picked_files(None) == []
        assert _parse_picked_files("") == []


class TestGoogleDrivePickerContext:
    """ENG-687 gave google_drive's Picker-granted files their own prompt block
    here. ENG-2071 removed the *listing* again: it had no project to scope by,
    so it named files the user granted in other projects, re-opening the leak
    ConnectionsService.picked_files_by_project() closes on the cowork-server
    side. What stays here is the availability paragraph — no file ids."""

    def test_oauth_connection_without_picked_files_shows_availability_only(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save("google_drive", "work", {"auth_type": "oauth", "account_email": "u@x.com"})
        ctx = build_datasource_context(v)
        assert "Connected Google Drive accounts are available" in ctx
        assert "IMPORTANT" not in ctx

    def test_picked_files_are_never_named_here(self, tmp_path):
        """ENG-2071: the availability paragraph still fires, but no file
        identity reaches the prompt from this function."""
        v = LocalDataVault(tmp_path)
        v.save("google_drive", "work", {
            "auth_type": "oauth",
            "_picked_files": json.dumps([{"id": "f1", "name": "Roadmap.gdoc"}]),
        })
        ctx = build_datasource_context(v)
        assert "Connected Google Drive accounts are available" in ctx
        assert "Roadmap.gdoc" not in ctx
        assert "f1" not in ctx
        assert "IMPORTANT — additional Drive files" not in ctx

    def test_file_granted_in_another_project_is_not_named(self, tmp_path):
        """The leak itself. A file tagged to Project A must not surface at all
        from here — this function cannot tell which project is active, which is
        exactly why it no longer lists any of them."""
        v = LocalDataVault(tmp_path)
        v.save("google_drive", "work", {
            "auth_type": "oauth",
            "_picked_files": json.dumps([
                {"id": "secret-a", "name": "ProjectA-Salaries.gsheet", "projects": ["Project A"]},
            ]),
        })
        ctx = build_datasource_context(v)
        assert "ProjectA-Salaries.gsheet" not in ctx
        assert "secret-a" not in ctx

    def test_picker_only_connection_still_shows_availability(self, tmp_path):
        """A Picker-granted connection with no `auth_type` still triggers the
        paragraph — unchanged from before ENG-2071, which is why presence is
        still tracked even though the list is gone."""
        v = LocalDataVault(tmp_path)
        v.save("google_drive", "work", {
            "_picked_files": json.dumps([{"id": "f1", "name": "Roadmap.gdoc"}]),
        })
        ctx = build_datasource_context(v)
        assert "Connected Google Drive accounts are available" in ctx
        assert "Roadmap.gdoc" not in ctx

    def test_malformed_picked_file_entries_are_dropped_not_crashed(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save("google_drive", "work", {
            "_picked_files": json.dumps(["not-a-dict", {"name": "missing-id"}]),
        })
        ctx = build_datasource_context(v)  # must not raise
        # Nothing well-formed survived and there is no oauth marker, so the
        # connection reads as "no Picker grant" and no paragraph appears.
        assert "Connected Google Drive accounts are available" not in ctx

    def test_no_google_drive_connection_no_guidance(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save("postgres", "prod", {"host": "db.acme.com"})
        ctx = build_datasource_context(v)
        assert "Google Drive" not in ctx
        assert "IMPORTANT" not in ctx

    def test_multiple_google_drive_connections_name_no_files(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save("google_drive", "work", {"_picked_files": json.dumps([{"id": "1", "name": "Work.gdoc"}])})
        v.save("google_drive", "personal", {"_picked_files": json.dumps([{"id": "2", "name": "Personal.gdoc"}])})
        ctx = build_datasource_context(v)
        assert "Connected Google Drive accounts are available" in ctx
        assert "Work.gdoc" not in ctx
        assert "Personal.gdoc" not in ctx

    def test_active_only_suppresses_other_connections_guidance(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save("google_drive", "work", {
            "auth_type": "oauth",
            "_picked_files": json.dumps([{"id": "1", "name": "Roadmap.gdoc"}]),
        })
        v.save("postgres", "prod", {"host": "db.acme.com"})
        # A different connection is active — Drive guidance must not leak in.
        ctx = build_datasource_context(v, active_only="postgres-prod")
        assert "Google Drive" not in ctx
        assert "Roadmap.gdoc" not in ctx

    def test_active_only_on_google_drive_still_shows_availability(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save("google_drive", "work", {
            "_picked_files": json.dumps([{"id": "1", "name": "Roadmap.gdoc"}]),
        })
        ctx = build_datasource_context(v, active_only="google_drive-work")
        assert "Connected Google Drive accounts are available" in ctx
        assert "Roadmap.gdoc" not in ctx
