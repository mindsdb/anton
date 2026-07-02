"""build_datasource_context surfaces a non-secret identity per connection.

ENG-508: the LLM must be able to tell connections apart (which Gmail, which DB)
without exposing secrets — so the system-prompt section shows the account email
or the DB host/name, never the credential, and never a dump of opaque/config
fields.
"""
from anton.core.datasources.data_vault import LocalDataVault
from anton.utils.datasources import _connection_identity, build_datasource_context


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
        assert "db.acme.com/sales" in ctx
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

    def test_label_preferred_over_email(self, tmp_path):
        v = LocalDataVault(tmp_path)
        v.save(
            "gmail", "support",
            {"email": "regtr@mail.com", "app_password": "x", "_label": "Support"},
        )
        ctx = build_datasource_context(v)
        assert "Support" in ctx       # the human label is shown
        assert "regtr@mail.com" not in ctx  # label preferred over the opaque email
