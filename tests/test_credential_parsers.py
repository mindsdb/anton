from __future__ import annotations

import json

from anton.core.datasources.datasource_registry import (
    DatasourceEngine,
    DatasourceField,
)
from anton.utils.credential_parsers import parse_credential_input


def postgres_engine() -> DatasourceEngine:
    return DatasourceEngine(
        engine="postgres",
        display_name="PostgreSQL",
        fields=[
            DatasourceField(name="host", required=True),
            DatasourceField(name="port", required=True, default="5432"),
            DatasourceField(name="database", required=True),
            DatasourceField(name="user", required=True),
            DatasourceField(name="password", required=True, secret=True),
            DatasourceField(name="schema", required=False),
            DatasourceField(name="ssl", required=False),
        ],
    )


def mysql_engine() -> DatasourceEngine:
    return DatasourceEngine(
        engine="mysql",
        display_name="MySQL",
        fields=[
            DatasourceField(name="host", required=True),
            DatasourceField(name="port", required=True, default="3306"),
            DatasourceField(name="database", required=True),
            DatasourceField(name="user", required=True),
            DatasourceField(name="password", required=True, secret=True),
        ],
    )


def test_uri_happy_path():
    result = parse_credential_input(
        "postgres://alice:secret@db.example.com:5432/analytics",
        postgres_engine(),
    )
    assert result is not None
    assert result.source == "uri"
    assert result.fields == {
        "host": "db.example.com",
        "port": "5432",
        "user": "alice",
        "password": "secret",
        "database": "analytics",
    }


def test_uri_url_encoded_password():
    result = parse_credential_input(
        "postgres://alice:p%40ss@db.example.com/analytics",
        postgres_engine(),
    )
    assert result is not None
    assert result.fields["password"] == "p@ss"


def test_uri_missing_components_only_sets_present_fields():
    result = parse_credential_input(
        "postgres://db.example.com/analytics",
        postgres_engine(),
    )
    assert result is not None
    assert result.fields == {
        "host": "db.example.com",
        "database": "analytics",
    }


def test_uri_wrong_scheme_for_engine_returns_none():
    result = parse_credential_input(
        "mysql://user:pw@host/db",
        postgres_engine(),
    )
    assert result is None


def test_uri_snowflake_rejected_for_postgres():
    result = parse_credential_input(
        "snowflake://user:pw@account.region/db",
        postgres_engine(),
    )
    assert result is None


def test_uri_postgresql_alias_accepted():
    result = parse_credential_input(
        "postgresql://alice:pw@db.example.com/analytics",
        postgres_engine(),
    )
    assert result is not None
    assert result.fields["host"] == "db.example.com"
    assert result.fields["user"] == "alice"


def test_uri_mariadb_alias_accepted_for_mysql():
    result = parse_credential_input(
        "mariadb://alice:pw@db.example.com/analytics",
        mysql_engine(),
    )
    assert result is not None
    assert result.fields["host"] == "db.example.com"


def test_uri_query_params_kept_when_engine_has_field():
    result = parse_credential_input(
        "postgres://alice:pw@host/db?schema=analytics&ssl=true&unknown=x",
        postgres_engine(),
    )
    assert result is not None
    assert result.fields["schema"] == "analytics"
    assert result.fields["ssl"] == "true"
    assert "unknown" not in result.fields


def test_uri_sslmode_dropped_when_engine_lacks_field():
    result = parse_credential_input(
        "postgres://alice:pw@host/db?sslmode=require",
        postgres_engine(),
    )
    assert result is not None
    assert "sslmode" not in result.fields


def test_json_all_fields_mapped():
    payload = json.dumps(
        {
            "host": "db.example.com",
            "port": "5432",
            "database": "analytics",
            "user": "alice",
            "password": "secret",
        }
    )
    result = parse_credential_input(payload, postgres_engine())
    assert result is not None
    assert result.source == "json"
    assert result.fields == {
        "host": "db.example.com",
        "port": "5432",
        "database": "analytics",
        "user": "alice",
        "password": "secret",
    }


def test_json_drops_unknown_keys():
    payload = json.dumps(
        {
            "host": "db.example.com",
            "database": "analytics",
            "bogus_field": "ignore me",
            "region": "us-east-1",
        }
    )
    result = parse_credential_input(payload, postgres_engine())
    assert result is not None
    assert result.fields == {
        "host": "db.example.com",
        "database": "analytics",
    }


def test_json_coerces_non_string_values():
    payload = json.dumps(
        {"host": "db.example.com", "port": 5432, "ssl": True}
    )
    result = parse_credential_input(payload, postgres_engine())
    assert result is not None
    assert result.fields["port"] == "5432"
    assert result.fields["ssl"] == "True"


def test_json_malformed_returns_none():
    result = parse_credential_input(
        "{not valid json}", postgres_engine()
    )
    assert result is None


def test_json_with_only_unknown_keys_returns_none():
    payload = json.dumps({"region": "us-east-1", "tenant": "acme"})
    result = parse_credential_input(payload, postgres_engine())
    assert result is None


def test_env_ref_to_uri(monkeypatch):
    monkeypatch.setenv(
        "TEST_DB_URL",
        "postgres://alice:pw@db.example.com:5432/analytics",
    )
    result = parse_credential_input(
        "$TEST_DB_URL", postgres_engine()
    )
    assert result is not None
    assert result.source == "env"
    assert result.fields["host"] == "db.example.com"
    assert result.fields["user"] == "alice"
    assert result.fields["port"] == "5432"


def test_env_ref_braced_syntax(monkeypatch):
    monkeypatch.setenv(
        "TEST_DB_URL",
        "postgres://alice:pw@db.example.com/analytics",
    )
    result = parse_credential_input(
        "${TEST_DB_URL}", postgres_engine()
    )
    assert result is not None
    assert result.source == "env"
    assert result.fields["host"] == "db.example.com"


def test_env_ref_to_json(monkeypatch):
    monkeypatch.setenv(
        "TEST_DB_JSON",
        json.dumps({"host": "db.example.com", "database": "analytics"}),
    )
    result = parse_credential_input(
        "$TEST_DB_JSON", postgres_engine()
    )
    assert result is not None
    assert result.source == "env"
    assert result.fields == {
        "host": "db.example.com",
        "database": "analytics",
    }


def test_env_ref_missing_variable(monkeypatch):
    monkeypatch.delenv("MISSING_THING", raising=False)
    result = parse_credential_input(
        "$MISSING_THING", postgres_engine()
    )
    assert result is None


def test_env_ref_with_prose_value_returns_none(monkeypatch):
    monkeypatch.setenv("PROSE_VAR", "just a plain string, not structured")
    result = parse_credential_input(
        "$PROSE_VAR", postgres_engine()
    )
    assert result is None


def test_file_json(tmp_path):
    payload = {
        "host": "db.example.com",
        "database": "analytics",
    }
    target = tmp_path / "creds.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    result = parse_credential_input(str(target), postgres_engine())
    assert result is not None
    assert result.source == "file"
    assert result.fields == payload


def test_file_uri(tmp_path):
    target = tmp_path / "conn.txt"
    target.write_text(
        "postgres://alice:pw@db.example.com/analytics\n",
        encoding="utf-8",
    )
    result = parse_credential_input(str(target), postgres_engine())
    assert result is not None
    assert result.source == "file"
    assert result.fields["host"] == "db.example.com"
    assert result.fields["user"] == "alice"


def test_file_nonexistent_returns_none(tmp_path):
    missing = tmp_path / "does-not-exist.json"
    result = parse_credential_input(str(missing), postgres_engine())
    assert result is None


def test_file_too_large_returns_none(tmp_path):
    huge = tmp_path / "big.json"
    huge.write_bytes(b"x" * (1024 * 1024 + 1))
    result = parse_credential_input(str(huge), postgres_engine())
    assert result is None


def test_empty_input_returns_none():
    assert parse_credential_input("", postgres_engine()) is None
    assert parse_credential_input("   ", postgres_engine()) is None
    none_text: str = None  # type: ignore[assignment]
    assert parse_credential_input(none_text, postgres_engine()) is None


def test_prose_input_returns_none():
    result = parse_credential_input(
        "my host is db.example.com and the password is secret",
        postgres_engine(),
    )
    assert result is None


def test_uri_not_first_token_returns_none():
    result = parse_credential_input(
        "use this: postgres://alice:pw@host/db",
        postgres_engine(),
    )
    assert result is None


def test_env_ref_single_dsn_parses_all_fields(monkeypatch):
    monkeypatch.setenv(
        "PG_DSN",
        "postgres://alice:secret@db.example.com:5432/analytics",
    )
    result = parse_credential_input("$PG_DSN", postgres_engine())
    assert result is not None
    assert result.source == "env"
    assert result.fields == {
        "host": "db.example.com",
        "port": "5432",
        "user": "alice",
        "password": "secret",
        "database": "analytics",
    }


def test_env_ref_single_missing_var_returns_none(monkeypatch):
    monkeypatch.delenv("PG_DSN_ABSENT", raising=False)
    result = parse_credential_input("$PG_DSN_ABSENT", postgres_engine())
    assert result is None


def test_env_ref_multiple_all_present_returns_none(monkeypatch):
    monkeypatch.setenv("PG_HOST", "db.example.com")
    monkeypatch.setenv("PG_PASSWORD", "secret")
    result = parse_credential_input(
        "$PG_HOST $PG_PASSWORD", postgres_engine()
    )
    assert result is None


def test_env_ref_multiple_one_missing_returns_none(monkeypatch):
    monkeypatch.setenv("PG_HOST", "db.example.com")
    monkeypatch.delenv("PG_PASSWORD_ABSENT", raising=False)
    result = parse_credential_input(
        "$PG_HOST $PG_PASSWORD_ABSENT", postgres_engine()
    )
    assert result is None


def test_env_ref_mixed_content_returns_none(monkeypatch):
    monkeypatch.setenv("PG_HOST", "db.example.com")
    result = parse_credential_input(
        "$PG_HOST sometext", postgres_engine()
    )
    assert result is None
