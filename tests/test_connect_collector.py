from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.connect_collector import (
    ConnectionCollector,
    ExtractedData,
    _ExtractionResult,
    extract_variables,
)
from anton.core.datasources.datasource_registry import (
    AuthMethod,
    DatasourceEngine,
    DatasourceField,
)


def _postgres_engine() -> DatasourceEngine:
    return DatasourceEngine(
        engine="postgres",
        display_name="PostgreSQL",
        fields=[
            DatasourceField(name="host", required=True, description="hostname"),
            DatasourceField(
                name="port", required=True, default="5432", description="port"
            ),
            DatasourceField(name="database", required=True, description="db name"),
            DatasourceField(name="user", required=True, description="username"),
            DatasourceField(
                name="password", required=True, secret=True, description="pwd"
            ),
            DatasourceField(name="schema", required=False, description="schema"),
        ],
    )


def _hubspot_choice_engine() -> DatasourceEngine:
    return DatasourceEngine(
        engine="hubspot",
        display_name="HubSpot",
        auth_method="choice",
        auth_methods=[
            AuthMethod(
                name="pat",
                display="Personal Access Token",
                fields=[
                    DatasourceField(name="access_token", required=True, secret=True)
                ],
            ),
            AuthMethod(
                name="oauth",
                display="OAuth2",
                fields=[
                    DatasourceField(name="client_id", required=True),
                    DatasourceField(
                        name="client_secret", required=True, secret=True
                    ),
                ],
            ),
        ],
    )


def _mock_session_with_extraction(
    *,
    variables: dict[str, str] | None = None,
    is_redirect: bool = False,
    redirect_engine: str = "",
    redirect_reason: str = "",
) -> MagicMock:
    """Build a session whose `_llm.generate_object` returns a known result."""
    extraction = _ExtractionResult(
        variables=variables or {},
        is_redirect=is_redirect,
        redirect_engine=redirect_engine,
        redirect_reason=redirect_reason,
    )
    session = MagicMock()
    session._llm = MagicMock()
    session._llm.generate_object = AsyncMock(return_value=extraction)
    return session


def _mock_session_raising(exc: Exception) -> MagicMock:
    """Build a session whose `_llm.generate_object` raises the given exception."""
    session = MagicMock()
    session._llm = MagicMock()
    session._llm.generate_object = AsyncMock(side_effect=exc)
    return session


# ─────────────────────────────────────────────────────────────────────────────
# ConnectionCollector
# ─────────────────────────────────────────────────────────────────────────────


class TestConnectionCollector:
    def test_initial_state_all_required_missing(self):
        c = ConnectionCollector(_postgres_engine())
        assert not c.is_complete
        assert len(c.missing_required) == 5
        assert c.next_field is not None
        assert c.next_field.name == "host"

    def test_fill_simple(self):
        c = ConnectionCollector(_postgres_engine())
        assert c.fill("host", "localhost") is True
        assert c.collected["host"] == "localhost"
        assert len(c.missing_required) == 4
        assert c.next_field is not None
        assert c.next_field.name == "port"

    def test_fill_unknown_field_rejected(self):
        c = ConnectionCollector(_postgres_engine())
        assert c.fill("made_up_field", "x") is False
        assert "made_up_field" not in c.collected

    def test_fill_empty_value_does_nothing(self):
        c = ConnectionCollector(_postgres_engine())
        assert c.fill("host", "") is True  # accepted (known field)
        assert "host" not in c.collected  # but not stored

    def test_fill_many(self):
        c = ConnectionCollector(_postgres_engine())
        accepted = c.fill_many(
            {
                "host": "db.x",
                "port": "5432",
                "user": "admin",
                "password": "secret",
                "database": "mydb",
                "unknown_garbage": "ignored",
            }
        )
        assert set(accepted) == {"host", "port", "user", "password", "database"}
        assert "unknown_garbage" not in c.collected
        assert c.is_complete

    def test_complete_when_all_required_filled(self):
        c = ConnectionCollector(_postgres_engine())
        c.fill_many(
            {
                "host": "x",
                "port": "5432",
                "database": "d",
                "user": "u",
                "password": "p",
            }
        )
        assert c.is_complete
        assert c.missing_required == []

    def test_optional_does_not_block_completion(self):
        c = ConnectionCollector(_postgres_engine())
        c.fill_many(
            {
                "host": "x",
                "port": "5432",
                "database": "d",
                "user": "u",
                "password": "p",
            }
        )
        assert c.is_complete
        # schema is optional and unfilled
        assert any(f.name == "schema" for f in c.missing_optional)

    def test_next_field_falls_back_to_optional_when_required_done(self):
        c = ConnectionCollector(_postgres_engine())
        c.fill_many(
            {"host": "x", "port": "5", "database": "d", "user": "u", "password": "p"}
        )
        # All required filled — next_field is the first optional
        assert c.next_field is not None
        assert c.next_field.name == "schema"

    def test_auth_method_active_fields(self):
        engine = _hubspot_choice_engine()
        pat = engine.auth_methods[0]
        c = ConnectionCollector(engine, auth_method=pat)
        # PAT method has only access_token, not client_id/client_secret
        assert {f.name for f in c.active_fields} == {"access_token"}
        c.fill("access_token", "abc123")
        assert c.is_complete

    def test_to_redirect_result(self):
        c = ConnectionCollector(_postgres_engine())
        c.fill_many({"host": "db.x", "user": "admin"})
        c.redirect_message = "actually it's mysql"
        result = c.to_redirect_result()
        assert result["status"] == "redirect"
        assert result["engine"] == "postgres"
        assert result["engine_display"] == "PostgreSQL"
        assert result["collected_variables"] == {"host": "db.x", "user": "admin"}
        assert "port" in result["missing_required"]
        assert "password" in result["missing_required"]
        assert result["redirect_message"] == "actually it's mysql"


# ─────────────────────────────────────────────────────────────────────────────
# extract_variables (LLM-driven)
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractVariables:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        engine = _postgres_engine()
        session = _mock_session_with_extraction()
        result = await extract_variables(
            "",
            expected_fields=engine.fields,
            current_engine="postgres",
            current_engine_display="PostgreSQL",
            known_engine_slugs=["postgres", "mysql"],
            session=session,
        )
        assert result.variables == {}
        assert not result.is_redirect
        # Empty input shouldn't even call the LLM
        session._llm.generate_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_extracts_variables_from_bulk_input(self):
        engine = _postgres_engine()
        session = _mock_session_with_extraction(
            variables={
                "host": "db.example.com",
                "port": "5432",
                "user": "admin",
            },
        )
        result = await extract_variables(
            "host=db.example.com port=5432 user=admin",
            expected_fields=engine.fields,
            current_engine="postgres",
            current_engine_display="PostgreSQL",
            known_engine_slugs=["postgres", "mysql"],
            session=session,
        )
        assert result.variables == {
            "host": "db.example.com",
            "port": "5432",
            "user": "admin",
        }
        assert not result.is_redirect

    @pytest.mark.asyncio
    async def test_passes_extraction_result_schema_to_llm(self):
        """Verify generate_object was called with the right schema class."""
        engine = _postgres_engine()
        session = _mock_session_with_extraction(variables={"host": "db.x"})
        await extract_variables(
            "host=db.x",
            expected_fields=engine.fields,
            current_engine="postgres",
            current_engine_display="PostgreSQL",
            known_engine_slugs=["postgres"],
            session=session,
        )
        session._llm.generate_object.assert_called_once()
        call_args = session._llm.generate_object.call_args
        assert call_args.args[0] is _ExtractionResult

    @pytest.mark.asyncio
    async def test_llm_parses_connection_string(self):
        engine = _postgres_engine()
        session = _mock_session_with_extraction(
            variables={
                "host": "db.example.com",
                "port": "5432",
                "user": "admin",
                "password": "secret",
                "database": "mydb",
            },
        )
        result = await extract_variables(
            "postgres://admin:secret@db.example.com:5432/mydb",
            expected_fields=engine.fields,
            current_engine="postgres",
            current_engine_display="PostgreSQL",
            known_engine_slugs=["postgres"],
            session=session,
        )
        assert result.variables["host"] == "db.example.com"
        assert result.variables["user"] == "admin"
        assert result.variables["password"] == "secret"
        assert result.variables["database"] == "mydb"

    @pytest.mark.asyncio
    async def test_llm_resolves_aliases(self):
        engine = _postgres_engine()
        session = _mock_session_with_extraction(
            variables={
                "host": "db.x",
                "user": "admin",
                "password": "secret",
            },
        )
        result = await extract_variables(
            "hostname=db.x username=admin pwd=secret",
            expected_fields=engine.fields,
            current_engine="postgres",
            current_engine_display="PostgreSQL",
            known_engine_slugs=["postgres"],
            session=session,
        )
        assert result.variables == {
            "host": "db.x",
            "user": "admin",
            "password": "secret",
        }

    @pytest.mark.asyncio
    async def test_llm_detects_redirect(self):
        engine = _postgres_engine()
        session = _mock_session_with_extraction(
            variables={},
            is_redirect=True,
            redirect_engine="mysql",
            redirect_reason="user wants mysql instead",
        )
        result = await extract_variables(
            "actually let's use mysql instead",
            expected_fields=engine.fields,
            current_engine="postgres",
            current_engine_display="PostgreSQL",
            known_engine_slugs=["postgres", "mysql"],
            session=session,
        )
        assert result.is_redirect
        assert result.redirect_engine == "mysql"
        assert "mysql" in result.redirect_reason

    @pytest.mark.asyncio
    async def test_llm_ignores_fields_not_in_expected_list(self):
        engine = _postgres_engine()
        session = _mock_session_with_extraction(
            variables={
                "host": "db.x",
                "bogus_field": "should be dropped",
            },
        )
        result = await extract_variables(
            "host=db.x bogus=y",
            expected_fields=engine.fields,
            current_engine="postgres",
            current_engine_display="PostgreSQL",
            known_engine_slugs=["postgres"],
            session=session,
        )
        assert result.variables == {"host": "db.x"}
        assert "bogus_field" not in result.variables

    @pytest.mark.asyncio
    async def test_llm_exception_returns_empty_result(self):
        engine = _postgres_engine()
        session = _mock_session_raising(RuntimeError("network error"))
        result = await extract_variables(
            "host=db.x",
            expected_fields=engine.fields,
            current_engine="postgres",
            current_engine_display="PostgreSQL",
            known_engine_slugs=["postgres"],
            session=session,
        )
        # Exception is caught → empty result, never crashes the flow
        assert result.variables == {}
        assert not result.is_redirect

    @pytest.mark.asyncio
    async def test_validation_error_returns_empty_result(self):
        """If generate_object raises a Pydantic ValidationError (rare with
        forced tool_choice, but possible), we fall back to empty result."""
        from pydantic import ValidationError as _PVE

        engine = _postgres_engine()
        try:
            _ExtractionResult.model_validate({"variables": "not a dict"})
        except _PVE as exc:
            session = _mock_session_raising(exc)
        result = await extract_variables(
            "anything",
            expected_fields=engine.fields,
            current_engine="postgres",
            current_engine_display="PostgreSQL",
            known_engine_slugs=["postgres"],
            session=session,
        )
        assert result.variables == {}
        assert not result.is_redirect


# ─────────────────────────────────────────────────────────────────────────────
# ExtractedData dataclass sanity
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractedData:
    def test_default_values(self):
        e = ExtractedData()
        assert e.variables == {}
        assert not e.is_redirect
        assert e.redirect_engine is None
        assert e.redirect_reason == ""
