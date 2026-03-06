"""
Unit tests for candidate SQL agent error classification helpers.
"""

from minds.agents.candidate_sql_agent.error_classifier import (
    ErrorCategory,
    classify_error,
    extract_nested_error,
    format_error_guidance,
)


class TestExtractNestedError:
    def test_extracts_nested_error_before_failed_query(self):
        msg = "Something\nError:\n  Table not found\n\nFailed Query:\nSELECT 1"
        assert extract_nested_error(msg) == "Table not found"

    def test_returns_none_when_no_nested_error(self):
        assert extract_nested_error("no wrapper here") is None


class TestClassifyError:
    def test_classifies_schema_linking_table_not_found(self):
        c = classify_error("table not found: users")
        assert c.category == ErrorCategory.SCHEMA_LINKING
        assert c.subcategory == "table_not_found"

    def test_classifies_dialect_unknown_function(self):
        c = classify_error("Unknown function FROM_UNIXTIME")
        assert c.category == ErrorCategory.DIALECT

    def test_nested_error_is_preferred_when_classified(self):
        msg = "Derived query on the external database.\nError:\ninvalid identifier\n\nFailed Query:\nSELECT 1"
        c = classify_error(msg)
        # Should classify the nested error, not the wrapper pattern.
        assert c.category == ErrorCategory.SCHEMA_LINKING


class TestFormatErrorGuidance:
    def test_formats_with_category_label(self):
        c = classify_error("syntax error near FROM")
        out = format_error_guidance(c)
        assert out.startswith("**SYNTAX ERROR")
        assert c.guidance in out
