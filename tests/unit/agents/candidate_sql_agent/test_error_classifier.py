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


class TestMSSQLErrorPatterns:
    def test_classifies_mssql_no_limit_error(self):
        c = classify_error("Incorrect syntax near 'LIMIT'.")
        assert c.category == ErrorCategory.DIALECT
        assert c.subcategory == "mssql_no_limit"
        assert "TOP" in c.guidance

    def test_classifies_mssql_order_by_in_subquery(self):
        msg = (
            "The ORDER BY clause is invalid in views, inline functions,"
            " derived tables, unless TOP or OFFSET is also specified."
        )
        c = classify_error(msg)
        assert c.category == ErrorCategory.DIALECT
        assert c.subcategory == "mssql_order_in_subquery"
        assert "CTE" in c.guidance

    def test_classifies_mssql_type_conversion(self):
        c = classify_error("Conversion failed when converting the nvarchar value 'abc' to data type int.")
        assert c.category == ErrorCategory.DIALECT
        assert c.subcategory == "mssql_type_conversion"
        assert "TRY_CAST" in c.guidance

    def test_classifies_mssql_arithmetic_overflow(self):
        c = classify_error("Arithmetic overflow error converting expression to data type int.")
        assert c.category == ErrorCategory.DIALECT
        assert c.subcategory == "mssql_arithmetic_overflow"
        assert "BIGINT" in c.guidance

    def test_classifies_mssql_invalid_column_name_with_bracket_hint(self):
        c = classify_error("Invalid column name 'Order Date'.")
        assert c.category == ErrorCategory.SCHEMA_LINKING
        assert c.subcategory == "column_not_found"
        assert "[column name]" in c.guidance.lower() or "brackets" in c.guidance.lower()

    def test_nested_mssql_error_extracted_and_classified(self):
        msg = (
            "Derived query on the external database.\n"
            "Error:\nConversion failed when converting the varchar value 'x' to int\n\n"
            "Failed Query:\nSELECT 1"
        )
        c = classify_error(msg)
        assert c.category == ErrorCategory.DIALECT
        assert c.subcategory == "mssql_type_conversion"
