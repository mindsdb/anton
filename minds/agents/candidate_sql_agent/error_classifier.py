"""
SQL Error Classifier based on Shen et al. taxonomy.

Classifies SQL errors into categories to provide targeted retry guidance:
- SCHEMA_LINKING: Wrong table/column names
- SYNTAX: SQL grammar errors
- SEMANTIC: Logic errors (wrong joins, missing conditions)
- VALUE: Wrong literals, formats, types
- DIALECT: Database-specific function issues
- UNKNOWN: Unclassified errors
"""

import re
from dataclasses import dataclass
from enum import Enum


class ErrorCategory(Enum):
    SCHEMA_LINKING = "schema_linking"
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    VALUE = "value"
    DIALECT = "dialect"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedError:
    category: ErrorCategory
    subcategory: str
    guidance: str
    original_error: str


MINDSDB_WRAPPER_PATTERNS = [
    (
        r"(?i)derived query on the external database",
        "external_db_error",
        (
            "External database error (details may be hidden). Check: "
            "1) Table/column names match catalog exactly, "
            "2) Data types are compatible, "
            "3) SQL syntax is valid for target database."
        ),
    ),
    (
        r"(?i)internally generated query derived",
        "internal_query_error",
        "Internal query processing error. Simplify the query structure.",
    ),
]

SCHEMA_LINKING_PATTERNS = [
    (
        r"(?i)object.*does not exist",
        "table_not_found",
        "Table does not exist. Check DATA CATALOG for exact table name with correct datasource prefix.",
    ),
    (r"(?i)table.*not found", "table_not_found", "Table not found. Verify the table name exists in DATA CATALOG."),
    (
        r"(?i)unknown.*integration",
        "wrong_datasource",
        (
            "Wrong datasource name. Use the EXACT datasource name from DATA CATALOG "
            "(e.g., `spider2_lite_thelook_ecommerce`)."
        ),
    ),
    (
        r"invalid identifier '[A-Z]+\.[A-Za-z_]+'",
        "snowflake_case_sensitivity",
        """SNOWFLAKE CASE SENSITIVITY ERROR: Column names must be double-quoted to preserve lowercase.

WRONG: SELECT o.user_id FROM ORDERS o
RIGHT: SELECT o."user_id" FROM ORDERS o

Pattern: alias."column_name" - always double-quote column names when using table aliases.""",
    ),
    (
        r"(?i)invalid identifier",
        "invalid_identifier",
        """Invalid identifier. For Snowflake native queries:
- Table names: UPPERCASE without quotes (ORDERS, PRODUCTS)
- Column names: Double-quoted lowercase (o."user_id", p."category")
- Always prefix columns with table alias: alias."column_name"
Example: SELECT o."user_id", p."category" FROM ORDERS o JOIN PRODUCTS p ON o."product_id" = p."id" """,
    ),
    (
        r"(?i)column.*not found",
        "column_not_found",
        "Column not found. Verify column name from DATA CATALOG or acquired knowledge.",
    ),
    (
        r"(?i)invalid column name",
        "column_not_found",
        "Invalid column name (T-SQL). Copy column names exactly from the schema. "
        "Wrap reserved words or names with spaces in square brackets: [column name].",
    ),
    (
        r"(?i)does not exist.*column",
        "column_not_found",
        "Column does not exist. Check the exact column name in the schema.",
    ),
    (
        r"(?i)ambiguous column",
        "ambiguous_column",
        'Ambiguous column reference. Add table alias prefix: alias."column_name".',
    ),
]

SYNTAX_PATTERNS = [
    (r"(?i)syntax error", "syntax_error", "SQL syntax error. Check for missing keywords, parentheses, or commas."),
    (
        r"(?i)unexpected.*token",
        "unexpected_token",
        "Unexpected token in SQL. Review SQL syntax near the error location.",
    ),
    (
        r"(?i)missing.*keyword",
        "missing_keyword",
        "Missing SQL keyword. Check for required keywords like SELECT, FROM, WHERE.",
    ),
    (r"(?i)parse error", "parse_error", "SQL parse error. Verify SQL structure and keyword placement."),
    (r"(?i)unparseable sql", "parse_error", "Unparseable SQL. Check for syntax issues and proper keyword usage."),
    (
        r"(?i)double quotes.*not allowed",
        "quoting_error",
        'Use backticks for identifiers, not double quotes. Example: `column_name` not "column_name".',
    ),
]

SEMANTIC_PATTERNS = [
    (
        r"(?i)missing.*group by",
        "missing_group_by",
        "Missing GROUP BY clause. Add GROUP BY for non-aggregated columns in SELECT.",
    ),
    (
        r"(?i)not in group by",
        "group_by_mismatch",
        "Column not in GROUP BY. Add the column to GROUP BY or use an aggregate function.",
    ),
    (r"(?i)aggregate.*without group", "aggregate_error", "Aggregate function without GROUP BY. Add GROUP BY clause."),
    (
        r"(?i)join.*condition",
        "join_error",
        "Missing or incorrect JOIN condition. Verify ON clause references correct columns.",
    ),
    (
        r"(?i)cartesian product",
        "missing_join",
        "Cartesian product detected. Add proper JOIN conditions between tables.",
    ),
]

VALUE_PATTERNS = [
    (r"(?i)type mismatch", "type_mismatch", "Type mismatch. Use CAST() to convert between types."),
    (r"(?i)cannot cast", "cast_error", "Cannot cast value. Check source type and use appropriate conversion function."),
    (r"(?i)invalid.*value", "invalid_value", "Invalid value. Check literal format matches expected data type."),
    (r"(?i)numeric value.*out of range", "overflow", "Numeric overflow. Value exceeds column type limits."),
]

DIALECT_PATTERNS = [
    # MS SQL (T-SQL) specific patterns
    (
        r"(?i)incorrect syntax near.*limit",
        "mssql_no_limit",
        "T-SQL does not support LIMIT. Use SELECT TOP n or OFFSET n ROWS FETCH NEXT m ROWS ONLY (requires ORDER BY).",
    ),
    (
        r"(?i)the order by clause is invalid.*unless.*top.*offset",
        "mssql_order_in_subquery",
        """T-SQL requires TOP or OFFSET/FETCH when using ORDER BY in a subquery.

FIX: Use a CTE and apply ORDER BY in the outermost SELECT:
  WITH ordered AS (SELECT * FROM table_name)
  SELECT * FROM ordered ORDER BY col""",
    ),
    (
        r"(?i)conversion failed when converting",
        "mssql_type_conversion",
        "T-SQL type conversion failed. Use TRY_CAST() or TRY_CONVERT() for unsafe conversions,"
        " or CAST(col AS target_type).",
    ),
    (
        r"(?i)arithmetic overflow error converting",
        "mssql_arithmetic_overflow",
        "Arithmetic overflow in T-SQL. Use CAST to a larger type such as BIGINT or DECIMAL(18,2).",
    ),
    (
        r"(?i)dateadd.*is not a recognized.*function name",
        "mssql_invalid_date_func",
        "Check DATEADD syntax: DATEADD(datepart, number, date)."
        " Valid dateparts: year, month, day, hour, minute, second.",
    ),
    (
        r"(?i)date_trunc.*does not support.*number",
        "epoch_to_date",
        """DATE_TRUNC requires TIMESTAMP, not NUMBER. The column stores Unix epoch (seconds since 1970).

FIX: Wrap the column with TO_TIMESTAMP() before DATE_TRUNC:
  WRONG: DATE_TRUNC('month', "created_at")
  RIGHT: DATE_TRUNC('month', TO_TIMESTAMP("created_at"))

For date comparisons with epoch columns:
  WHERE TO_TIMESTAMP("created_at") >= '2020-01-01'""",
    ),
    (
        r"(?i)invalid type.*for.*date",
        "epoch_to_date",
        "Column is numeric epoch, not date. Use TO_TIMESTAMP(col) first.",
    ),
    (
        r"(?i)can not convert.*timestamp.*into.*number",
        "epoch_comparison",
        """Cannot compare TIMESTAMP with NUMBER column. The column stores Unix epoch (seconds).

FIX: Convert the date to epoch for comparison, or convert the column to timestamp:
  WRONG: WHERE "created_at" >= '2022-01-01'
  RIGHT: WHERE TO_TIMESTAMP("created_at") >= '2022-01-01'
  OR:    WHERE "created_at" >= DATE_PART('epoch', '2022-01-01'::TIMESTAMP)""",
    ),
    (
        r"(?i)numeric value.*is not recognized",
        "epoch_comparison",
        """Cannot use date string with numeric epoch column. The column stores Unix epoch (seconds).

FIX: Convert the column to timestamp before comparing:
  WRONG: WHERE "delivered_at" >= '2022-01-01'
  RIGHT: WHERE TO_TIMESTAMP("delivered_at") >= '2022-01-01'""",
    ),
    (
        r"(?i)unknown function.*from_unixtime",
        "function_not_supported",
        "FROM_UNIXTIME not supported. Use TO_TIMESTAMP(col) instead.",
    ),
    (
        r"(?i)log10.*not supported",
        "function_not_supported",
        "LOG10 not supported in Snowflake. Use LN(x)/LN(10) for base-10 log.",
    ),
    (
        r"(?i)function.*does not exist",
        "function_not_supported",
        "Function not available in this database. Check for database-specific alternatives.",
    ),
    (
        r"(?i)invalid.*timestamp",
        "timestamp_error",
        "Invalid timestamp format. Use standard format or TO_TIMESTAMP() for conversion.",
    ),
]


def extract_nested_error(error_message: str) -> str | None:
    """
    Extract nested error details from MindsDB wrapper errors.

    MindsDB wraps external database errors. This extracts the actual error if present.
    """
    import re

    match = re.search(r"Error:\s*\n?\s*(.+?)(?:\n\nFailed Query:|$)", error_message, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def classify_error(error_message: str) -> ClassifiedError:
    """
    Classify an SQL error into a category with targeted guidance.

    Args:
        error_message: The error message from SQL execution

    Returns:
        ClassifiedError with category, subcategory, and guidance
    """
    nested_error = extract_nested_error(error_message)
    if nested_error:
        nested_result = _classify_error_internal(nested_error)
        if nested_result.category != ErrorCategory.UNKNOWN:
            return nested_result

    return _classify_error_internal(error_message)


def _classify_error_internal(error_message: str) -> ClassifiedError:
    """Internal classification logic."""
    for pattern, subcategory, guidance in MINDSDB_WRAPPER_PATTERNS:
        if re.search(pattern, error_message):
            return ClassifiedError(
                category=ErrorCategory.UNKNOWN,
                subcategory=subcategory,
                guidance=guidance,
                original_error=error_message,
            )

    for pattern, subcategory, guidance in SCHEMA_LINKING_PATTERNS:
        if re.search(pattern, error_message):
            return ClassifiedError(
                category=ErrorCategory.SCHEMA_LINKING,
                subcategory=subcategory,
                guidance=guidance,
                original_error=error_message,
            )

    for pattern, subcategory, guidance in DIALECT_PATTERNS:
        if re.search(pattern, error_message):
            return ClassifiedError(
                category=ErrorCategory.DIALECT,
                subcategory=subcategory,
                guidance=guidance,
                original_error=error_message,
            )

    for pattern, subcategory, guidance in VALUE_PATTERNS:
        if re.search(pattern, error_message):
            return ClassifiedError(
                category=ErrorCategory.VALUE,
                subcategory=subcategory,
                guidance=guidance,
                original_error=error_message,
            )

    for pattern, subcategory, guidance in SEMANTIC_PATTERNS:
        if re.search(pattern, error_message):
            return ClassifiedError(
                category=ErrorCategory.SEMANTIC,
                subcategory=subcategory,
                guidance=guidance,
                original_error=error_message,
            )

    for pattern, subcategory, guidance in SYNTAX_PATTERNS:
        if re.search(pattern, error_message):
            return ClassifiedError(
                category=ErrorCategory.SYNTAX,
                subcategory=subcategory,
                guidance=guidance,
                original_error=error_message,
            )

    return ClassifiedError(
        category=ErrorCategory.UNKNOWN,
        subcategory="unclassified",
        guidance="Review the error message and check SQL syntax, table/column names, and data types.",
        original_error=error_message,
    )


def format_error_guidance(classified_error: ClassifiedError) -> str:
    """
    Format classified error into guidance text for retry prompt.

    Args:
        classified_error: The classified error

    Returns:
        Formatted guidance string
    """
    category_labels = {
        ErrorCategory.SCHEMA_LINKING: "SCHEMA ERROR",
        ErrorCategory.SYNTAX: "SYNTAX ERROR",
        ErrorCategory.SEMANTIC: "LOGIC ERROR",
        ErrorCategory.VALUE: "VALUE/TYPE ERROR",
        ErrorCategory.DIALECT: "DATABASE DIALECT ERROR",
        ErrorCategory.UNKNOWN: "ERROR",
    }

    label = category_labels.get(classified_error.category, "ERROR")

    return f"""**{label} ({classified_error.subcategory})**
{classified_error.guidance}"""
