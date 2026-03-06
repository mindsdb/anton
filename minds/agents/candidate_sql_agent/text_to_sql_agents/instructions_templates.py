"""
Streamlined Text-to-SQL prompt templates.

Reduced from ~1050 lines to ~250 lines based on SOTA research showing
that simpler prompts with fewer rules perform better.
"""

QUERY_PLANNING_INSTRUCTIONS = """
You are a query planning assistant. Create a plan to answer the user's question using SQL.

**DATA CATALOG:**
{data_catalog_context}

## Rules
1. **Prefer single FINAL step** when the data catalog has all needed table/column info
2. **Every plan MUST end with exactly one FINAL step** - no exceptions
3. Use EXPLORATORY steps only when you need to discover unknown values or verify schema

## Step Types
- **EXPLORATORY**: Discover unknown info (e.g., what values exist in a status column)
- **FINAL**: Answer the user's question. Has two modes:
  - `final_action: "query"` (default): Execute SQL to get the answer
  - `final_action: "summarize"`: Explain from acquired knowledge (rare)

## Output
For each step, specify:
- description: What this step does
- type: "exploratory" or "final"
- data_catalog_subset: Which datasources and tables are needed
- final_action: "query" or "summarize" (only for FINAL steps)
"""


QUERY_PLANNING_RETRY_INSTRUCTIONS = """
Your previous plan failed validation. Fix it based on the error.

**DATA CATALOG:**
{data_catalog_context}

**PREVIOUS PLAN:**
{failed_query_plan}

**ERROR:**
{error_message}

## Common Fixes
- "Missing FINAL step": Add a FINAL step as the last step
- Plans with only EXPLORATORY steps always fail - you MUST have a FINAL step
"""


SQL_GENERATION_INSTRUCTIONS = """
Generate a SQL query to accomplish the task.

**DATA CATALOG:**
{data_catalog_subset_context}

**ACQUIRED KNOWLEDGE:**
{acquired_knowledge}
"""
# This should be combined either with the MINDSDB_SQL_INSTRUCTIONS or NATIVE_SNOWFLAKE_SQL_INSTRUCTIONS


NATIVE_SNOWFLAKE_SQL_INSTRUCTIONS = """
## Critical Rules
1. **Use ONLY tables from the DATA CATALOG** - never invent table names
2. **Use ONLY columns from the catalog or acquired knowledge** - never guess
3. **All calculations must be in SQL** - use SUM, AVG, COUNT, GROUP BY, etc.

## SNOWFLAKE CASE SENSITIVITY (CRITICAL)
- **Table names**: UPPERCASE without quotes (e.g., ORDERS, PRODUCTS)
- **Column names**: Must be double-quoted to preserve case (e.g., "category", "user_id")
- **Aliases**: Use double quotes for lowercase (e.g., AS "total_count")
- **With table alias**: Use alias."column" (e.g., p."category", oi."sale_price")

## Snowflake SQL Syntax

### Date/Time Functions
- Date truncation: `DATE_TRUNC('month', column)`
- Date math: `DATEADD(day, -1, column)` or `column + INTERVAL '1 month'`
- Current date: `CURRENT_DATE()`
- Extract parts: `YEAR(col)`, `MONTH(col)`, `DAY(col)`
- Epoch to timestamp: `TO_TIMESTAMP(epoch_seconds)` or `TO_TIMESTAMP(epoch_ms / 1000)`

### Window Functions (fully supported)
- `ROW_NUMBER() OVER (PARTITION BY col ORDER BY col2)`
- `LAG(col, 1) OVER (PARTITION BY col ORDER BY col2)`
- `LEAD(col, 1) OVER (ORDER BY col)`
- `SUM(col) OVER (PARTITION BY col2)`
- `QUALIFY ROW_NUMBER() OVER (...) = 1`

### CTEs (fully supported)
```sql
WITH cte1 AS (
    SELECT col1, col2 FROM TABLE_NAME
),
cte2 AS (
    SELECT col1, SUM(col2) as total FROM cte1 GROUP BY col1
)
SELECT * FROM cte2
```

### LATERAL FLATTEN (VARIANT arrays)
- Always access flattened fields via `alias.VALUE:<field>` (not `alias.valuename`)
- Example:
```sql
FROM PUBLICATIONS p,
     LATERAL FLATTEN(input => p."assignee_harmonized") ah
WHERE ah.VALUE:name::STRING = 'DENSO CORP'
```
- For numeric fields:
```sql
COALESCE(
  ep.VALUE:"int_value"::NUMBER,
  ep.VALUE:"float_value"::FLOAT,
  ep.VALUE:"double_value"::FLOAT
)
```

### Aggregations
- Standard: `SUM()`, `AVG()`, `COUNT()`, `MAX()`, `MIN()`
- Count distinct: `COUNT(DISTINCT col)`
- Safe division: `num / NULLIF(denom, 0)`
- Conditional: `SUM(CASE WHEN condition THEN value ELSE 0 END)`
"""


MINDSDB_SQL_INSTRUCTIONS = """
## Critical Rules
1. **Use ONLY tables from the DATA CATALOG** - never invent table names
2. **Use ONLY columns from the catalog or acquired knowledge** - never guess
3. **All calculations must be in SQL** - use SUM, AVG, COUNT, GROUP BY, etc.
4. **CRITICAL: Use the EXACT datasource name from the DATA CATALOG** - e.g., `spider2_lite_thelook_ecommerce`.`TABLE_NAME`
   - NEVER use placeholder names like `datasource`, `ds`, or `mindsdb`
   - Copy the datasource name EXACTLY as shown in the catalog
5. Backticks: Use separate backticks for datasource and table: `actual_datasource_name`.`TABLE_NAME`
6. **IMPORTANT**: Use backticks around ALL column names in WHERE/AND/OR clauses, including aliases
   - Example: `WHERE `row_num` = 1` not `WHERE row_num = 1`

## SQL Syntax (MindsDB - MySQL-compatible, auto-translated to target database)

### Date/Time Functions
- Date truncation: `DATE_TRUNC('month', column)` - column must be DATE/TIMESTAMP type
- Date math: `column - INTERVAL 1 DAY` or `column + INTERVAL 1 MONTH`
- Current date: `CURRENT_DATE`
- Extract parts: `YEAR(col)`, `MONTH(col)`, `DAY(col)`
- Time difference: `DATEDIFF(end_date, start_date)` returns days

### Type Casting
- **Numeric epoch to timestamp**: `TO_TIMESTAMP(numeric_col)` for seconds, `TO_TIMESTAMP(numeric_col / 1000)` for milliseconds
- **Timestamp to date**: `DATE(timestamp_col)` or `CAST(timestamp_col AS DATE)`
- Standard casts: `CAST(col AS INTEGER)`, `CAST(col AS VARCHAR)`
- Check column type in catalog - if type is FIXED/NUMBER, it may be epoch timestamp

### Aggregations
- Standard: `SUM()`, `AVG()`, `COUNT()`, `MAX()`, `MIN()`
- Count distinct: `COUNT(DISTINCT col)`
- Safe division: `num / NULLIF(denom, 0)`
- Conditional: `SUM(CASE WHEN condition THEN value ELSE 0 END)`

### Window Functions
- `ROW_NUMBER() OVER (PARTITION BY col ORDER BY col2)`
- `LAG(col, 1) OVER (ORDER BY col)`
- `SUM(col) OVER (PARTITION BY col2)`

### CTEs
```sql
WITH cte_name AS (
    SELECT ... FROM `your_actual_datasource`.`TABLE_NAME`
)
SELECT ... FROM cte_name
```
Note: Replace `your_actual_datasource` with the EXACT datasource name from the DATA CATALOG.
"""


SQL_GENERATION_RETRY_INSTRUCTIONS = """
Your SQL query failed. Fix it based on the error.

**DATA CATALOG:**
{data_catalog_subset_context}

**ACQUIRED KNOWLEDGE:**
{acquired_knowledge}

**FAILED QUERY:**
{failed_query}

**ERROR:**
{error_message}

**TARGETED FIX:**
{error_guidance}
"""
# This should be combined with the NATIVE_SNOWFLAKE_SQL_ERROR_INSTRUCTIONS
# Ideally, there would also be guidance for MindsDB SQL errors


NATIVE_SNOWFLAKE_SQL_ERROR_INSTRUCTIONS = """
## SNOWFLAKE CASE SENSITIVITY (CRITICAL)
- **Table names**: UPPERCASE without quotes (e.g., ORDERS, PRODUCTS)
- **Column names**: Must be double-quoted to preserve case (e.g., "category", "user_id")
- **Aliases**: Use double quotes for lowercase (e.g., AS "total_count")
- **With table alias**: Use alias."column" (e.g., p."category", oi."sale_price")

## Key Fixes by Error Type

### Invalid identifier errors (e.g., 'P.CATEGORY')
Column names must be double-quoted to preserve lowercase:
```sql
SELECT p."category", SUM(oi."sale_price") AS "revenue"
FROM ORDER_ITEMS oi
JOIN PRODUCTS p ON oi."product_id" = p."id"
```

### Schema Errors (table/column not found)
- Inside native query, use just `TABLE_NAME` (no datasource prefix)
- Double-quote column names to preserve case
- Use EXACT datasource name in the outer wrapper

### Dialect Errors (Snowflake-specific inside native query)
- DATE_TRUNC on numeric: `DATE_TRUNC('month', TO_TIMESTAMP("col"))`
- Epoch to date: `DATE(TO_TIMESTAMP("numeric_col"))`
- Window functions: `LAG("col", 1) OVER (PARTITION BY "x" ORDER BY "y")`
"""

# TODO: Add MindsDB SQL error instructionss


SCHEMA_LINKING_PROMPT = """You are a schema linking expert. Given a natural language question and a database schema, identify the relevant tables and columns needed to answer the question.

## Task
Analyze the question and determine which tables and columns from the schema are required to generate a SQL query.

## Rules
1. Include ALL tables needed for JOINs, even if not directly mentioned in the question
2. Include foreign key columns needed for joining tables
3. Be conservative - missing a required table is worse than including an extra one
4. Only include columns that are actually needed for the query
5. Use fully-qualified table names: <datasource>.<table>

## Question
{question}

## Available Schema
{schema}

## Output Format
Respond with a JSON object containing:
- tables: list of fully-qualified table names (datasource.table)
- columns: dict mapping each table to list of relevant column names
- joins: list of join relationships as [table1, col1, table2, col2]
- reasoning: brief explanation of why these schema elements are needed
"""
