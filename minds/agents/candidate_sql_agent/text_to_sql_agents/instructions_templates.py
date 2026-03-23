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
4. **Use the EXACT datasource names from the DATA CATALOG** in `data_catalog_subset`.
   - Hints like `GA4`, `GA360`, `FDA`, etc. are NOT datasource names.
   - Never invent or abbreviate datasource names.

## Step Types
- **EXPLORATORY**: Discover unknown info (e.g., what values exist in a status column, or querying `INFORMATION_SCHEMA` to find table/column names when the catalog is incomplete)
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

NEVER use UNION ALL to combine date-sharded tables (e.g., EVENTS_YYYYMMDD). Query the unified parent view or use the LIKE / REGEXP table matching syntax supported by your Snowflake configuration.
"""

NATIVE_BIGQUERY_SQL_INSTRUCTIONS = """
You are generating **BigQuery Standard SQL ONLY**.

## HARD CONSTRAINTS (NEVER VIOLATE)
1. Use **ONLY tables present in the DATA CATALOG**. Never invent table names.
2. Use **ONLY columns present in the catalog or provided external knowledge**. Never guess.
3. Perform **ALL calculations in SQL** (SUM, COUNT, AVG, GROUP BY, etc.).
4. Output **ONE complete, executable BigQuery SQL query** and nothing else.
   - The query MUST begin with `SELECT` or `WITH`. Never output a bare expression (e.g., `DATE_DIFF(...)` alone is not a valid query).
   - The query MUST include `FROM` referencing actual data tables — **never query `INFORMATION_SCHEMA`** in the final answer.
5. Use **BigQuery Standard SQL** (Legacy SQL is NOT allowed).
6. **Never output placeholders** (no `...`, no "Repeat for", no TODOs).
7. **Never output comments** (`--` or `/* */`) in the final SQL.

---

## IDENTIFIERS & TABLE REFERENCES
- Use **backticks (`)** for all identifiers.
- Use the **exact table identifier as provided by the catalog**:
  - If catalog provides `project.dataset.table` → use it fully qualified.
  - If catalog provides `dataset.table` → use that form.
  - Do NOT invent project or dataset names.
- Do NOT use double quotes (`"`).

---

## CASTING & TYPES
- Use `CAST(col AS INT64 | FLOAT64 | NUMERIC | STRING | DATE | TIMESTAMP)`
- Use `SAFE_CAST(...)` when type may be invalid or inconsistent.
- Never use `::type` casting.

---

## NULL & ERROR SAFETY
- Use `SAFE_DIVIDE(a, b)` for ratios.
- Use `IFNULL(x, y)` or `COALESCE(x, y)` for null handling.
- Use `COUNTIF(condition)` for conditional counts.

---

## DATE & TIME (BIGQUERY STYLE)
- Truncate dates: `DATE_TRUNC(date_col, MONTH)`
- Truncate timestamps: `TIMESTAMP_TRUNC(ts_col, MONTH)`
- Date arithmetic: `DATE_ADD(date_col, INTERVAL 7 DAY)`
- Extract parts: `EXTRACT(YEAR FROM date_col)`
- Parse strings:
  - `PARSE_DATE('%Y%m%d', str_col)`
  - `PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', str_col)`

❌ Never use `DATEADD`, `TO_DATE`, `DATEDIFF`, or Postgres/Snowflake date functions.

---

## STRINGS & FILTERING
- Use `CONCAT(a, b)` for string concatenation.
- Case-insensitive matching:
  - `LOWER(col) LIKE LOWER('%value%')`
  - or `REGEXP_CONTAINS(col, r'(?i)value')`
- Do NOT use `ILIKE`.

---

## AGGREGATION & GROUPING
- Use explicit column names in `GROUP BY` (avoid positional `GROUP BY 1`).
- All non-aggregated selected columns MUST appear in `GROUP BY`.

---

## JOINS
- Use explicit `JOIN ... ON ...` clauses.
- Never use implicit joins (comma joins).
- Join only on columns that exist in the catalog.

---

## ARRAYS & UNNEST (CRITICAL FOR BIGQUERY)
- If a column is an ARRAY or REPEATED field:
  - Use `CROSS JOIN UNNEST(array_col) AS alias`
- For arrays of STRUCTs:
  - `CROSS JOIN UNNEST(t.array_col) AS a`
  - Access fields as `a.field_name`
- Do NOT invent join tables to flatten arrays.

---

## JSON HANDLING
- If JSON column type is **STRING**:
  - Use `JSON_EXTRACT_SCALAR(json_str, '$.path')`
- If JSON column type is **JSON**:
  - Use `JSON_VALUE(json_col, '$.path')`
  - Use `JSON_QUERY(json_col, '$.path')` for objects/arrays

---

## WILDCARD (DATE-SHARDED) TABLES
BigQuery supports date-sharded table families (e.g., `events_20240101`, `events_20240102`).
- Query them with a **wildcard table** and filter via `_TABLE_SUFFIX`:
  ```sql
  SELECT * FROM `project.dataset.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20240101' AND '20240131'
  ```
- **Never** enumerate individual shards with UNION ALL — use the wildcard instead.
- `_TABLE_SUFFIX` is always a STRING; use string comparison (e.g., `>= '20240101'`).

---

## DIALECT BAN LIST (NEVER USE)
❌ `ILIKE`
❌ `TOP`
❌ `NVL`
❌ `DATEDIFF`
❌ `DATEADD`
❌ `TO_DATE`
❌ `::type`
❌ Double-quoted identifiers
❌ Legacy SQL syntax
❌ Placeholder tokens (`...`)
❌ SQL comments (`--`, `/* */`)
❌ `UNION ALL` to enumerate date-sharded table suffixes (use wildcard `*` + `_TABLE_SUFFIX`)

---

## QUERY CONSTRUCTION STRATEGY
1. Identify required tables from the catalog.
2. Start with `FROM` using catalog table identifiers.
3. Add necessary `JOIN`s with explicit keys.
4. Apply `WHERE` filters.
5. Apply aggregations and `GROUP BY`.
6. Add `ORDER BY` if required.
7. Add `LIMIT` if appropriate.

---

Return ONLY the final BigQuery SQL query.
"""


NATIVE_MSSQL_SQL_INSTRUCTIONS = """
You are generating **Microsoft SQL Server T-SQL ONLY**.

## HARD CONSTRAINTS (NEVER VIOLATE)
1. Use **ONLY tables present in the DATA CATALOG**. Never invent table names.
2. **ALWAYS use schema-qualified table names**: the catalog lists tables as `datasource.SCHEMA.TABLE_NAME` — in SQL write `SCHEMA.TABLE_NAME`. Example: catalog `mydb.dbo.ORDERS` → write `dbo.ORDERS`. ❌ NEVER write just `ORDERS`.
3. Use **ONLY columns present in the catalog or provided external knowledge**. Never guess.
4. Perform **ALL calculations in SQL** (SUM, COUNT, AVG, GROUP BY, etc.).
5. Output **ONE executable T-SQL query** and nothing else.
6. **Never output placeholders** (no `...`, no "Repeat for", no TODOs).
7. **Never output comments** (`--` or `/* */`) in the final SQL.

---

## IDENTIFIERS & QUOTING
- Identifiers are **case-insensitive** by default — use exact casing from the catalog.
- Wrap reserved words or names with spaces in square brackets: `[Order Date]`, `[select]`.
- Do **not** use backticks or double quotes for identifiers.

---

## ROW LIMITING & PAGINATION
- To limit results: `SELECT TOP 100 * FROM table_name ORDER BY col`
- For offset pagination (requires ORDER BY):
  ```sql
  SELECT * FROM table_name
  ORDER BY col
  OFFSET 10 ROWS FETCH NEXT 20 ROWS ONLY
  ```
- **Never use `LIMIT` or `LIMIT ... OFFSET`** — these are not T-SQL.

---

## DATE & TIME
- Current date/time: `GETDATE()`, `GETUTCDATE()`
- Current date only: `CAST(GETDATE() AS DATE)`
- Date arithmetic: `DATEADD(day, -30, GETDATE())`, `DATEADD(month, 1, col)`
- Date difference: `DATEDIFF(day, start_date, end_date)`
- Truncate to month: `DATEFROMPARTS(YEAR(col), MONTH(col), 1)`
- Parse string to date: `CAST('2024-01-15' AS DATE)` or `CONVERT(DATE, '2024-01-15', 23)`
- Unix epoch to datetime: `DATEADD(second, epoch_col, CAST('1970-01-01' AS DATETIME))`
- Extract parts: `YEAR(col)`, `MONTH(col)`, `DAY(col)`, `DATEPART(hour, col)`

❌ Never use `DATE_TRUNC`, `NOW()`, `TO_DATE`, `INTERVAL`, `EXTRACT(... FROM ...)` — these are not T-SQL.

---

## TYPE CASTING
- `CAST(col AS INT)` / `CAST(col AS BIGINT)`
- `CAST(col AS VARCHAR(255))` / `CAST(col AS NVARCHAR(MAX))`
- `CAST(col AS DECIMAL(18, 2))` / `CAST(col AS FLOAT)`
- `CAST(col AS DATE)` / `CAST(col AS DATETIME)` / `CAST(col AS DATETIME2)`
- `CAST(col AS BIT)` for boolean

---

## NULL & SAFE OPERATIONS
- Null coalescing: `ISNULL(col, default_value)` or `COALESCE(col1, col2, default_value)`
- Safe division: `col / NULLIF(denom, 0)` or `CASE WHEN denom = 0 THEN NULL ELSE col / denom END`

---

## STRING OPERATIONS
- Concatenation: `col1 + ' ' + col2` or `CONCAT(col1, col2)` (prefer CONCAT for null safety)
- Length: `LEN(col)` (not `LENGTH`)
- Substring: `SUBSTRING(col, start, length)` (1-based index)
- Find position: `CHARINDEX(substring, col)` (not `INSTR`)
- Case-insensitive search: `WHERE LOWER(col) LIKE LOWER('%value%')`
- Pattern matching: `WHERE col LIKE '%pattern%'`

❌ Never use `ILIKE` — use `LIKE` with `LOWER()` instead.
❌ Never use `||` for concatenation — use `+` or `CONCAT()`.

---

## AGGREGATIONS & GROUPING
- Standard: `SUM()`, `AVG()`, `COUNT()`, `MAX()`, `MIN()`
- Count distinct: `COUNT(DISTINCT col)`
- Conditional count: `SUM(CASE WHEN condition THEN 1 ELSE 0 END)`
- Use explicit column names in `GROUP BY` (not positional numbers).
- All non-aggregated SELECT columns must appear in `GROUP BY`.

---

## WINDOW FUNCTIONS (fully supported)
- `ROW_NUMBER() OVER (ORDER BY col)`
- `ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY order_col)`
- `SUM(col) OVER (PARTITION BY group_col ORDER BY order_col ROWS UNBOUNDED PRECEDING)`
- `LAG(col, 1) OVER (ORDER BY col)`
- `LEAD(col, 1) OVER (ORDER BY col)`
- `RANK() OVER (ORDER BY col DESC)`

---

## CTEs (fully supported)
```sql
WITH cte_name AS (
    SELECT col1, col2 FROM table_name WHERE condition
),
cte2 AS (
    SELECT col1, SUM(col2) AS total FROM cte_name GROUP BY col1
)
SELECT * FROM cte2
```

---

## JSON HANDLING
- Extract scalar value: `JSON_VALUE(json_col, '$.path')`
- Extract object/array: `JSON_QUERY(json_col, '$.path')`
- Check if valid JSON: `ISJSON(col) = 1`

---

## JOINS
- Use explicit `JOIN ... ON ...` clauses.
- Never use implicit comma joins.
- Join only on columns that exist in the catalog.

---

## DIALECT BAN LIST (NEVER USE)
❌ `LIMIT` / `LIMIT ... OFFSET` (use `TOP` or `OFFSET...FETCH`)
❌ `ILIKE` (use `LIKE` with `LOWER()`)
❌ `::type` casting (use `CAST()`)
❌ `NOW()` (use `GETDATE()`)
❌ `DATE_TRUNC` (use `DATEFROMPARTS` / `CAST`)
❌ `INTERVAL '...'` syntax
❌ `EXTRACT(... FROM ...)` (use `YEAR()`, `MONTH()`, `DAY()`, `DATEPART()`)
❌ `QUALIFY` (Snowflake-only)
❌ Backticks for identifiers (use `[brackets]` only when needed)
❌ `||` string concatenation (use `+` or `CONCAT()`)
❌ Legacy SQL syntax
❌ Placeholder tokens or SQL comments in output

---

## TABLE REFERENCES
- The data catalog lists tables as `datasource.SCHEMA.TABLE_NAME`.
- In your SQL, always reference tables using the **schema-qualified name**: `SCHEMA.TABLE_NAME`.
- The schema is always present — it may be a named schema (e.g., `SALES`) or the default `dbo`.
- Examples: `mydb.SALES.ORDERS` → write `SALES.ORDERS`; `mydb.dbo.ORDERS` → write `dbo.ORDERS`.
- ❌ NEVER use unqualified table names — always prefix with the schema.

---

## QUERY CONSTRUCTION STRATEGY
1. Identify required tables from the catalog.
2. Start with `FROM` using schema-qualified table names: `SCHEMA.TABLE_NAME`.
3. Add necessary `JOIN`s with explicit ON keys.
4. Apply `WHERE` filters.
5. Apply aggregations and `GROUP BY`.
6. Add `ORDER BY` if required.
7. Add `SELECT TOP n` or `OFFSET...FETCH` if limiting results.

Return ONLY the final T-SQL query.
"""


NATIVE_MSSQL_SQL_ERROR_INSTRUCTIONS = """
## MS SQL (T-SQL) Fixes by Error Type

### Invalid object name (table not found)
- Always use schema-qualified names: `SCHEMA.TABLE_NAME` (e.g., `SALES.ORDERS`, not just `ORDERS`).
- The full catalog path is `datasource.SCHEMA.TABLE_NAME` — use only `SCHEMA.TABLE_NAME` in SQL.
- The schema may be a named schema or the default `dbo` — use whichever appears in the catalog.
- ❌ Never reference a table without its schema prefix.

### Invalid column name / column not found
- Check the exact column name in the DATA CATALOG — T-SQL is case-insensitive but the name must exist.
- Wrap reserved words or names with spaces in brackets: `[Order Date]`, `[rank]`.
- Do not use backticks or double quotes for column names.

### Incorrect syntax near / syntax error
- T-SQL does NOT support `LIMIT` — replace with `SELECT TOP n` or `OFFSET n ROWS FETCH NEXT m ROWS ONLY`.
- T-SQL does NOT support `ILIKE` — replace with `LIKE` and wrap values in `LOWER()`.
- `OFFSET...FETCH` requires an `ORDER BY` clause.
- Check for missing parentheses or commas around subqueries.

### Conversion failed when converting
- Use `CAST(col AS INT)`, `CAST(col AS DATE)`, `CAST(col AS VARCHAR(255))` etc.
- For epoch columns: `DATEADD(second, epoch_col, CAST('1970-01-01' AS DATETIME))`.
- Use `TRY_CAST()` or `TRY_CONVERT()` for unsafe conversions.

### Aggregate function in WHERE clause
- Move aggregate conditions to `HAVING` instead of `WHERE`.

### ORDER BY in subquery without TOP or FOR XML
- Add `TOP (100) PERCENT` or use a CTE/outer query for ordered subqueries.
- Better: use a CTE and apply ORDER BY in the outermost query.
"""


MINDSDB_SQL_INSTRUCTIONS = """
## Critical Rules
1. **Use ONLY tables from the DATA CATALOG** - never invent table names
2. **Use ONLY columns from the catalog or acquired knowledge** - never guess
3. **All calculations must be in SQL** - use SUM, AVG, COUNT, GROUP BY, etc.
4. **CRITICAL: Use the EXACT datasource name from the DATA CATALOG** - e.g., `spider2_lite_thelook_ecommerce`.`TABLE_NAME`
   - NEVER use placeholder names like `datasource`, `ds`, or `mindsdb`
   - Copy the datasource name EXACTLY as shown in the catalog
5. To query columns that contain special characters, use ticks around the column name, e.g. SELECT `OrderVol('000)` FROM table
6. To enforce case sensitivity of columns, use backticks instead of double quotes, e.g. SELECT `TotalCostUSD` FROM table
7. Always wrap date values in single quotes (e.g., DATE = '2024-01-02'), not as raw numbers or unquoted ISO strings.
8. Role and semantic disambiguation: determine the real-world role implied by the question (e.g., buyer, seller, investor, investee) and map to the correct column accordingly. Do not rely on generic column names like "Company", "Name", or "Amount" to infer role.
9. Example mappings: "What company invested…?" → return the investor column (provider). "What company received the investment…?" → return the company/investee column (receiver). "Who sold…?" → seller column. "Who bought…?" → buyer column.
10. Always prioritize semantic intent over literal column names.
11. When possible, add an ORDER BY clause to the query to sort the results by the most relevant column.

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

NATIVE_BIGQUERY_SQL_ERROR_INSTRUCTIONS = """
## BigQuery Fixes by Error Type

### Unknown function / syntax errors
- BigQuery does not support `->` JSON access. Use `JSON_VALUE` / `JSON_QUERY`.
- Use `PARSE_DATE('%Y%m%d', date_string)` for GA-style dates.

### Type errors
- Use `CAST(... AS INT64)` or `SAFE_CAST(... AS INT64)` instead of `UNSIGNED`.
- Use `FLOAT64` / `NUMERIC` for decimals.
"""


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
