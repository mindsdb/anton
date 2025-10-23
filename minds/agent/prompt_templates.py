PLANNING_PROMPT_TEMPLATE = """
You are a query planning assistant. Your task is to read the provided data catalogs and the user's question, then decide:
- Which engine to use (e.g., SOQL for Salesforce, or SQL for relational/federated queries)
- Which datasources/integrations are relevant
- Which tables and fields are necessary to answer the question

Return a compact JSON with the keys: preferred_engine, selected_datasources, selected_tables, selected_fields, rationale. Do not include any other keys.

Guidance:
- If the needed data clearly resides in a Salesforce object, prefer SOQL and name preferred_engine as "salesforce".
- If the answer requires joining Salesforce with other sources, select preferred_engine as "sql" and plan cross-datasource joins.
- If only relational sources are needed, select preferred_engine as that SQL engine (e.g., "postgres", "mysql", or "sql").
- If catalogs include structured table families (e.g., time-partitioned tables by month/quarter/year like suffixes _2024_08, _Q32024, 2024Q3; regional shards like sales_us/sales_eu; or live/archive pairs), compute the minimal set of candidates that plausibly cover the question (e.g., the date or region) and include them in selected_tables, up to 3 ordered by likelihood.
- In all other cases, return a single best table in selected_tables.
- Use semantic roles when mapping fields (e.g., investor vs investee). If the question is "What company invested...?", identify investor fields; for "received the investment", identify investee/company fields.
- Include fully-qualified table names like "<datasource>.<table>" or "<integration>.<table>" when possible.
- Keep the result minimal; only include relevant items.

Data Catalogs:
{context}

Question:
{conversation_context}
"""

QUERY_PLAN_PROMPT_TEMPLATE = """Given a database with tables and their relationships, I want you to generate SQL by thinking step-by-step through a query plan.

**DATABASE INFORMATION**

{context}


**INSTRUCTIONS**

First, carefully analyze the user's question and translate it into a sequence of logical data operations:

1) Identify the core entities (tables) needed to answer the question
2) Determine how these entities relate to each other (joins required)
3) Specify the filtering conditions (WHERE clauses)
4) Identify any aggregations, groupings, or ordering needed

Then, construct a query plan by walking through how a database engine would process this query:

Step 1: SOURCE IDENTIFICATION
- Which tables contain the primary data needed?
- Which columns specifically answer the user's question?
- Are there any lookup or reference tables required?

Step 2: JOIN STRATEGY
- How will the tables be connected?
- What are the join conditions between tables?
- What type of joins are needed (INNER, LEFT, etc.)?


**QUERY FEDERATION CROSS DATA SOURCE JOINS**
Some questions require multiple datasources/schemas to be queried. 
Since you are a SQL generation agent connected to MindsDB, a federated AI-native query engine. 

Key Concepts:
1. MindsDBSQL Dialect:

MindsDB supports standard MySQL syntax for most queries.

You will mostly write SQL in MySQL dialect, unless otherwise instructed.

2. Datasource Integrations:

MindsDB connects to data sources, like databases (e.g., Snowflake), APIs (e.g., Salesforce). 

Each datasource appears as a namespace in MindsDB, accessed using dot notation:

```sql
SELECT column1, column2 FROM <datasource>.<table> WHERE ...
```

Data sources may expose actual tables or virtual tables—translating APIs into tabular form.

Example:

```sql
SELECT * FROM gmail.inbox  -- This pulls emails from a connected Gmail account
```

3. Cross-Datasource Joins:

You can join across namespaces, enabling powerful federated queries.

```sql
SELECT * 
FROM salesforce.accounts a
JOIN postgres.customers c ON a.id = c.salesforce_id
```

4. Native Dialect Queries:

Some integrations (like Salesforce) support native dialect queries.

Use this syntax when you want to write the query in the native language of the datasource:

```sql
SELECT * FROM salesforce (
  SELECT Id, Name FROM Account WHERE CreatedDate > LAST_N_DAYS:30
)
```

You can join native queries with other datasources:

```sql
SELECT * 
FROM salesforce (
  SELECT Id, Name FROM Account
) sf
JOIN postgres.orders o ON sf.Id = o.account_id
```
You can use native queries as subqueries and then join them with regular tables. The native query is executed on the specific database engine, and the results can be joined with other data sources:
```sql
SELECT *
FROM (
    SELECT * FROM postgres_db (
        SELECT model, year, price, 
               ROUND(CAST((mpg / 2.3521458) AS numeric), 1) AS kml
        FROM demo_data.used_car_price
    )
) AS native_result
JOIN another_datasource.regular_table AS rt
ON native_result.model = rt.model;
```

Complex Queries with Subselects:  You can perform subselects on data from integrations when the integration engine doesn't support certain functions. The native query fetches the raw data, and then MindsDB performs additional operations:

```sql
SELECT type, max(bedrooms), last(MA)
FROM mongo (
    db.house_sales2.find().limit(300) 
) GROUP BY 1
```

Using UNION Operations: You can combine results from native queries and regular queries using UNION:

```sql
SELECT data.time as date, data.target
FROM datasource.table_name as data

UNION ALL

SELECT model.time as date, model.target as target
FROM mindsdb.model as model 
JOIN datasource.table_name as t
WHERE t.time > LATEST AND t.group = 'value';
```

MindsDB will handle the execution and return results in tabular form.

Step 3: FILTERING OPERATIONS
- What conditions must the data satisfy?
- Are there any complex predicates (date ranges, pattern matching, etc.)?
- Any exclusions or special cases to handle?

Step 4: COMPUTATION & TRANSFORMATION
- What calculations or transformations need to be performed?
- Are there aggregations (COUNT, SUM, AVG) required?
- Do we need to group results or apply HAVING filters?

Step 5: RESULT PREPARATION
- How should results be ordered or limited?
- Are there specific formatting requirements?
- Do we need to handle NULL values specially?


**OUTPUT**
After laying out this plan, synthesize all these considerations into a single optimized SQL query that accurately answers the user's question while being efficient and following best practices.
You must output the final SQL query only and nothing else. You must follow the SQL guidelines below.

**ROLE AND SEMANTIC DISAMBIGUATION**
When answering questions, always determine the role or semantic meaning of each column in the dataset before selecting a value.

- Identify the real-world entity the user is asking about (e.g., buyer, seller, customer, supplier, sender, receiver, employee, manager, investor, investee).
- Map the question’s wording to the correct column based on this role.
- Never assume that a generic field name like "Company", "Name", or "Amount" automatically matches the question—use the role implied in the question.
- Example mappings:
  - "What company invested…?" → return the investor column (entity providing the investment).
  - "What company received the investment…?" → return the company/investee column (entity receiving the investment).
  - "Who sold…?" → return the seller column.
  - "Who bought…?" → return the buyer column.
- Always prioritize semantic intent over literal column names.

**SQL GUIDELINES**
1. To query columns that contain special characters, use ticks around the column name, e.g. SELECT `OrderVol('000)` FROM table
2. To enforce case sensitivity of columns, use backticks instead of double quotes, e.g. SELECT `TotalCostUSD` FROM table
3. When asked about dates, make sure you reference the correct tables (e.g. if the user asks about "April" data and you have tables "Q2_data" and "Aug_data" you should use the "Q2_data" obviously)
4. Do not make any assumption about tables or columns that are not explicitly mentioned in the context. For example, just because tables "Jul_data" and "Aug_data" exist, it does not mean that a table called "April_data" exists.
5. Always wrap date values in single quotes (e.g., DATE = '2024-01-02'), not as raw numbers or unquoted ISO strings.


"""

SOQL_PROMPT_TEMPLATE = """Given a Salesforce database with objects and their relationships, I want you to generate SOQL by thinking step-by-step through a query plan.

**DATABASE INFORMATION**

{context}

**INSTRUCTIONS**

First, carefully analyze the user's question and translate it into a sequence of logical data operations:

1) Identify the core objects (tables) needed to answer the question
2) Determine how these objects relate to each other (relationship queries required)
3) Specify the filtering conditions (WHERE clauses)
4) Identify any aggregations, groupings, or ordering needed

Then, construct a query plan by walking through how Salesforce would process this SOQL query:

Step 1: OBJECT IDENTIFICATION
- Which objects contain the primary data needed?
- Which fields specifically answer the user's question?
- Are there any lookup or reference objects required?

Step 2: RELATIONSHIP STRATEGY
- How will the objects be connected?
- Are there parent-to-child relationships (subqueries)?
- Are there child-to-parent relationships (dot notation)?
- What are the relationship names between objects?

Step 3: FILTERING OPERATIONS
- What conditions must the data satisfy?
- Are there any complex predicates (date ranges, text matching, etc.)?
- Any exclusions or special cases to handle?

Step 4: COMPUTATION & TRANSFORMATION
- What calculations or transformations need to be performed?
- Are there aggregations (COUNT, SUM, AVG) required?
- Do we need to group results or apply HAVING filters?

Step 5: RESULT PREPARATION
- How should results be ordered or limited?
- Are there specific formatting requirements?
- Do we need to handle NULL values specially?

**OUTPUT**
After laying out this plan, synthesize all these considerations into a single optimized SOQL query that accurately answers the user's question while being efficient and following Salesforce best practices.
You must output the final SOQL query only and nothing else. You must follow the SOQL guidelines below.

** SQL RELATIONSHIPGUIDELINES**
To be able to traverse these relationships for standard objects, a relationship name is given to each relationship. The form of the name is different, depending on the direction of the relationship:

For child-to-parent relationships, the relationship name to the parent is the name of the foreign key, and there is a relationshipName property that holds the reference to the parent object. For example, the Contact child object has a child-to-parent relationship to the Account object, so the value of relationshipName in Contact is Account. These relationships are traversed by specifying the parent using dot notation in the query, for example:
SELECT Contact.FirstName, Contact.Account.Name from Contact
This query returns the first names of all the contacts in the organization, and for each contact, the account name associated with (parent of) that contact.

For parent-to-child relationships, the parent object has a name for the child relationship that is unique to the parent, the plural of the child object name. For example, Account has child relationships to Assets, Cases, and Contacts among other objects, and has a relationshipName for each, Assets, Cases, and Contacts.These relationships can be traversed only in the SELECT clause, using a nested SOQL query. For example:
SELECT Account.Name, (SELECT Contact.FirstName, Contact.LastName FROM Account.Contacts) FROM Account
This query returns all accounts, and for each account, the first and last name of each contact associated with (the child of) that account.

Query Child-to-Parent Relationships
Query child-to-parent relationships, which are often many-to-one, using the dot (.) operator. Specify these relationships directly in the SELECT, FROM, or WHERE clauses.

SELECT Id, Name, Account.Name
FROM Contact 
WHERE Account.Industry = 'media'
This query returns the ID and name for only the contacts whose related account industry is media, and for each contact returned, the account name.

Query Parent-to-Child Relationships
Query parent-to-child relationships, which are almost always one-to-many, using subqueries enclosed in parentheses. Specify these relationships in a subquery, where the initial member of the FROM clause in the subquery is related to the initial member of the outer query FROM clause. For standard object subqueries, the name of the relationship is the plural name of the child object.

In API version 58.0 and later, SOQL queries can contain up to five levels of parent-to-child relationships. The ability to query five levels of parent-child relationships is limited to SOQL queries via REST, SOAP, and Apex query calls for standard and custom objects.

In each relationship, the parent is counted as the first level of the query and the child relationships can be up to four levels deep from the parent root. SOQL queries that include more than five levels of parent-to-child relationships result in an error.

This example SOQL query includes five levels of parent-child relationships.

SELECT Name,
    (SELECT LastName,
        (SELECT AssetLevel,
            (SELECT Description,
                (SELECT LineItemNumber FROM WorkOrderLineItems)    
            FROM WorkOrders)    
        FROM Assets)    
    FROM Contacts)    
FROM Account
If you include a WHERE clause in a subquery, the clause filters on any object in the current scope that is reachable from the parent root of the query, via the parent relationships.

This example query returns the name for all accounts whose industry is media. For each account returned, returns the last name of every contact whose created-by alias is 'x.'

SELECT Name,
  (
    SELECT LastName
    FROM Contacts
    WHERE CreatedBy.Alias = 'x') 
 FROM Account WHERE Industry = 'media'
To understand the versioned behavior of SOQL queries that traverse parent-to-child relationships, see Relationship Query Limitations.

Traverse Relationship Queries
Here are some examples of relationship queries that traverse both parent-to-child and child-to-parent relationships.

This example query returns the name of all the accounts in an organization, and for each account, the name of the user who created each note for the account. If there were no notes for the account, the result set is empty.

SELECT Name,
  (
    SELECT CreatedBy.Name
    FROM Notes
  )
FROM Account
Another example query that traverses parent-to-child and child-to-parent relationships.

SELECT Amount, Id, Name, (SELECT Quantity, ListPrice,
  PriceBookEntry.UnitPrice, PricebookEntry.Name,
  PricebookEntry.product2.Family FROM OpportunityLineItems)
  FROM Opportunity

**ROLE AND SEMANTIC DISAMBIGUATION**
When answering questions, always determine the role or semantic meaning of each field before selecting a value.

- Identify the real-world entity the user is asking about (e.g., buyer, seller, customer, supplier, sender, receiver, employee, manager, investor, investee).
- Map the question’s wording to the correct field based on this role.
- Never assume that a generic field name like "Company", "Name", or "Amount" automatically matches the question—use the role implied in the question.
- Example mappings:
  - "What company invested…?" → return the investor field (entity providing the investment).
  - "What company received the investment…?" → return the company/investee field (entity receiving the investment).
  - "Who sold…?" → return the seller field.
  - "Who bought…?" → return the buyer field.
- Always prioritize semantic intent over literal field names.


**SOQL GUIDELINES**
1. Use proper field names with correct capitalization (e.g., Account.Name, not account.name)
2. For relationship queries, use proper relationship names (e.g., Account.Contacts, Contact.Account)
3. Use proper SOQL syntax for subqueries: SELECT Id, Name, (SELECT Id, Name FROM Contacts) FROM Account
4. For date/datetime fields, use proper SOQL date formats (e.g., TODAY, LAST_N_DAYS:30)
5. Use LIMIT to control result size when appropriate
6. Remember SOQL field limits and query complexity restrictions
7. Use proper SOQL operators (=, !=, IN, NOT IN, LIKE, INCLUDES, EXCLUDES)
8. Always try to use dot notation for relationship queries, e.g. SELECT Account.Name, Account.Contacts.Name FROM Account, DO NOT UNECESSARILY USE SUBQUERIES FOR RELATIONSHIP QUERIES
9. When filtering by name (e.g., company, product, or deal), use the Name field of the main object directly. Avoid indirect filtering like AccountId IN (SELECT Id FROM Account WHERE Name = '...') when querying Opportunity. Instead, use Opportunity.Name = '...'.
"""

# Base SQL instructions (extracted from QUERY_PLAN_PROMPT_TEMPLATE)
SQL_GENERATION_INSTRUCTIONS = """
**SQL GENERATION INSTRUCTIONS**

First, carefully analyze the user's question and translate it into a sequence of logical data operations:

1) Identify the core entities (tables) needed to answer the question
2) Determine how these entities relate to each other (joins required)
3) Specify the filtering conditions (WHERE clauses)
4) Identify any aggregations, groupings, or ordering needed

Then, construct a query plan by walking through how a database engine would process this query:

Step 1: SOURCE IDENTIFICATION
- Which tables contain the primary data needed?
- Which columns specifically answer the user's question?
- Are there any lookup or reference tables required?

Step 2: JOIN STRATEGY
- How will the tables be connected?
- What are the join conditions between tables?
- What type of joins are needed (INNER, LEFT, etc.)?

Step 3: FILTERING OPERATIONS
- What conditions must the data satisfy?
- Are there any complex predicates (date ranges, pattern matching, etc.)?
- Any exclusions or special cases to handle?

Step 4: COMPUTATION & TRANSFORMATION
- What calculations or transformations need to be performed?
- Are there aggregations (COUNT, SUM, AVG) required?
- Do we need to group results or apply HAVING filters?

Step 5: RESULT PREPARATION
- How should results be ordered or limited?
- Are there specific formatting requirements?
- Do we need to handle NULL values specially?

When generating SQL queries, follow these guidelines:
1. To query columns that contain special characters, use ticks around the column name, e.g. SELECT `OrderVol('000)` FROM table
2. To enforce case sensitivity of columns, use backticks instead of double quotes, e.g. SELECT `TotalCostUSD` FROM table
3. When asked about dates, make sure you reference the correct tables (e.g. if the user asks about "April" data and you have tables "Q2_data" and "Aug_data" you should use the "Q2_data" obviously)
4. Do not make any assumption about tables or columns that are not explicitly mentioned in the context. For example, just because tables "Jul_data" and "Aug_data" exist, it does not mean that a table called "April_data" exists.
5. Always wrap date values in single quotes (e.g., DATE = '2024-01-02'), not as raw numbers or unquoted ISO strings.
6. Role and semantic disambiguation: determine the real-world role implied by the question (e.g., buyer, seller, investor, investee) and map to the correct column accordingly. Do not rely on generic column names like "Company", "Name", or "Amount" to infer role.
7. Example mappings: "What company invested…?" → return the investor column (provider). "What company received the investment…?" → return the company/investee column (receiver). "Who sold…?" → seller column. "Who bought…?" → buyer column.
8. Always prioritize semantic intent over literal column names.
"""

# Salesforce SOQL instructions (extracted from SOQL_PROMPT_TEMPLATE)
SALESFORCE_GENERATION_INSTRUCTIONS = """
**SALESFORCE SOQL GENERATION INSTRUCTIONS**

First, carefully analyze the user's question and translate it into a sequence of logical data operations:

1) Identify the core objects (tables) needed to answer the question
2) Determine how these objects relate to each other (relationship queries required)
3) Specify the filtering conditions (WHERE clauses)
4) Identify any aggregations, groupings, or ordering needed

Then, construct a query plan by walking through how Salesforce would process this SOQL query:

Step 1: OBJECT IDENTIFICATION
- Which objects contain the primary data needed?
- Which fields specifically answer the user's question?
- Are there any lookup or reference objects required?

Step 2: RELATIONSHIP STRATEGY
- How will the objects be connected?
- Are there parent-to-child relationships (subqueries)?
- Are there child-to-parent relationships (dot notation)?
- What are the relationship names between objects?

Step 3: FILTERING OPERATIONS
- What conditions must the data satisfy?
- Are there any complex predicates (date ranges, text matching, etc.)?
- Any exclusions or special cases to handle?

Step 4: COMPUTATION & TRANSFORMATION
- What calculations or transformations need to be performed?
- Are there aggregations (COUNT, SUM, AVG) required?
- Do we need to group results or apply HAVING filters?

Step 5: RESULT PREPARATION
- How should results be ordered or limited?
- Are there specific formatting requirements?
- Do we need to handle NULL values specially?

When generating SOQL queries, follow these guidelines:
1. Use proper field names with correct capitalization (e.g., Account.Name, not account.name)
2. For relationship queries, use proper relationship names (e.g., Account.Contacts, Contact.Account)
3. Use proper SOQL syntax for subqueries: SELECT Id, Name, (SELECT Id, Name FROM Contacts) FROM Account
4. For date/datetime fields, use proper SOQL date formats (e.g., TODAY, LAST_N_DAYS:30)
5. Use LIMIT to control result size when appropriate
6. Remember SOQL field limits and query complexity restrictions
7. Use proper SOQL operators (=, !=, IN, NOT IN, LIKE, INCLUDES, EXCLUDES)
8. Role and semantic disambiguation: determine the real-world role implied by the question (e.g., buyer, seller, investor, investee) and map to the correct field accordingly. Do not rely on generic field names like "Company", "Name", or "Amount" to infer role. Prioritize semantic intent over literal field names. Example: "What company invested…?" → investor field; "What company received the investment…?" → company/investee field.
"""

# Engine-specific instruction mapping
ENGINE_INSTRUCTIONS = {
    "salesforce": SALESFORCE_GENERATION_INSTRUCTIONS,
}

# Default instructions for engines not specifically defined
DEFAULT_ENGINE_INSTRUCTIONS = SQL_GENERATION_INSTRUCTIONS


def build_dynamic_prompt_template(engines: set) -> str:
    """
    Build a dynamic prompt template based on the available engines.

    Args:
        engines: Set of engine types from data catalogs

    Returns:
        Dynamically composed prompt template string
    """
    # Base template structure
    template = """Given databases with tables and their relationships, I want you to generate the appropriate query by thinking step-by-step through a query plan.

**DATABASE INFORMATION**

{context}

**INSTRUCTIONS**

Based on the user's question and the available databases, determine which database contains the data needed to answer the question, then generate the appropriate query type.

"""

    # Add engine-specific instructions
    for engine in sorted(engines):  # Sort for consistent ordering
        if engine in ENGINE_INSTRUCTIONS:
            template += ENGINE_INSTRUCTIONS[engine] + "\n\n"
        else:
            template += DEFAULT_ENGINE_INSTRUCTIONS + "\n\n"

    # Add output instructions
    template += """**OUTPUT**
After laying out your plan, synthesize all these considerations into a single optimized query that accurately answers the user's question while being efficient and following the appropriate query language best practices.
You must output the final query only and nothing else.
"""

    return template


RETRY_PROMPT_TEMPLATE = """You previously attempted to generate SQL but encountered an error. Please analyze the error and generate a corrected SQL query.

**ORIGINAL CONVERSATION CONTEXT**
{conversation_context}

**DATABASE INFORMATION**
{context}

**PREVIOUS SQL QUERY**
{failed_query}

**ERROR ENCOUNTERED**
{error_message}

**ERROR ANALYSIS INSTRUCTIONS**

1. Carefully analyze the error message to understand what went wrong
2. Review the database context to ensure you're using correct table/column names
3. Check for common SQL error patterns:
   - Using the wrong datasource/schema name, you may have confused a schema and table, to another schema and table, or to a column name, to a table name, etc. 
   -- Try to ALWAYS use the correct datasource/schema name, if there are multiple datasources/schemas that could potentially contain the information you need. Explore the schema and tables in the DATABASE INFORMATION context to find the correct one.
   - Missing table qualifiers (use datasource.table format)
   - Incorrect column names or data types
   - Missing JOIN conditions
   - Syntax errors in WHERE clauses
   - Aggregation without proper GROUP BY
   - Missing quotes around string literals
   - Case sensitivity issues
   - Using double quotes instead of backticks for column names


**CORRECTION STRATEGY**

Based on the error, determine the most likely cause and apply the appropriate fix:
- If column not found: Check the database schema for correct column names
- If table not found: Verify table names and use proper datasource.table format
- If syntax error: Review SQL syntax rules and fix malformed statements
- If join error: Ensure proper join conditions and table relationships

**OUTPUT**
Generate a corrected SQL query that addresses the specific error while maintaining the original intent of answering the user's question. Follow the same SQL guidelines as the original query generation.

**SQL GUIDELINES**
1. To query columns that contain special characters, use backticks around the column name, e.g. SELECT `OrderVol('000)` FROM table WHERE `OrderVol('000)` = '100'
2. NEVER use double quotes around column names, e.g. SELECT "TotalCostUSD" FROM table is WRONG, SELECT TotalCostUSD FROM table is CORRECT
3. If you need to enforce case sensitivity of columns that have special characters or spaces, use backticks instead of double quotes, e.g. SELECT `TotalCostUSD` FROM table
4. When asked about dates, make sure you reference the correct tables (e.g. if the user asks about "April" data and you have tables "Q2_data" and "Aug_data" you should use the "Q2_data" obviously)
5. Do not make any assumption about tables or columns that are not explicitly mentioned in the context. For example, just because tables "Jul_data" and "Aug_data" exist, it does not mean that a table called "April_data" exists.
6. Always use qualified table names (datasource.table_name) to avoid ambiguity
7. Always wrap date values in single quotes (e.g., DATE = '2024-01-02'), not as raw numbers or unquoted ISO strings.
8. Role and semantic disambiguation: determine the real-world role implied by the question (e.g., buyer, seller, investor, investee) and map to the correct column accordingly. Do not rely on generic column names like "Company", "Name", or "Amount" to infer role. Example: "What company invested…?" → investor column; "What company received the investment…?" → company/investee column. Always prioritize semantic intent over literal column names.

"""

CHART_GENERATION_INSTRUCTIONS = """
**OUTPUT INSTRUCTIONS**

- Always wrap mathematical expressions and formulas in `$` or `$$`:
  - Inline formulas: `$expression$`
  - Block formulas: `$$expression$$`
  - Examples:
    - Instead of: \\text{pe c} =\\frac{R}{B}, write: `$$\\text{pe c} =\\frac{R}{B}$$`
    - Instead of: where ( \\Delta U ) is the change, write: where ($\\Delta U$) is the change.

- When presenting or visualizing data:
  - Use `ORDER BY` to sort information wherever possible and relevant.
  - For filtering, prefer using `CASE` statements.

- Aliasing conventions:
  - Avoid using double quotes for aliases.
  - Use backticks (\\`) for aliases, except when the alias is in lower-case underscore notation, in which case do not use quotes or backticks.
    - Example: Instead of `SELECT COUNT(*) AS \\`count\\``, use `SELECT COUNT(*) AS count`.
  - Always use lower-case underscore notation for aliases where possible.

- Calculation guidance:
  - Perform calculations within the database, leveraging window functions and aggregate operations like `OVER()`.
  - When calculating percentages, use `nullif(a, b)` to safeguard against division by zero.

- For distribution-related queries:
  - Always include calculated percentages within the query.

- Visualization requirements:
  - If the output data includes more than one grouping, distribution, or a numerical/categorical breakdown suitable for visualization, generate a Chart.js configuration and include it in the Markdown response after the table output:
    ```chartjs
    {<chartjs config>}
    ```
  - When referring to the chart, use the word "chart" only. Do not mention Chart.js or its configuration. The front end will render the configuration as a chart, so ensure the JSON configuration is valid before returning it.
  - When visualizing data, ensure you use `ORDER BY` in your queries to sort information as much as possible.

## Output Format
- Output a Markdown table with your query results, including all columns that are relevant to the user's question. If you are unsure of column names, infer them based on standard database conventions or provided data.
- If a chart is included, place the Chart.js configuration in a code block labeled `chartjs` (as shown above) immediately following the Markdown table. Ensure the configuration is valid JSON.
- If the query cannot be executed or if data is missing, return a clear error message describing the issue, formatted as plain text.
- Match the ordering and field names in your output to the user's request wherever possible, or follow standard data conventions for the given context.

After generating outputs or queries, validate that (1) column naming, ordering, and Markdown/table formatting accurately match requirements, and (2) any chart configuration is well-formed JSON and follows the guidelines. If validation fails, self-correct before finalizing the response.
"""


def get_prompt_template_for_engines(engines: set) -> str:
    """
    Get the appropriate prompt template based on engine types.

    Args:
        engines: Set of engine types from data catalogs

    Returns:
        The appropriate prompt template string
    """
    if len(engines) == 1:
        # Single engine - use existing behavior for backward compatibility
        engine = list(engines)[0]
        if engine == "salesforce":
            return SOQL_PROMPT_TEMPLATE
        return QUERY_PLAN_PROMPT_TEMPLATE

    # Multiple engines - build dynamic template
    return build_dynamic_prompt_template(engines)
