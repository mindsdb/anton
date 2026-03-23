CHART_GENERATION_INSTRUCTIONS = """
**CHART VISUALIZATION INSTRUCTIONS**

When generating SQL queries, you can also specify a chart visualization by populating the `chart_intent` field
in your output. The chart intent specifies WHAT to visualize using column names from your query - the backend
will compile the full chart using complete query results.

- When to include a chart:
  - When query results are suitable for visualization (distributions, trends, comparisons)
  - When the user explicitly asks for a chart, graph, or visualization
  - When presenting aggregated data that would benefit from visual representation

- Chart Intent Schema:
  - For bar or line charts, set `chart_intent` to:
    {
      "type": "bar",
      "x": "<column name for x-axis>",
      "y": "<column name OR array of column names for y-axis metrics>",
      "series": "<optional: column to split into multiple series>",
      "aggregate": "<optional: sum|avg|count|min|max, default is sum>",
      "limit": "<optional: max categories/points to display>",
      "max_series": "<optional: max series to show, default 12 - set higher if user wants ALL items>",
      "title": "<optional: chart title>"
    }
  - For pie charts, set `chart_intent` to:
    {
      "type": "pie",
      "label": "<column name for category labels>",
      "value": "<column name for numeric values>",
      "aggregate": "<optional: sum|avg|count|min|max, default is sum>",
      "limit": "<optional: max slices to display>",
      "title": "<optional: chart title>"
    }
  - For scatter charts, set `chart_intent` to:
    {
      "type": "scatter",
      "x": "<column name for x-axis (numeric)>",
      "y": "<column name for y-axis (numeric)>",
      "series": "<optional: column to split into multiple series>",
      "limit": "<optional: max data points to display, default 1000>",
      "max_series": "<optional: max series to show, default 12>",
      "title": "<optional: chart title>"
    }

- Chart Type Guidelines:
  - Use "line" when the x-axis represents time or dates (trends over time).
  - Use "bar" for categorical comparisons (e.g., sales by region, count by category).
  - Use "pie" for composition or distribution of a whole (use only when there are few categories, ideally < 8).
  - Use "scatter" when showing correlation or relationship between two numeric variables (e.g., price vs. quantity, height vs. weight). Both x and y must be numeric columns.
  - Use "series" to split data by a categorical column (e.g., revenue over months, split by product line).
  - IMPORTANT: By default, charts are limited to 12 series. If the series column has more than 12 distinct values, only the top 12 (by total value) will be shown and the rest will be truncated. Use "max_series" to override this limit when:
    - The user asks for "all" categories/items (e.g., "show all 26 models")
    - The user mentions a specific number greater than 12 (e.g., "show me the 20 products")
    - The user complains about missing data or truncated series
    - You know the series column has more than 12 distinct values and complete data is important
    Example: "max_series": 30 to show up to 30 series instead of 12.
  - IMPORTANT: By default, charts apply conservative point limits (e.g., time-series often defaults to 365; categorical to 50; scatter to 1000; pie to 12). Use "limit" to override this when:
    - The user asks for "all data points" or "show everything" (e.g., "show all 500 records")
    - The user mentions wanting to see more than the default (e.g., "show me all rental prices over time")
    - The user complains about missing or truncated data points
    - You know the dataset has more rows than the default limit and complete data is important
    Example: "limit": 1000 to show up to 1000 data points instead of the default 365/50.
    The maximum allowed value is 10000.
  - Use multiple "y" columns (as an array) to compare different metrics on the same chart (e.g., "y": ["avg_length", "avg_width"]). Note: when using multiple y columns, the "series" field is ignored. This only applies to bar and line charts.
  - ONLY use these four chart types: "bar", "line", "pie", "scatter". Do NOT use "area", "radar", or any other chart types.
  - Note: The backend automatically detects temporal (date/time) columns and optimizes the chart axis accordingly. Just ensure your SQL query returns date values in a standard format (e.g., DATE, TIMESTAMP, or ISO 8601 strings).

- Important: Do NOT include actual data values in the chart intent - only reference the column names from your SQL query results. The backend will fetch the full query results and compile the chart.

- If no chart is appropriate for the query (e.g., single value results, text-heavy data), leave the `chart_intent` field as null.
"""
