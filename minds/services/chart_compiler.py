"""
Chart compiler service for transforming DataFrames into Chart.js configurations.

This module takes a pandas DataFrame and a chart intent, then compiles them
into a complete Chart.js configuration with appropriate aggregation, sorting,
and truncation.
"""

import pandas as pd

from minds.common.logger import get_logger
from minds.common.settings.app_settings import get_app_settings
from minds.schemas.charts import ChartMeta, ChartWarning, PieIntent, ScatterIntent, XYIntent

logger = get_logger(__name__)

# Processing limits
_settings = get_app_settings()
MAX_ROWS_TO_PROCESS = _settings.chart_compiler.max_rows_to_process
MAX_SERIES = _settings.chart_compiler.max_series

# Default limits by context
DEFAULT_LIMIT_TEMPORAL = 365
DEFAULT_LIMIT_CATEGORICAL = 50
DEFAULT_LIMIT_PIE = 12
DEFAULT_LIMIT_SCATTER = 1000


def _is_temporal(series: pd.Series) -> bool:
    """
    Check if a pandas Series contains temporal data.

    A series is considered temporal if:
    - It's already a datetime type, OR
    - It contains string values that look like dates (at least 80% parseable)

    Numeric columns (int/float) are NOT treated as temporal to avoid
    misinterpreting small integers as Unix timestamps.

    Args:
        series: The pandas Series to check.

    Returns:
        True if the series is temporal, False otherwise.
    """
    non_null = series.dropna()
    if non_null.empty:
        return False

    # If already datetime type, it's temporal
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # If numeric type (int or float), do NOT treat as temporal
    # This prevents integers like 0, 1, 2, 3 from being interpreted as Unix timestamps
    if pd.api.types.is_numeric_dtype(series):
        return False

    # For string/object types, try to parse as datetime
    try:
        # Only attempt parsing on string-like data
        parsed = pd.to_datetime(non_null, errors="coerce", utc=True)
        parse_rate = parsed.notna().mean()

        # Additional sanity check: ensure parsed dates are in a reasonable range
        # (between year 1900 and 2100) to avoid false positives
        if parse_rate >= 0.8:
            valid_dates = parsed.dropna()
            if not valid_dates.empty:
                min_year = valid_dates.min().year
                max_year = valid_dates.max().year
                if min_year >= 1900 and max_year <= 2100:
                    return True

        return False
    except Exception:
        return False


def _to_datetime_if_temporal(series: pd.Series) -> tuple[pd.Series, bool]:
    """
    Convert a Series to datetime if it appears to be temporal.

    Args:
        series: The pandas Series to potentially convert.

    Returns:
        Tuple of (converted series, is_temporal flag).
    """
    if _is_temporal(series):
        return pd.to_datetime(series, errors="coerce", utc=True), True
    return series, False


def _aggregate_series(grouped: pd.core.groupby.SeriesGroupBy, agg: str) -> pd.Series:
    """
    Apply aggregation function to a grouped Series.

    Args:
        grouped: The grouped Series to aggregate.
        agg: Aggregation function name.

    Returns:
        Aggregated Series.
    """
    if agg == "sum":
        return grouped.sum()
    elif agg == "avg":
        return grouped.mean()
    elif agg == "min":
        return grouped.min()
    elif agg == "max":
        return grouped.max()
    elif agg == "count":
        return grouped.count()
    else:
        return grouped.sum()  # Default fallback


def _compile_xy_chart(
    df: pd.DataFrame,
    intent: XYIntent,
    warnings: list[dict],
) -> tuple[dict, ChartMeta]:
    """
    Compile a bar or line chart from DataFrame and XYIntent.

    Args:
        df: Source DataFrame.
        intent: Chart intent specification.
        warnings: List to append warnings to.

    Returns:
        Tuple of (Chart.js config, metadata).
    """
    xcol = intent.x
    scol = intent.series

    logger.debug(f"Compiling XY chart with max_series={intent.max_series}")

    # Normalize y to list for uniform handling
    ycols = intent.y if isinstance(intent.y, list) else [intent.y]
    multi_y = len(ycols) > 1

    # If multiple y columns, series is ignored (each y column becomes a series)
    if multi_y and scol:
        warnings.append(
            {
                "code": "SERIES_IGNORED",
                "message": "Series column ignored when multiple Y columns specified.",
            }
        )
        scol = None

    # Validate required columns
    missing = [c for c in [xcol] + ycols if c not in df.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {', '.join(missing)}")

    # Validate series column if specified
    if scol and scol not in df.columns:
        warnings.append(
            {
                "code": "UNKNOWN_SERIES",
                "message": f"Series column '{scol}' not found; rendering single series.",
            }
        )
        scol = None

    df2 = df.copy()

    # Temporal detection and conversion for X axis
    df2[xcol], is_time = _to_datetime_if_temporal(df2[xcol])

    # Check if X column is numeric (for sorting purposes)
    is_numeric_x = pd.api.types.is_numeric_dtype(df2[xcol])

    # Track field types for metadata
    fields = {
        xcol: "temporal" if is_time else ("quantitative" if is_numeric_x else "nominal"),
    }
    for yc in ycols:
        fields[yc] = "quantitative"
    if scol:
        fields[scol] = "nominal"

    # Coerce Y columns to numeric
    for yc in ycols:
        df2[yc] = pd.to_numeric(df2[yc], errors="coerce")
    df2 = df2.dropna(subset=[xcol] + ycols)

    # Determine aggregation function
    agg = intent.aggregate or "sum"

    default_limit = DEFAULT_LIMIT_TEMPORAL if is_time else DEFAULT_LIMIT_CATEGORICAL
    limit = intent.limit or default_limit

    # Build datasets based on mode: multi-y OR series
    datasets = []
    if multi_y:
        # Multiple Y columns mode: each y column becomes a dataset
        # Group by X only, aggregate each y column
        aggregated = {}
        for yc in ycols:
            grouped = df2.groupby(xcol, dropna=False)[yc]
            aggregated[yc] = _aggregate_series(grouped, agg)

        # Build X domain from first y column's aggregation
        first_agg = aggregated[ycols[0]]

        if is_time:
            x_domain = sorted(first_agg.index.dropna().unique())
            if len(x_domain) > limit:
                x_domain = x_domain[-limit:]
                warnings.append(
                    {
                        "code": "TRUNCATED",
                        "message": f"Limited to {limit} most recent points.",
                    }
                )
        elif is_numeric_x:
            x_domain = sorted(first_agg.index.dropna().unique())
            if len(x_domain) > limit:
                x_domain = x_domain[:limit]
                warnings.append(
                    {
                        "code": "TRUNCATED",
                        "message": f"Limited to first {limit} values.",
                    }
                )
        else:
            # Categorical: sort by sum of all y columns
            total = sum(aggregated[yc] for yc in ycols)
            total = total.sort_values(ascending=False)
            x_domain = total.head(limit).index.tolist()
            if len(total) > limit:
                warnings.append(
                    {
                        "code": "TRUNCATED",
                        "message": f"Limited to top {limit} categories.",
                    }
                )

        # Build dataset for each y column
        for yc in ycols:
            agg_series = aggregated[yc]
            data = [float(agg_series.get(xv, 0.0)) for xv in x_domain]
            datasets.append({"label": yc, "data": data})

    else:
        # Single Y column mode (original logic)
        ycol = ycols[0]

        # Group by X (and series if present)
        group_cols = [xcol] + ([scol] if scol else [])
        grouped = df2.groupby(group_cols, dropna=False)[ycol]
        g = _aggregate_series(grouped, agg).reset_index(name="__val__")

        # Truncate series if too many
        if scol:
            series_limit = intent.max_series or MAX_SERIES
            totals = g.groupby(scol)["__val__"].sum().sort_values(ascending=False)
            keep = totals.head(series_limit).index.tolist()
            if len(totals) > series_limit:
                warnings.append(
                    {
                        "code": "SERIES_TRUNCATED",
                        "message": f"Limited to top {series_limit} series.",
                    }
                )
            g = g[g[scol].isin(keep)]

        # Build X domain
        if is_time:
            x_domain = sorted(g[xcol].dropna().unique())
            if len(x_domain) > limit:
                x_domain = x_domain[-limit:]
                warnings.append(
                    {
                        "code": "TRUNCATED",
                        "message": f"Limited to {limit} most recent points.",
                    }
                )
        elif is_numeric_x:
            x_domain = sorted(g[xcol].dropna().unique())
            if len(x_domain) > limit:
                x_domain = x_domain[:limit]
                warnings.append(
                    {
                        "code": "TRUNCATED",
                        "message": f"Limited to first {limit} values.",
                    }
                )
        else:
            totals = g.groupby(xcol)["__val__"].sum().sort_values(ascending=False)
            x_domain = totals.head(limit).index.tolist()
            if len(totals) > limit:
                warnings.append(
                    {
                        "code": "TRUNCATED",
                        "message": f"Limited to top {limit} categories.",
                    }
                )

        g = g[g[xcol].isin(x_domain)]

        # Build datasets
        if scol:
            for series_name, sg in g.groupby(scol):
                m = dict(zip(sg[xcol], sg["__val__"], strict=False))
                data = [float(m.get(xv, 0.0)) for xv in x_domain]
                datasets.append({"label": str(series_name), "data": data})
        else:
            m = dict(zip(g[xcol], g["__val__"], strict=False))
            data = [float(m.get(xv, 0.0)) for xv in x_domain]
            datasets.append({"label": intent.title or ycol, "data": data})

    # Format labels
    labels = [pd.Timestamp(v).date().isoformat() for v in x_domain] if is_time else [str(v) for v in x_domain]

    # Determine if legend should be shown (multiple series or multiple y columns)
    show_legend = bool(scol) or multi_y

    # Y-axis label: use single column name, or generic label for multiple
    y_axis_label = ycols[0] if len(ycols) == 1 else "Value"

    # Build Chart.js config
    config = {
        "type": intent.type,
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "legend": {"display": show_legend},
                "title": {"display": bool(intent.title), "text": intent.title or ""},
                "tooltip": {"enabled": True},
            },
            "scales": {
                "x": {
                    "type": "timeseries" if is_time else "category",
                    "title": {
                        "display": True,
                        "text": xcol,
                    },
                },
                "y": {
                    "title": {
                        "display": True,
                        "text": y_axis_label,
                    },
                    "beginAtZero": True,
                },
            },
        },
    }

    meta = ChartMeta(
        row_count=len(df),
        used_rows=len(df2),
        points=len(labels),
        series=len(datasets),
        fields=fields,
    )

    return config, meta


def _compile_pie_chart(
    df: pd.DataFrame,
    intent: PieIntent,
    warnings: list[dict],
) -> tuple[dict, ChartMeta]:
    """
    Compile a pie chart from DataFrame and PieIntent.

    Args:
        df: Source DataFrame.
        intent: Chart intent specification.
        warnings: List to append warnings to.

    Returns:
        Tuple of (Chart.js config, metadata).
    """
    label_col, value_col = intent.label, intent.value

    # Validate required columns
    missing = [c for c in [label_col, value_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {', '.join(missing)}")

    df2 = df.copy()

    # Track field types for metadata
    fields = {
        label_col: "nominal",
        value_col: "quantitative",
    }

    # Coerce value column to numeric
    df2[value_col] = pd.to_numeric(df2[value_col], errors="coerce")
    df2 = df2.dropna(subset=[label_col, value_col])

    # Determine aggregation function
    agg = intent.aggregate or "sum"

    # Group by label and aggregate
    grouped = df2.groupby(label_col)[value_col]
    g = _aggregate_series(grouped, agg)
    g = g.sort_values(ascending=False)

    # Apply limit with "Other" bucket
    limit = intent.limit or DEFAULT_LIMIT_PIE
    if len(g) > limit:
        top = g.head(limit)
        other = float(g.iloc[limit:].sum())
        g = top.copy()
        if other > 0:
            g["Other"] = other
        warnings.append(
            {
                "code": "TRUNCATED",
                "message": f"Limited to top {limit} categories (+Other).",
            }
        )

    labels = [str(k) for k in g.index.tolist()]
    data = [float(v) for v in g.values.tolist()]

    # Build Chart.js config
    config = {
        "type": "pie",
        "data": {"labels": labels, "datasets": [{"data": data}]},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "legend": {"display": True},
                "title": {"display": bool(intent.title), "text": intent.title or ""},
                "tooltip": {"enabled": True},
            },
        },
    }

    meta = ChartMeta(
        row_count=len(df),
        used_rows=len(df2),
        points=len(labels),
        series=1,
        fields=fields,
    )

    return config, meta


def _compile_scatter_chart(
    df: pd.DataFrame,
    intent: ScatterIntent,
    warnings: list[dict],
) -> tuple[dict, ChartMeta]:
    """
    Compile a scatter chart from DataFrame and ScatterIntent.

    Scatter charts use a different data format than line/bar charts:
    - Data points are {x, y} objects instead of arrays aligned with labels
    - Both axes are linear (numeric) scales

    Args:
        df: Source DataFrame.
        intent: Chart intent specification.
        warnings: List to append warnings to.

    Returns:
        Tuple of (Chart.js config, metadata).
    """
    xcol = intent.x
    ycol = intent.y
    scol = intent.series

    logger.debug(f"Compiling scatter chart with max_series={intent.max_series}")

    # Validate required columns
    missing = [c for c in [xcol, ycol] if c not in df.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {', '.join(missing)}")

    # Validate series column if specified
    if scol and scol not in df.columns:
        warnings.append(
            {
                "code": "UNKNOWN_SERIES",
                "message": f"Series column '{scol}' not found; rendering single series.",
            }
        )
        scol = None

    df2 = df.copy()

    # Track field types for metadata
    fields = {
        xcol: "quantitative",
        ycol: "quantitative",
    }
    if scol:
        fields[scol] = "nominal"

    # Coerce X and Y columns to numeric (scatter requires numeric axes)
    df2[xcol] = pd.to_numeric(df2[xcol], errors="coerce")
    df2[ycol] = pd.to_numeric(df2[ycol], errors="coerce")
    df2 = df2.dropna(subset=[xcol, ycol])

    # Apply point limit
    limit = intent.limit or DEFAULT_LIMIT_SCATTER
    if len(df2) > limit:
        df2 = df2.head(limit)
        warnings.append(
            {
                "code": "TRUNCATED",
                "message": f"Limited to first {limit} data points.",
            }
        )

    # Build datasets
    datasets = []
    total_points = 0

    if scol:
        # Multiple series mode: split by series column
        series_limit = intent.max_series or MAX_SERIES
        series_counts = df2[scol].value_counts()
        keep_series = series_counts.head(series_limit).index.tolist()

        if len(series_counts) > series_limit:
            warnings.append(
                {
                    "code": "SERIES_TRUNCATED",
                    "message": f"Limited to top {series_limit} series.",
                }
            )

        for series_name in keep_series:
            series_data = df2[df2[scol] == series_name]
            points = [{"x": float(row[xcol]), "y": float(row[ycol])} for _, row in series_data.iterrows()]
            datasets.append({"label": str(series_name), "data": points})
            total_points += len(points)
    else:
        # Single series mode
        points = [{"x": float(row[xcol]), "y": float(row[ycol])} for _, row in df2.iterrows()]
        datasets.append({"label": intent.title or f"{ycol} vs {xcol}", "data": points})
        total_points = len(points)

    # Determine if legend should be shown (multiple series)
    show_legend = bool(scol)

    # Build Chart.js config
    config = {
        "type": "scatter",
        "data": {"datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "legend": {"display": show_legend},
                "title": {"display": bool(intent.title), "text": intent.title or ""},
                "tooltip": {"enabled": True},
            },
            "scales": {
                "x": {
                    "type": "linear",
                    "title": {
                        "display": True,
                        "text": xcol,
                    },
                },
                "y": {
                    "type": "linear",
                    "title": {
                        "display": True,
                        "text": ycol,
                    },
                },
            },
        },
    }

    meta = ChartMeta(
        row_count=len(df),
        used_rows=len(df2),
        points=total_points,
        series=len(datasets),
        fields=fields,
    )

    return config, meta


def compile_chartjs(
    df: pd.DataFrame,
    intent: XYIntent | PieIntent | ScatterIntent,
) -> tuple[dict, list[ChartWarning], ChartMeta]:
    """
    Compile a Chart.js configuration from a DataFrame and chart intent.

    This is the main entry point for chart compilation. It handles row limiting,
    dispatches to the appropriate chart-type compiler, and formats the response.

    Args:
        df: Source pandas DataFrame with query results.
        intent: Chart intent specification (XYIntent, PieIntent, or ScatterIntent).

    Returns:
        Tuple of (Chart.js config dict, list of warnings, metadata).

    Raises:
        ValueError: If required columns are missing from the DataFrame.
    """
    warnings: list[dict] = []
    original_row_count = len(df)

    # Apply row limit if needed
    if len(df) > MAX_ROWS_TO_PROCESS:
        df = df.head(MAX_ROWS_TO_PROCESS)
        warnings.append(
            {
                "code": "ROW_LIMIT",
                "message": f"Processed first {MAX_ROWS_TO_PROCESS} rows of {original_row_count} total.",
            }
        )
    elif len(df) == MAX_ROWS_TO_PROCESS:
        # This commonly happens when the upstream query is already limited.
        warnings.append(
            {
                "code": "ROW_LIMIT",
                "message": (
                    f"Chart input is capped at {MAX_ROWS_TO_PROCESS} rows for performance; "
                    "additional rows may be omitted."
                ),
            }
        )

    # Lowercase the column names of the DataFrame
    # This is done to ensure that the column names are consistent with the intent
    # The column names related to the intent are also converted in the individual models
    df.columns = df.columns.str.lower()

    # Dispatch to appropriate compiler
    if isinstance(intent, XYIntent):
        config, meta = _compile_xy_chart(df, intent, warnings)
    elif isinstance(intent, PieIntent):
        config, meta = _compile_pie_chart(df, intent, warnings)
    elif isinstance(intent, ScatterIntent):
        config, meta = _compile_scatter_chart(df, intent, warnings)
    else:
        raise ValueError(f"Unknown intent type: {type(intent)}")

    # Update meta with original row count (before truncation)
    meta.row_count = original_row_count

    # Convert warning dicts to ChartWarning objects
    warning_objects = [ChartWarning(**w) for w in warnings]

    return config, warning_objects, meta
