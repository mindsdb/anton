"""
Chart intent schemas for the Chart API.

This module contains Pydantic v2 models for chart intent requests and responses.
The LLM outputs a small "intent" JSON, which the backend compiles into a full
Chart.js configuration using the message's SQL query results.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Type aliases for clarity
Aggregation = Literal["sum", "avg", "count", "min", "max"]
ChartType = Literal["bar", "line", "pie", "scatter"]


class BaseIntent(BaseModel):
    """Base model for chart intents with common fields."""

    type: ChartType = Field(..., description="Chart type")
    title: str | None = Field(default=None, description="Chart title")
    aggregate: Aggregation | None = Field(
        default=None,
        description="Aggregation function to apply (default: sum)",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=10000,
        description=(
            "Maximum data points to display. Defaults:  365 for time-series, "
            "50 for categorical XY charts, 12 for pie, 1000 for scatter. "
            'If the user asks for all/full data, set "limit" to a high value (e.g., 10000); '
            "the backend will still cap processing via CHART_COMPILER__MAX_ROWS_TO_PROCESS."
        ),
    )

    model_config = {"extra": "forbid"}  # Reject unknown keys


class XYIntent(BaseIntent):
    """Intent for bar and line charts with X/Y axes."""

    type: Literal["bar", "line"] = Field(..., description="Chart type (bar or line)")
    x: str = Field(..., description="Column name for X-axis (category or time)")
    y: str | list[str] = Field(
        ...,
        description="Column name(s) for Y-axis. Use a list to plot multiple metrics as separate series.",
    )
    series: str | None = Field(
        default=None,
        description="Column name to split into multiple datasets/series (ignored if y is a list)",
    )
    max_series: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum number of series to display (default: 12)",
    )

    @field_validator("x", "y", "series", mode="before")
    def lowercase_fields(cls, v: str) -> str:
        if isinstance(v, str):
            return v.lower()
        return v


class PieIntent(BaseIntent):
    """Intent for pie charts with label/value."""

    type: Literal["pie"] = Field(..., description="Chart type (pie)")
    label: str = Field(..., description="Column name for category labels")
    value: str = Field(..., description="Column name for numeric values")

    @field_validator("label", "value", mode="before")
    def lowercase_fields(cls, v: str) -> str:
        if isinstance(v, str):
            return v.lower()
        return v


class ScatterIntent(BaseIntent):
    """Intent for scatter charts with two numeric axes."""

    type: Literal["scatter"] = Field(..., description="Chart type (scatter)")
    x: str = Field(..., description="Column name for X-axis (numeric)")
    y: str = Field(..., description="Column name for Y-axis (numeric)")
    series: str | None = Field(
        default=None,
        description="Column name to split into multiple datasets/series",
    )
    max_series: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum number of series to display (default: 12)",
    )

    @field_validator("x", "y", "series", mode="before")
    def lowercase_fields(cls, v: str) -> str:
        if isinstance(v, str):
            return v.lower()
        return v


# Discriminated union for chart intent
ChartIntent = XYIntent | PieIntent | ScatterIntent


class RenderChartType(str, Enum):
    """Compiled chart kind for RenderPlan (Chart.js `type` string values)."""

    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"


class ChartOutputFormat(str, Enum):
    CHARTJS = "chartjs"
    PNG = "png"
    IMAGE_URL = "image_url"


class ChartRequest(BaseModel):
    """Request model for the unified chart endpoint."""

    intent: ChartIntent = Field(
        ...,
        discriminator="type",
        description="Chart intent specification",
    )
    output: ChartOutputFormat = Field(
        default=ChartOutputFormat.CHARTJS,
        description=(
            "Desired output format. "
            "'chartjs' returns a Chart.js configuration for frontend rendering. "
            "'png' returns server-rendered PNG bytes directly. "
            "'image_url' returns a URL to an authenticated server-rendered image endpoint."
        ),
    )


@dataclass(frozen=True)
class AxisSpec:
    title: str | None
    scale_type: str


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    values: list[float] | None = None
    points: list[tuple[float, float]] | None = None


@dataclass(frozen=True)
class RenderPlan:
    chart_type: RenderChartType
    title: str | None
    show_legend: bool
    labels: list[Any]
    series: list[SeriesSpec]
    x_axis: AxisSpec
    y_axis: AxisSpec


class ChartWarning(BaseModel):
    """Warning message from chart compilation."""

    code: str = Field(..., description="Warning code (e.g., TRUNCATED, ROW_LIMIT)")
    message: str = Field(..., description="Human-readable warning message")


class ChartMeta(BaseModel):
    """Metadata about the chart compilation."""

    row_count: int = Field(..., description="Total rows in the source data")
    used_rows: int = Field(..., description="Number of rows processed after limits")
    points: int = Field(..., description="Number of data points in the chart")
    series: int = Field(..., description="Number of series/datasets in the chart")
    fields: dict[str, str] | None = Field(
        default=None,
        description="Field type information (e.g., temporal, quantitative, nominal)",
    )


class ChartResponse(BaseModel):
    """Response model for chart generation endpoint."""

    format: Literal["chartjs"] = Field(
        default="chartjs",
        description="Chart library format",
    )
    config: dict = Field(..., description="Complete Chart.js configuration")
    meta: ChartMeta = Field(..., description="Metadata about the chart")
    warnings: list[ChartWarning] = Field(
        default_factory=list,
        description="Warnings from chart compilation",
    )


class ChartImageResponse(BaseModel):
    """Response model for server-rendered chart image URLs."""

    image_url: str = Field(
        ...,
        description="Opaque authenticated URL for fetching the rendered chart image",
    )
    meta: ChartMeta = Field(..., description="Metadata about the chart")
    warnings: list[ChartWarning] = Field(
        default_factory=list,
        description="Warnings from chart compilation",
    )


class ChartImageTokenPayload(BaseModel):
    """Opaque payload embedded in an image URL token."""

    intent: ChartIntent = Field(
        ...,
        discriminator="type",
        description="Chart intent specification",
    )
