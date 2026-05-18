"""
Server-side chart renderer using Matplotlib.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from minds.common.logger import get_logger
from minds.common.settings.app_settings import get_app_settings
from minds.schemas.charts import AxisSpec, RenderChartType, RenderPlan, SeriesSpec

logger = get_logger(__name__)

_settings = get_app_settings()
IMAGE_WIDTH = _settings.chart_renderer.image_width
IMAGE_HEIGHT = _settings.chart_renderer.image_height
IMAGE_DPI = _settings.chart_renderer.image_dpi

# Re-export for backward compatibility
__all__ = ["AxisSpec", "RenderPlan", "SeriesSpec", "render_chart_image"]


def render_chart_image(
    render_plan: RenderPlan,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
) -> bytes:
    """
    Render a RenderPlan as a PNG image via Matplotlib.

    Args:
        render_plan: A renderer-agnostic chart specification.
        width: Image width in pixels.
        height: Image height in pixels.
    Returns:
        PNG image bytes.
    """
    return _render_plan_with_matplotlib(render_plan, width=width, height=height)


def _render_plan_with_matplotlib(render_plan: RenderPlan, width: int, height: int) -> bytes:
    """Render a normalized chart plan to PNG bytes using Matplotlib Agg."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(
        figsize=(max(width, 1) / IMAGE_DPI, max(height, 1) / IMAGE_DPI),
        dpi=IMAGE_DPI,
        facecolor="white",
    )
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.9 if render_plan.title else 0.96, bottom=0.18)

    if render_plan.chart_type in (RenderChartType.BAR, RenderChartType.LINE):
        _draw_xy_matplotlib(axis, render_plan)
    elif render_plan.chart_type == RenderChartType.PIE:
        _draw_pie_matplotlib(axis, render_plan)
    else:
        _draw_scatter_matplotlib(axis, render_plan)

    if render_plan.title:
        axis.set_title(render_plan.title)

    if render_plan.chart_type != RenderChartType.PIE:
        if render_plan.x_axis.title:
            axis.set_xlabel(render_plan.x_axis.title)
        if render_plan.y_axis.title:
            axis.set_ylabel(render_plan.y_axis.title)
        axis.grid(axis="y", alpha=0.2)

    if render_plan.show_legend and render_plan.chart_type != RenderChartType.PIE:
        labeled_series = [series for series in render_plan.series if series.label]
        if labeled_series:
            axis.legend()

    buffer = BytesIO()
    figure.savefig(buffer, format="png", facecolor=figure.get_facecolor())
    figure.clear()
    return buffer.getvalue()


def _draw_xy_matplotlib(axis: Any, render_plan: RenderPlan) -> None:
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    labels = render_plan.labels
    x_values, is_date_axis = _resolve_xy_x_values(labels, render_plan.x_axis.scale_type)

    if render_plan.chart_type == RenderChartType.BAR:
        positions = list(range(len(labels)))
        series_count = max(len(render_plan.series), 1)
        total_width = 0.8
        bar_width = total_width / series_count

        for index, series in enumerate(render_plan.series):
            offset = (index - (series_count - 1) / 2) * bar_width
            axis.bar(
                [position + offset for position in positions],
                series.values or [],
                width=bar_width,
                label=series.label or None,
            )

        axis.set_xticks(positions)
        axis.set_xticklabels([str(label) for label in labels], rotation=30, ha="right")
        return

    for series in render_plan.series:
        axis.plot(x_values, series.values or [], label=series.label or None, linewidth=2)

    if is_date_axis:
        locator = AutoDateLocator()
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        axis.figure.autofmt_xdate()
    elif any(not isinstance(label, int | float) for label in labels):
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels([str(label) for label in labels], rotation=30, ha="right")


def _draw_pie_matplotlib(axis: Any, render_plan: RenderPlan) -> None:
    values = render_plan.series[0].values if render_plan.series else []
    axis.pie(values, labels=[str(label) for label in render_plan.labels], startangle=90)
    axis.axis("equal")


def _draw_scatter_matplotlib(axis: Any, render_plan: RenderPlan) -> None:
    for series in render_plan.series:
        points = series.points or []
        axis.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            label=series.label or None,
            alpha=0.85,
        )


def _resolve_xy_x_values(labels: list[Any], scale_type: str) -> tuple[list[Any], bool]:
    if scale_type == "timeseries":
        import pandas as pd
        from matplotlib.dates import date2num

        parsed = pd.to_datetime(pd.Series(labels), errors="coerce", utc=True)
        if parsed.notna().all():
            return date2num(parsed.dt.tz_convert(None).tolist()), True

    if all(isinstance(label, int | float) for label in labels):
        return labels, False

    return list(range(len(labels))), False
