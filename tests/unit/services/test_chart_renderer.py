import pytest

from minds.schemas.charts import AxisSpec, RenderPlan, SeriesSpec
from minds.services.chart_renderer import render_chart_image

PNG_HEADER = b"\x89PNG\r\n\x1a\n"


@pytest.mark.unit
class TestRenderChartImage:
    def test_returns_png_bytes_for_bar(self):
        plan = RenderPlan(
            chart_type="bar",
            title="Bar Chart",
            show_legend=False,
            labels=["a", "b"],
            series=[SeriesSpec(label="v", values=[1, 2])],
            x_axis=AxisSpec(title="category", scale_type="category"),
            y_axis=AxisSpec(title="count", scale_type="linear"),
        )

        result = render_chart_image(plan, width=800, height=400)

        assert result.startswith(PNG_HEADER)
        assert len(result) > len(PNG_HEADER)

    def test_returns_png_bytes_for_pie(self):
        plan = RenderPlan(
            chart_type="pie",
            title="Regions",
            show_legend=True,
            labels=["North", "South", "East"],
            series=[SeriesSpec(label="", values=[40, 35, 25])],
            x_axis=AxisSpec(title=None, scale_type="category"),
            y_axis=AxisSpec(title=None, scale_type="linear"),
        )

        result = render_chart_image(plan, width=800, height=400)

        assert result.startswith(PNG_HEADER)
        assert len(result) > len(PNG_HEADER)

    def test_returns_png_bytes_for_scatter(self):
        plan = RenderPlan(
            chart_type="scatter",
            title=None,
            show_legend=True,
            labels=[],
            series=[SeriesSpec(label="points", points=[(1, 2), (3, 4)])],
            x_axis=AxisSpec(title="x", scale_type="linear"),
            y_axis=AxisSpec(title="y", scale_type="linear"),
        )

        result = render_chart_image(plan, width=800, height=400)

        assert result.startswith(PNG_HEADER)
        assert len(result) > len(PNG_HEADER)

    def test_returns_png_bytes_for_line_with_string_labels(self):
        plan = RenderPlan(
            chart_type="line",
            title=None,
            show_legend=True,
            labels=["Jan", "Feb", "Mar"],
            series=[SeriesSpec(label="revenue", values=[100, 200, 150])],
            x_axis=AxisSpec(title="month", scale_type="category"),
            y_axis=AxisSpec(title="amount", scale_type="linear"),
        )

        result = render_chart_image(plan, width=800, height=400)

        assert result.startswith(PNG_HEADER)
        assert len(result) > len(PNG_HEADER)

    def test_returns_png_bytes_for_timeseries_line(self):
        plan = RenderPlan(
            chart_type="line",
            title=None,
            show_legend=True,
            labels=["2024-01-01", "2024-01-02", "2024-01-03"],
            series=[SeriesSpec(label="daily", values=[10, 20, 30])],
            x_axis=AxisSpec(title="date", scale_type="timeseries"),
            y_axis=AxisSpec(title="value", scale_type="linear"),
        )

        result = render_chart_image(plan, width=800, height=400)

        assert result.startswith(PNG_HEADER)
        assert len(result) > len(PNG_HEADER)

    def test_returns_png_bytes_for_numeric_x_line(self):
        plan = RenderPlan(
            chart_type="line",
            title=None,
            show_legend=False,
            labels=[1, 2, 3, 4],
            series=[SeriesSpec(label="values", values=[10, 20, 15, 25])],
            x_axis=AxisSpec(title="step", scale_type="category"),
            y_axis=AxisSpec(title="value", scale_type="linear"),
        )

        result = render_chart_image(plan, width=800, height=400)

        assert result.startswith(PNG_HEADER)
        assert len(result) > len(PNG_HEADER)
