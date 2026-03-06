from unittest.mock import patch

import pandas as pd
import pytest

from minds.schemas.charts import PieIntent, ScatterIntent, XYIntent
from minds.services.chart_compiler import (
    DEFAULT_LIMIT_TEMPORAL,
    MAX_ROWS_TO_PROCESS,
    _aggregate_series,
    _is_temporal,
    _to_datetime_if_temporal,
    compile_chartjs,
)


def test_is_temporal_guards_numeric_and_detects_datetime_and_strings():
    assert _is_temporal(pd.Series([1, 2, 3])) is False
    assert _is_temporal(pd.to_datetime(pd.Series(["2020-01-01", "2020-01-02"]))) is True
    assert _is_temporal(pd.Series(["2020-01-01", "2020-01-02", "not-a-date", "2020-01-03", "2020-01-04"])) is True
    assert _is_temporal(pd.Series(["not-a-date", "also-bad"])) is False
    assert _is_temporal(pd.Series([None, None])) is False


def test_is_temporal_handles_datetime_parse_exceptions():
    # Force the internal datetime parse call to raise.
    with patch("minds.services.chart_compiler.pd.to_datetime", side_effect=RuntimeError("boom")):
        assert _is_temporal(pd.Series(["2020-01-01", "2020-01-02"])) is False


def test_aggregate_series_supports_known_aggs_and_defaults_to_sum():
    df = pd.DataFrame({"k": ["a", "a", "b"], "v": [1, 2, 3]})
    grouped = df.groupby("k")["v"]
    assert _aggregate_series(grouped, "sum").to_dict() == {"a": 3, "b": 3}
    assert _aggregate_series(grouped, "avg").to_dict() == {"a": 1.5, "b": 3.0}
    assert _aggregate_series(grouped, "min").to_dict() == {"a": 1, "b": 3}
    assert _aggregate_series(grouped, "max").to_dict() == {"a": 2, "b": 3}
    assert _aggregate_series(grouped, "count").to_dict() == {"a": 2, "b": 1}
    assert _aggregate_series(grouped, "nope").to_dict() == {"a": 3, "b": 3}


def test_compile_chartjs_xy_multi_y_ignores_series_and_truncates_categories():
    df = pd.DataFrame(
        {
            "X": [f"c{i}" for i in range(30)],
            "Y1": list(range(30)),
            "Y2": list(range(30)),
            "S": ["s"] * 30,
        }
    )
    intent = XYIntent(type="bar", x="x", y=["y1", "y2"], series="s", limit=5)
    config, warnings, meta = compile_chartjs(df, intent)

    assert config["type"] == "bar"
    assert len(config["data"]["datasets"]) == 2
    assert any(w.code == "SERIES_IGNORED" for w in warnings)
    assert any(w.code == "TRUNCATED" for w in warnings)
    assert meta.points <= 5


def test_compile_chartjs_xy_unknown_series_falls_back_to_single_series():
    df = pd.DataFrame({"X": ["a", "b"], "Y": [1, 2]})
    intent = XYIntent(type="line", x="x", y="y", series="missing")
    config, warnings, meta = compile_chartjs(df, intent)

    assert config["type"] == "line"
    assert any(w.code == "UNKNOWN_SERIES" for w in warnings)
    assert meta.series == 1


def test_compile_chartjs_pie_truncates_and_adds_other_bucket():
    df = pd.DataFrame({"label": [f"c{i}" for i in range(20)], "value": [1] * 20})
    intent = PieIntent(type="pie", label="label", value="value", limit=3)
    config, warnings, meta = compile_chartjs(df, intent)

    assert config["type"] == "pie"
    assert any(w.code == "TRUNCATED" for w in warnings)
    assert "Other" in config["data"]["labels"]
    assert meta.series == 1


def test_compile_chartjs_scatter_limits_points_and_truncates_series():
    series_vals = (["a", "b", "c"] * 10)[:30]
    df = pd.DataFrame(
        {
            "x": list(range(30)),
            "y": list(range(30)),
            "series": series_vals,
        }
    )
    intent = ScatterIntent(type="scatter", x="x", y="y", series="series", limit=5, max_series=1)
    config, warnings, meta = compile_chartjs(df, intent)

    assert config["type"] == "scatter"
    assert any(w.code == "TRUNCATED" for w in warnings)
    assert any(w.code == "SERIES_TRUNCATED" for w in warnings)
    assert len(config["data"]["datasets"]) == 1
    assert meta.points <= 5


def test_compile_chartjs_xy_with_series_truncates_and_handles_numeric_x():
    # Numeric X uses "first N values" truncation path
    df = pd.DataFrame(
        {
            "x": list(range(20)) * 3,
            "y": [1] * 60,
            "series": ["s1"] * 20 + ["s2"] * 20 + ["s3"] * 20,
        }
    )
    intent = XYIntent(type="bar", x="x", y="y", series="series", limit=3, max_series=2)
    config, warnings, meta = compile_chartjs(df, intent)
    assert any(w.code == "SERIES_TRUNCATED" for w in warnings)
    assert any(w.code == "TRUNCATED" for w in warnings)
    assert len(config["data"]["datasets"]) <= 2
    assert meta.points == 3


def test_compile_chartjs_xy_temporal_keeps_most_recent_points():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"x": dates.astype(str), "y": list(range(10))})
    intent = XYIntent(type="line", x="x", y="y", limit=3)
    config, warnings, meta = compile_chartjs(df, intent)
    assert any(w.code == "TRUNCATED" for w in warnings)
    # last label should be the most recent date
    assert config["data"]["labels"][-1] == dates[-1].date().isoformat()
    assert meta.points == 3


def test_compile_chartjs_xy_temporal_default_truncates_to_default_limit_when_limit_unset():
    periods = min(DEFAULT_LIMIT_TEMPORAL + 10, MAX_ROWS_TO_PROCESS)
    dates = pd.date_range("2020-01-01", periods=periods, freq="H")
    df = pd.DataFrame({"x": dates.astype(str), "y": list(range(periods))})
    intent = XYIntent(type="line", x="x", y="y")
    config, warnings, meta = compile_chartjs(df, intent)

    if periods > DEFAULT_LIMIT_TEMPORAL:
        assert any(w.code == "TRUNCATED" for w in warnings)
        assert meta.points == DEFAULT_LIMIT_TEMPORAL
        assert len(config["data"]["labels"]) == DEFAULT_LIMIT_TEMPORAL
        # last label should still be the most recent date
        assert config["data"]["labels"][-1] == dates[-1].date().isoformat()
    else:
        assert not any(w.code == "TRUNCATED" for w in warnings)
        assert meta.points == periods
        assert len(config["data"]["labels"]) == periods


def test_compile_chartjs_xy_temporal_limit_high_shows_all_points_up_to_processing_cap():
    periods = min(DEFAULT_LIMIT_TEMPORAL + 10, MAX_ROWS_TO_PROCESS)
    dates = pd.date_range("2020-01-01", periods=periods, freq="H")
    df = pd.DataFrame({"x": dates.astype(str), "y": list(range(periods))})
    intent = XYIntent(type="line", x="x", y="y", limit=10000)
    config, warnings, meta = compile_chartjs(df, intent)
    assert not any(w.code == "TRUNCATED" for w in warnings)
    assert meta.points == periods
    assert len(config["data"]["labels"]) == periods


def test_compile_chartjs_xy_multi_y_temporal_and_numeric_x_truncation_paths():
    # multi-y + temporal x -> "most recent points" truncation path
    dates = pd.date_range("2020-01-01", periods=10, freq="D").astype(str)
    df_time = pd.DataFrame({"x": dates, "y1": list(range(10)), "y2": list(range(10))})
    config, warnings, meta = compile_chartjs(df_time, XYIntent(type="line", x="x", y=["y1", "y2"], limit=3))
    assert any(w.code == "TRUNCATED" for w in warnings)
    assert meta.points == 3

    # multi-y + numeric x -> "first N values" truncation path
    df_num = pd.DataFrame({"x": list(range(10)), "y1": list(range(10)), "y2": list(range(10))})
    config2, warnings2, meta2 = compile_chartjs(df_num, XYIntent(type="bar", x="x", y=["y1", "y2"], limit=3))
    assert any(w.code == "TRUNCATED" for w in warnings2)
    assert meta2.points == 3


def test_compile_chartjs_xy_single_y_categorical_truncation_path():
    df = pd.DataFrame({"x": [f"c{i}" for i in range(10)], "y": list(range(10))})
    _config, warnings, meta = compile_chartjs(df, XYIntent(type="bar", x="x", y="y", limit=3))
    assert any(w.code == "TRUNCATED" for w in warnings)
    assert meta.points == 3


def test_compile_chartjs_validates_missing_columns_for_each_intent():
    with pytest.raises(ValueError, match="Unknown column"):
        compile_chartjs(pd.DataFrame({"x": [1]}), XYIntent(type="bar", x="x", y="y"))

    with pytest.raises(ValueError, match="Unknown column"):
        compile_chartjs(pd.DataFrame({"label": ["a"]}), PieIntent(type="pie", label="label", value="value"))

    with pytest.raises(ValueError, match="Unknown column"):
        compile_chartjs(pd.DataFrame({"x": [1]}), ScatterIntent(type="scatter", x="x", y="y"))


def test_compile_chartjs_applies_row_limit_and_rejects_unknown_intent_type():
    df = pd.DataFrame({"x": list(range(MAX_ROWS_TO_PROCESS + 1)), "y": [1] * (MAX_ROWS_TO_PROCESS + 1)})
    intent = XYIntent(type="bar", x="x", y="y")
    _config, warnings, meta = compile_chartjs(df, intent)
    assert any(w.code == "ROW_LIMIT" for w in warnings)
    assert meta.row_count == MAX_ROWS_TO_PROCESS + 1

    with pytest.raises(ValueError, match="Unknown intent type"):
        compile_chartjs(pd.DataFrame({"x": [1], "y": [1]}), object())


def test_to_datetime_if_temporal_converts_when_parseable():
    s, is_time = _to_datetime_if_temporal(pd.Series(["2020-01-01", "2020-01-02"]))
    assert is_time is True
    assert str(s.dtype).startswith("datetime64")

    s2, is_time2 = _to_datetime_if_temporal(pd.Series([1, 2, 3]))
    assert is_time2 is False


def test_compile_chartjs_scatter_unknown_series_falls_back_single_series_and_uses_title_label():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    intent = ScatterIntent(type="scatter", x="x", y="y", series="missing", title="T", limit=10)
    config, warnings, meta = compile_chartjs(df, intent)
    assert any(w.code == "UNKNOWN_SERIES" for w in warnings)
    assert config["data"]["datasets"][0]["label"] == "T"
    assert meta.series == 1
