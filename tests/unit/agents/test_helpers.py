import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _load_real_helpers():
    sys.modules.pop("minds.agents.helpers", None)
    return importlib.import_module("minds.agents.helpers")


def test_model_for_uses_llm_config():
    helpers = _load_real_helpers()
    mind = Mock(provider="p", model_name="m")
    with patch("minds.agents.helpers.get_llm_config", return_value="cfg") as get_cfg:
        assert helpers.model_for(mind) == "cfg"
        get_cfg.assert_called_once_with("p", "m")


def test_mind_layer_prefers_system_prompt_then_prompt_template():
    helpers = _load_real_helpers()
    mind = Mock(parameters={"system_prompt": "S", "prompt_template": "T"})
    assert helpers.mind_layer(mind) == "S"
    mind = Mock(parameters={"prompt_template": "T"})
    assert helpers.mind_layer(mind) == "T"
    mind = Mock(parameters={})
    assert helpers.mind_layer(mind) == ""


def test_charting_layer_returns_instructions():
    helpers = _load_real_helpers()
    assert "chart" in helpers.charting_layer().lower()


def test_current_date_time_layer_format():
    helpers = _load_real_helpers()
    text = helpers.current_date_time_layer()
    assert text.startswith("The current date and time is ")
    assert len(text) > 30


def test_is_native_query_mode_enabled_rules():
    helpers = _load_real_helpers()
    settings = SimpleNamespace(
        use_native_query_mode=True,
        native_query_mode_supported_engines=["snowflake", "bigquery"],
    )
    mind = Mock(
        parameters={},
        mind_datasources=[SimpleNamespace(datasource=SimpleNamespace(engine="snowflake"))],
    )
    assert helpers.is_native_query_mode_enabled(mind, settings) is True

    mind.parameters = {"use_native_query_mode": False}
    assert helpers.is_native_query_mode_enabled(mind, settings) is False

    mind.parameters = {}
    mind.mind_datasources = [
        SimpleNamespace(datasource=SimpleNamespace(engine="snowflake")),
        SimpleNamespace(datasource=SimpleNamespace(engine="bigquery")),
    ]
    assert helpers.is_native_query_mode_enabled(mind, settings) is False

    mind.mind_datasources = [SimpleNamespace(datasource=SimpleNamespace(engine="duckdb"))]
    assert helpers.is_native_query_mode_enabled(mind, settings) is False
